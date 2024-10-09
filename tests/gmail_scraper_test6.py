from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import email
import json
import os
import traceback
from tqdm import tqdm

class GmailAPIClient:
    def __init__(self, credentials_file):
        """
        Initialize the GmailAPIClient to handle OAuth authentication.

        :param credentials_file: Path to the OAuth 2.0 credentials file
        """
        self.credentials_file = credentials_file
        self.creds = None
        self.service = None

    def authenticate(self):
        """Authenticate using OAuth and store credentials."""
        flow = InstalledAppFlow.from_client_secrets_file(
            self.credentials_file,
            scopes=['https://www.googleapis.com/auth/gmail.readonly']
        )
        self.creds = flow.run_local_server(port=0)
        self.service = build('gmail', 'v1', credentials=self.creds)

    def fetch_emails(self, label_ids, first_page_only=False, max_pages=None):
        """
        Fetch emails with the specified label IDs using pagination to get all the emails.

        :param label_ids: List of Gmail label IDs (e.g., 'INBOX', 'SENT')
        :param first_page_only: Boolean flag to control fetching only the first page
        :param max_pages: Maximum number of pages to fetch (None for no limit)
        :return: A list of email data
        """
        try:
            email_data = []
            next_page_token = None
            page_count = 0  # Counter for the number of pages fetched
            total_messages_fetched = 0
            while True:
                # Fetch a batch of messages, using nextPageToken if available
                result = self.service.users().messages().list(
                    userId='me',
                    labelIds=label_ids,
                    pageToken=next_page_token,
                    maxResults=250
                ).execute()

                messages = result.get('messages', [])
                #print(f"Fetched {len(messages)} messages from labels {label_ids}")  
                total_messages_fetched += len(messages)  
                
                for msg in tqdm(messages, desc="Fetching Emails"):
                    try:
                        msg_data = self.service.users().messages().get(
                            userId='me',
                            id=msg['id'],
                            format='full'
                        ).execute()
                        email_data.append(msg_data)  # Keep the full message data for categorization
                    except HttpError as error:
                        # Handle the specific error and continue
                        if error.resp.status in [400, 404]:
                            print(f"Skipping message {msg['id']} due to error: {error}")
                            continue
                        else:
                            # Re-raise the error if it's not one we expect
                            raise

                # Update the page token to get the next page
                next_page_token = result.get('nextPageToken')
                page_count += 1  # Increment the page counter

                # Break the loop based on conditions
                if first_page_only or not next_page_token:
                    break  # Exit the loop when only the first page is needed or no more pages are available

                if max_pages is not None and page_count >= max_pages:
                    print(f"Reached maximum page limit of {max_pages}. Stopping fetch.")
                    break  # Exit the loop when the maximum number of pages is reached
            
            print(f"Total messages fetched from labels {label_ids}: {total_messages_fetched}")  # Add this line
            return email_data
        except HttpError as error:
            print(f"An error occurred inside fetch_email: {error}")
            return []

class EmailCategorizer:
    def __init__(self, credentials_file):
        """
        Initialize the EmailCategorizer with Gmail API details and OAuth credentials.

        :param credentials_file: Path to the OAuth 2.0 credentials file
        """
        self.gmail_api_client = GmailAPIClient(credentials_file)

    def connect(self):
        """Establish a connection to the Gmail API using OAuth."""
        self.gmail_api_client.authenticate()

    def categorize_emails(self, first_page_only=False, max_pages=None):
        """
        Categorize emails into replied and unreplied.

        :param first_page_only: Boolean flag to control fetching only the first page
        :param max_pages: Maximum number of pages to fetch (None for no limit)
        :return: A tuple of two lists: (replied_emails, unreplied_emails)
        """
        # Fetch emails from INBOX and SENT labels
        inbox_emails = self.gmail_api_client.fetch_emails(['INBOX'], first_page_only=first_page_only, max_pages=max_pages)
        sent_emails = self.gmail_api_client.fetch_emails(['SENT'], first_page_only=first_page_only, max_pages=max_pages)

        # Extract message IDs from sent emails
        sent_message_ids = set()
        for data in tqdm(sent_emails, desc="Processing Sent Emails"):
            payload = data.get('payload', {})
            headers = payload.get('headers', [])
            in_reply_to = None
            message_id = None
            for header in headers:
                if header['name'] == 'In-Reply-To':
                    in_reply_to = header['value']
                elif header['name'] == 'Message-ID':
                    message_id = header['value']
            if in_reply_to:
                sent_message_ids.add(in_reply_to)
            elif message_id:
                sent_message_ids.add(message_id)

        # Categorize inbox emails
        replied_emails = []
        unreplied_emails = []

        for data in tqdm(inbox_emails, desc="Processing Inbox Emails"):
            payload = data.get('payload', {})
            headers = payload.get('headers', [])
            message_id = None
            for header in headers:
                if header['name'] == 'Message-ID':
                    message_id = header['value']
                    break

            if message_id in sent_message_ids:
                replied_emails.append(data)
            else:
                unreplied_emails.append(data)

        return replied_emails, unreplied_emails

    def extract_relevant_data(self, email_data):
        """Extract only the text body, attachment information, sender, subject, and time sent from the email."""
        payload = email_data.get('payload', {})
        headers = payload.get('headers', [])
        parts = payload.get('parts', [])
        has_attachments = any(part.get('filename') for part in parts)

        # Extract the text body
        text_body = ""
        # Handle nested parts (multipart/mixed, multipart/alternative)
        parts_to_extract = []
        if parts:
            parts_to_extract = parts
        elif payload.get('body', {}).get('data'):
            parts_to_extract = [payload]

        for part in parts_to_extract:
            mime_type = part.get('mimeType', '')
            if mime_type == 'text/plain':
                data = part.get('body', {}).get('data')
                if data:
                    missing_padding = len(data) % 4
                    if missing_padding:
                        data += '=' * (4 - missing_padding)
                    try:
                        text_body = base64.urlsafe_b64decode(data).decode('utf-8')
                        break  # Stop after finding the first text/plain part
                    except Exception as e:
                        print(f"Error decoding text body: {e}")
            elif mime_type == 'multipart/alternative' or mime_type == 'multipart/mixed':
                # Recursively handle nested parts
                sub_parts = part.get('parts', [])
                for sub_part in sub_parts:
                    if sub_part.get('mimeType') == 'text/plain':
                        data = sub_part.get('body', {}).get('data')
                        if data:
                            missing_padding = len(data) % 4
                            if missing_padding:
                                data += '=' * (4 - missing_padding)
                            try:
                                text_body = base64.urlsafe_b64decode(data).decode('utf-8')
                                break  # Stop after finding the first text/plain part
                            except Exception as e:
                                print(f"Error decoding nested text body: {e}")
                if text_body:
                    break  # Stop if text body is found

        # Extract sender, subject, and time sent from headers
        sender = next((header['value'] for header in headers if header['name'] == 'From'), "Unknown")
        subject = next((header['value'] for header in headers if header['name'] == 'Subject'), "No Subject")
        date = next((header['value'] for header in headers if header['name'] == 'Date'), "Unknown Date")

        # Also keep essential headers for future reference if needed
        message_id = next((header['value'] for header in headers if header['name'] == 'Message-ID'), None)
        in_reply_to = next((header['value'] for header in headers if header['name'] == 'In-Reply-To'), None)

        return {
            'sender': sender,
            'subject': subject,
            'date': date,
            'text_body': text_body,
            'has_attachments': has_attachments,
            'message_id': message_id,
            'in_reply_to': in_reply_to
        }

    def save_emails(self, emails, folder_path):
        """Save emails in JSON format in the specified folder."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for idx, email_data in enumerate(tqdm(emails, desc=f"Saving Emails in {folder_path}")):
            relevant_data = self.extract_relevant_data(email_data)  # Extract relevant data before saving
            email_file = os.path.join(folder_path, f'email_{idx + 1}.json')
            with open(email_file, 'w', encoding='utf-8') as f:
                json.dump(relevant_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Replace with your OAuth credentials file
    CREDENTIALS_FILE = 'client_secret.json'

    # Create an instance of the EmailCategorizer
    categorizer = EmailCategorizer(CREDENTIALS_FILE)
    print('DEBUG: created categorizer')
    try:
        # Connect to the Gmail API using OAuth
        categorizer.connect()
        print('DEBUG: connected categorizer')

        # Set first_page_only to True for debugging purposes, False for full extraction
        first_page_only = False

        # Set the maximum number of pages to fetch
        max_pages = 6  # Change this value as needed; set to None for no limit

        # Categorize emails
        replied, unreplied = categorizer.categorize_emails(first_page_only=first_page_only, max_pages=max_pages)
        print('DEBUG: entered categorizer and stored mail')

        # Save categorized emails
        categorizer.save_emails(replied, 'data/replied')
        categorizer.save_emails(unreplied, 'data/unreplied')

        # Print results
        print(f'END OF PROGRAM. REPLIED EMAILS: {len(replied)},  UNREPLIED EMAILS: {len(unreplied)}')

    except Exception as e:
        print(f"An error occurred in the main body: {e}")