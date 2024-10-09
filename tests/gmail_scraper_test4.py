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

    def fetch_emails(self, label_ids):
        """
        Fetch emails with the specified label IDs using pagination to get all the emails.

        :param label_ids: List of Gmail label IDs (e.g., 'INBOX', 'SENT')
        :return: A list of email data
        """
        try:
            email_data = []
            next_page_token = None

            while True:
                # Fetch a batch of messages, using nextPageToken if available
                result = self.service.users().messages().list(
                    userId='me',
                    labelIds=label_ids,
                    pageToken=next_page_token
                ).execute()

                messages = result.get('messages', [])
                for msg in tqdm(messages, desc="Fetching Emails"):
                    msg_data = self.service.users().messages().get(userId='me', id=msg['id']).execute()
                    email_data.append(msg_data)

                # Update the page token to get the next page
                next_page_token = result.get('nextPageToken')
                
                if not next_page_token:
                    break  # Exit the loop when no more pages are available

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

    def categorize_emails(self):
        """
        Categorize emails into replied and unreplied.

        :return: A tuple of two lists: (replied_emails, unreplied_emails)
        """
        # Fetch emails from INBOX and SENT labels
        inbox_emails = self.gmail_api_client.fetch_emails(['INBOX'])
        sent_emails = self.gmail_api_client.fetch_emails(['SENT'])

        # Extract message IDs from sent emails
        sent_message_ids = set()
        for data in tqdm(sent_emails, desc="Processing Sent Emails"):
            payload = data['payload']
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
            payload = data['payload']
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

    def save_emails(self, emails, folder_path):
        """Save emails in JSON format in the specified folder."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        for idx, email_data in enumerate(tqdm(emails, desc=f"Saving Emails in {folder_path}")):
            email_file = os.path.join(folder_path, f'email_{idx + 1}.json')
            with open(email_file, 'w', encoding='utf-8') as f:
                json.dump(email_data, f, ensure_ascii=False, indent=4)

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

        # Categorize emails
        replied, unreplied = categorizer.categorize_emails()
        print('DEBUG: entered categorizer and stored mail')

        # Save categorized emails
        categorizer.save_emails(replied, 'data/replied')
        categorizer.save_emails(unreplied, 'data/unreplied')

        # Print results
        print("Replied Emails:")
        for msg_data in tqdm(replied, desc="Printing Replied Emails"):
            try:
                headers = msg_data['payload']['headers']
                from_header = next(header['value'] for header in headers if header['name'] == 'From')
                subject_header = next(header['value'] for header in headers if header['name'] == 'Subject')
                print(f"From: {from_header}, Subject: {subject_header}")
            except Exception as e:
                print(f"An error occured in the main body (Replied Emails): {e}")
                traceback.print_exc()

        print("\nUnreplied Emails:")
        for msg_data in tqdm(unreplied, desc="Printing Unreplied Emails"):
            try:
                headers = msg_data['payload']['headers']
                # Find the 'From' header, or set to 'Unknown' if not present
                from_header = next((header['value'] for header in headers if header['name'] == 'From'), "Unknown")
                # Find the 'Subject' header, or set to 'No Subject' if not present
                subject_header = next((header['value'] for header in headers if header['name'] == 'Subject'), "No Subject")
                print(f"From: {from_header}, Subject: {subject_header}")
            except Exception as e:
                print(f"An error occurred in the main body (Unreplied Emails): {e}")

    except Exception as e:
        print(f"An error occurred in the main body: {e}")

    print(f'END OF PROGRAM. REPLIED EMAILS: {len(replied)},  UNREPLIED EMAILS: {len(unreplied)}')