from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import email
import traceback

# RETRIEVES JUST 1 PAGE OF EMALS, 0 REPLIED EMAILS, 

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
        Fetch emails with the specified label IDs.

        :param label_ids: List of Gmail label IDs (e.g., 'INBOX', 'SENT')
        :return: A list of email data
        """
        try:
            result = self.service.users().messages().list(userId='me', labelIds=label_ids).execute()
            messages = result.get('messages', [])

            email_data = []
            for msg in messages:
                msg_data = self.service.users().messages().get(userId='me', id=msg['id']).execute()
                email_data.append(msg_data)

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
        for data in sent_emails:
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

        for data in inbox_emails:
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

    def decode_message(self, message):
        """Decode a Gmail message into a readable format."""
        msg_str = base64.urlsafe_b64decode(message['payload']['body']['data'].encode('ASCII')).decode('utf-8')
        msg = email.message_from_string(msg_str)
        return msg

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
        print('DEBUG: entered categorizer and sotred mail')

        # Print results
        print("Replied Emails:")
        for msg_data in replied:
            try:
                headers = msg_data['payload']['headers']
                from_header = next(header['value'] for header in headers if header['name'] == 'From')
                subject_header = next(header['value'] for header in headers if header['name'] == 'Subject')
                print(f"From: {from_header}, Subject: {subject_header}")
            except Exception as e:
                print(f"An error occured in the main body (Replied Emails): {e}")
                traceback.print_exc()


        print("\nUnreplied Emails:")
        for msg_data in unreplied:
            try:
                headers = msg_data['payload']['headers']
                from_header = next(header['value'] for header in headers if header['name'] == 'From')
                subject_header = next(header['value'] for header in headers if header['name'] == 'Subject')
                print(f"From: {from_header}, Subject: {subject_header}")
            except Exception as e:
                print(f"An error occured in the main body (Unreplied Emails): {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"An error occurred in the main body: {e}")

    print(f'END OF PROGRAM. REPLIED EMAILS: {len(replied)},  UNREPLIED EMAILS: {len(unreplied)}')