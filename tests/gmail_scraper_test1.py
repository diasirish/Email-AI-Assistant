from imapclient import IMAPClient
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import google.auth.transport.requests
import email

class GmailOAuthClient:
    def __init__(self, credentials_file):
        """
        Initialize the GmailOAuthClient to handle OAuth authentication.

        :param credentials_file: Path to the OAuth 2.0 credentials file
        """
        self.credentials_file = credentials_file
        self.creds = None

    def authenticate(self):
        """Authenticate using OAuth and store credentials."""
        flow = InstalledAppFlow.from_client_secrets_file(
            self.credentials_file,
            scopes=['https://mail.google.com/']
        )
        self.creds = flow.run_local_server(port=0)

    def refresh_token(self):
        """Refresh the OAuth token if it has expired."""
        if self.creds and self.creds.expired and self.creds.refresh_token:
            self.creds.refresh(google.auth.transport.requests.Request())

    def get_auth_string(self):
        """Return the authentication string required for IMAPClient."""
        return f"user={self.creds.client_id}\1auth=Bearer {self.creds.token}\1\1"


class EmailCategorizer:
    def __init__(self, host, credentials_file, ssl=True):
        """
        Initialize the EmailCategorizer with IMAP server details and OAuth credentials.

        :param host: The IMAP server address (e.g., 'imap.gmail.com')
        :param credentials_file: Path to the OAuth 2.0 credentials file
        :param ssl: Use SSL for the connection (default True)
        """
        self.host = host
        self.ssl = ssl
        self.client = None
        self.gmail_oauth_client = GmailOAuthClient(credentials_file)

    def connect(self):
        """Establish a connection to the IMAP server using OAuth."""
        # Authenticate using OAuth
        self.gmail_oauth_client.authenticate()

        # Refresh token if necessary
        self.gmail_oauth_client.refresh_token()

        # Use IMAPClient to connect
        self.client = IMAPClient(self.host, ssl=self.ssl)
        auth_string = self.gmail_oauth_client.get_auth_string()

        # Authenticate using XOAUTH2 with the IMAPClient
        self.client._imap.authenticate('XOAUTH2', lambda x: auth_string)

    def fetch_emails(self, folder):
        """
        Fetch emails from the specified folder.

        :param folder: The folder to fetch emails from (e.g., 'INBOX', 'Sent')
        :return: A dictionary of email data
        """
        self.client.select_folder(folder)
        messages = self.client.search(['ALL'])
        fetched = self.client.fetch(messages, ['FLAGS', 'RFC822'])
        return fetched

    def categorize_emails(self):
        """
        Categorize emails into replied and unreplied.

        :return: A tuple of two lists: (replied_emails, unreplied_emails)
        """
        # Fetch emails from INBOX and Sent folders
        inbox_emails = self.fetch_emails('INBOX')
        sent_emails = self.fetch_emails('Sent')

        # Extract message IDs from sent emails
        sent_message_ids = set()
        for msgid, data in sent_emails.items():
            msg = email.message_from_bytes(data[b'RFC822'])
            # Collect Message-ID or In-Reply-To to identify replied messages
            if msg['In-Reply-To']:
                sent_message_ids.add(msg['In-Reply-To'])
            elif msg['Message-ID']:
                sent_message_ids.add(msg['Message-ID'])

        # Categorize inbox emails
        replied_emails = []
        unreplied_emails = []

        for msgid, data in inbox_emails.items():
            msg = email.message_from_bytes(data[b'RFC822'])
            message_id = msg['Message-ID']

            if message_id in sent_message_ids:
                replied_emails.append(msg)
            else:
                unreplied_emails.append(msg)

        return replied_emails, unreplied_emails

    def close_connection(self):
        """Logout and close the connection to the IMAP server."""
        if self.client:
            self.client.logout()


if __name__ == "__main__":
    # Replace with your OAuth credentials file
    HOST = 'imap.gmail.com'
    CREDENTIALS_FILE = 'client_secret.json'

    # Create an instance of the EmailCategorizer
    categorizer = EmailCategorizer(HOST, CREDENTIALS_FILE)

    try:
        # Connect to the IMAP server using OAuth
        categorizer.connect()

        # Categorize emails
        replied, unreplied = categorizer.categorize_emails()

        # Print results
        print("Replied Emails:")
        for msg in replied:
            print(f"From: {msg['From']}, Subject: {msg['Subject']}")

        print("\nUnreplied Emails:")
        for msg in unreplied:
            print(f"From: {msg['From']}, Subject: {msg['Subject']}")
    
    finally:
        # Close the connection to the server
        categorizer.close_connection()