# scripts/utils.py

import os
import json
import re
import unicodedata
from bs4 import BeautifulSoup
import pandas as pd

def load_emails(folder, label):
    data = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith('.json'):
                with open(os.path.join(root, filename), 'r') as f:
                    email = json.load(f)
                    text = email.get('subject', '') + ' ' + email.get('text_body', '')
                    data.append({'text': text, 'label': label})
    return data

def clean_email(text):
    # Parse and remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Normalize Unicode characters to remove unwanted symbols
    text = unicodedata.normalize('NFKD', text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove specific special characters
    text = re.sub(r'[🔥*]', '', text)

    # Remove text between [DEL: ... :DEL]
    text = re.sub(r'\[del:.*?:del\]', '', text)

    # Remove other square-bracketed items (e.g., [1], [2])
    text = re.sub(r'\[.*?\]', '', text)

    # Remove escape characters and excessive whitespace
    text = re.sub(r'\r\n|\n|\t', ' ', text)

    # Remove mentions of 'image' or placeholders
    text = re.sub(r'\bimage\b', '', text)

    # Remove non-Cyrillic, non-alphanumeric characters except basic punctuation
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9,.!?@+ ]+', '', text)

    # Remove redundant spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text