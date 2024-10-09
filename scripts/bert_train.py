# scripts/train_model.py

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from utils import load_emails, clean_email

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# PyTorch and Transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train DistilBERT model for email classification.')
    parser.add_argument('--data_folder', type=str, default='data/', help='Path to the data folder.')
    parser.add_argument('--output_folder', type=str, default='outputs/', help='Path to save outputs.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--use_mlflow', action='store_true', help='Use MLflow for tracking experiments.')
    return parser.parse_args()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),  # Shape: (max_length,)
            'attention_mask': encoding['attention_mask'].flatten(),  # Shape: (max_length,)
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc='Training'):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model = model.eval()
    y_preds = []
    y_true = []
    y_scores = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, dim=1)

            y_preds.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probabilities[:, 1].cpu().numpy())

    return y_true, y_preds, y_scores

def main():
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_folder, exist_ok=True)
    model_output_dir = os.path.join(args.output_folder, 'model_weights')
    metrics_output_dir = os.path.join(args.output_folder, 'metrics')
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(metrics_output_dir, exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    replied_emails = load_emails(os.path.join(args.data_folder, 'data_mail1/replied/'), 1)
    unreplied_emails = load_emails(os.path.join(args.data_folder, 'data_mail1/unreplied/'), 0)
    replied_emails_2 = load_emails(os.path.join(args.data_folder, 'data_mail2/replied/'), 1)
    unreplied_emails_2 = load_emails(os.path.join(args.data_folder, 'data_mail2/unreplied/'), 0)

    # Combine and create a DataFrame
    all_emails = replied_emails + unreplied_emails + replied_emails_2 + unreplied_emails_2
    df = pd.DataFrame(all_emails)
    print(f'Columns of df are: {df.columns}')
    # Clean text data
    df['text'] = df['text'].apply(clean_email)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Create datasets and dataloaders
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_length=args.max_length)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Handle class imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model.to(device)
    class_weights = class_weights.to(device)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    EPOCHS = args.epochs
    best_f1 = 0.0
    metrics_history = []

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f'Train loss: {train_loss:.4f}')

        y_true, y_preds, y_scores = eval_model(model, test_loader, device)
        recall = recall_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, zero_division=0)
        f1 = f1_score(y_true, y_preds, zero_division=0)
        accuracy = (np.array(y_preds) == np.array(y_true)).mean()
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)

        print(f'Validation Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}')
        print(f'Validation ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}')

        # Save metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        metrics_history.append(epoch_metrics)

        # Save model if performance improves
        if f1 > best_f1:
            best_f1 = f1
            model_save_path = os.path.join(model_output_dir, f'best_model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')

    # Save final model
    final_model_path = os.path.join(model_output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')

    # Save metrics to a JSON file
    metrics_file = os.path.join(metrics_output_dir, 'metrics_history.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=4)
    print(f'Metrics saved to {metrics_file}')

    # Optionally, you can save the metrics as a CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv(os.path.join(metrics_output_dir, 'metrics_history.csv'), index=False)

if __name__ == '__main__':
    main()