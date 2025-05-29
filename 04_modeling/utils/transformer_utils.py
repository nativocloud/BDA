"""
Transformer model utility functions for fake news detection project.
This module contains functions for creating and training transformer-based models.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

class NewsDataset(Dataset):
    """Dataset for transformer-based models."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize dataset.
        
        Args:
            texts (list): List of text documents
            labels (list): List of labels
            tokenizer: Transformer tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_transformer_data(spark_df, text_col="text", label_col="label"):
    """
    Prepare data for transformer models.
    
    Args:
        spark_df: Spark DataFrame
        text_col (str): Column containing text data
        label_col (str): Column containing labels
        
    Returns:
        pd.DataFrame: Pandas DataFrame
    """
    # Convert to pandas
    pandas_df = spark_df.select(text_col, label_col).toPandas()
    
    return pandas_df

def create_transformer_model(model_name="distilbert-base-uncased", num_labels=2):
    """
    Create a transformer model for text classification.
    
    Args:
        model_name (str): Name of the pretrained model
        num_labels (int): Number of output labels
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    return model, tokenizer

def train_transformer_model(model, tokenizer, train_texts, train_labels, val_texts, val_labels, 
                           batch_size=8, epochs=3, learning_rate=2e-5, max_length=512, 
                           model_path=None, device=None):
    """
    Train a transformer model.
    
    Args:
        model: Transformer model
        tokenizer: Transformer tokenizer
        train_texts (list): Training texts
        train_labels (list): Training labels
        val_texts (list): Validation texts
        val_labels (list): Validation labels
        batch_size (int): Batch size
        epochs (int): Number of epochs
        learning_rate (float): Learning rate
        max_length (int): Maximum sequence length
        model_path (str): Path to save the model
        device: Torch device
        
    Returns:
        tuple: (trained_model, training_stats)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Move model to device
    model.to(device)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    training_stats = []
    best_val_accuracy = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true = labels.cpu().numpy()
                
                val_preds.extend(preds)
                val_true.extend(true)
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(val_true, val_preds)
        
        # Save stats
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        }
        
        training_stats.append(epoch_stats)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if model_path and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(os.path.join(os.path.dirname(model_path), 'tokenizer'))
            print(f"Model saved to {model_path}")
    
    return model, training_stats

def evaluate_transformer_model(model, tokenizer, test_texts, test_labels, batch_size=8, max_length=512, log_dir=None, device=None):
    """
    Evaluate a transformer model.
    
    Args:
        model: Transformer model
        tokenizer: Transformer tokenizer
        test_texts (list): Test texts
        test_labels (list): Test labels
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        log_dir (str): Directory to save evaluation logs
        device: Torch device
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and dataloader
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Move model to device
    model.to(device)
    
    # Evaluation
    model.eval()
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            
            test_preds.extend(preds)
            test_true.extend(true)
    
    # Calculate metrics
    accuracy = accuracy_score(test_true, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_true, test_preds, average='binary')
    
    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # Log metrics
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "transformer_evaluation_metrics.txt"), "w") as f:
            f.write(f"Transformer Model Evaluation\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(test_true, test_preds))
    
    return metrics

def save_transformer_model(model, tokenizer, model_path):
    """
    Save transformer model and tokenizer.
    
    Args:
        model: Transformer model
        tokenizer: Transformer tokenizer
        model_path (str): Path to save the model
    """
    # Save model and tokenizer
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(os.path.join(model_path, 'tokenizer'))
    
    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {os.path.join(model_path, 'tokenizer')}")

def load_transformer_model(model_path):
    """
    Load transformer model and tokenizer.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
    
    print(f"Model loaded from {model_path}")
    print(f"Tokenizer loaded from {os.path.join(model_path, 'tokenizer')}")
    
    return model, tokenizer

# Last modified: May 29, 2025
