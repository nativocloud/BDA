"""
LSTM and deep learning utility functions for fake news detection project.
This module contains functions for creating and training LSTM models.
"""

import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def spark_df_to_pandas(spark_df, text_col="text", label_col="label"):
    """
    Convert Spark DataFrame to Pandas DataFrame for deep learning.
    
    Args:
        spark_df: Spark DataFrame
        text_col (str): Column containing text data
        label_col (str): Column containing labels
        
    Returns:
        pd.DataFrame: Pandas DataFrame
    """
    # Collect data from Spark DataFrame
    pandas_df = spark_df.select(text_col, label_col).toPandas()
    
    return pandas_df

def prepare_text_data(texts, max_features=10000, max_len=200):
    """
    Prepare text data for LSTM model.
    
    Args:
        texts (list): List of text documents
        max_features (int): Maximum number of words to keep
        max_len (int): Maximum sequence length
        
    Returns:
        tuple: (tokenizer, sequences)
    """
    # Create tokenizer
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    return tokenizer, padded_sequences

def create_lstm_model(max_features=10000, embedding_dim=128, max_len=200):
    """
    Create an LSTM model for text classification.
    
    Args:
        max_features (int): Maximum number of words to keep
        embedding_dim (int): Dimension of the embedding layer
        max_len (int): Maximum sequence length
        
    Returns:
        tf.keras.Model: LSTM model
    """
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=max_len))
    
    # LSTM layers
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def train_lstm_model(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=10, model_path=None):
    """
    Train an LSTM model.
    
    Args:
        model: LSTM model
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        batch_size (int): Batch size
        epochs (int): Number of epochs
        model_path (str): Path to save the model
        
    Returns:
        tuple: (trained_model, history)
    """
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    # Add model checkpoint if path is provided
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        callbacks.append(ModelCheckpoint(model_path, save_best_only=True))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history

def evaluate_lstm_model(model, X_test, y_test, log_dir=None):
    """
    Evaluate an LSTM model.
    
    Args:
        model: LSTM model
        X_test: Test data
        y_test: Test labels
        log_dir (str): Directory to save evaluation logs
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
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
        with open(os.path.join(log_dir, "lstm_evaluation_metrics.txt"), "w") as f:
            f.write(f"LSTM Model Evaluation\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
    
    return metrics

def save_lstm_model(model, tokenizer, model_path, tokenizer_path):
    """
    Save LSTM model and tokenizer.
    
    Args:
        model: LSTM model
        tokenizer: Tokenizer
        model_path (str): Path to save the model
        tokenizer_path (str): Path to save the tokenizer
    """
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    
    # Save tokenizer
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    import pickle
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

def load_lstm_model(model_path, tokenizer_path):
    """
    Load LSTM model and tokenizer.
    
    Args:
        model_path (str): Path to the saved model
        tokenizer_path (str): Path to the saved tokenizer
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load tokenizer
    import pickle
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    print(f"Tokenizer loaded from {tokenizer_path}")
    
    return model, tokenizer

# Last modified: May 29, 2025
