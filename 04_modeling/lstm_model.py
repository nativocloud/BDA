"""
Script to implement and run a simplified LSTM model for fake news detection.

This script loads the sampled data, performs preprocessing,
trains a simple LSTM model, and evaluates its performance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from pyspark.sql import SparkSession

def create_directory_structure():
    """Create directory structure for models and logs in DBFS.
    
    This function creates the necessary directories for storing:
    - Models: Trained LSTM models and tokenizers
    - Logs: Performance metrics and visualizations
    """
    # In Databricks, we use dbutils to interact with DBFS
    directories = [
        "dbfs:/FileStore/fake_news_detection/models/lstm",
        "dbfs:/FileStore/fake_news_detection/logs"
    ]
    
    for directory in directories:
        # Remove dbfs: prefix for dbutils.fs.mkdirs
        dir_path = directory.replace("dbfs:", "")
        dbutils.fs.mkdirs(dir_path)
        print(f"Created directory: {directory}")

def train_lstm_model():
    """
    Load sampled data, train an LSTM model, and evaluate its performance.
    """
    # Set paths for Databricks
    data_path = "dbfs:/FileStore/fake_news_detection/data/sample_data/sample_news.parquet"
    models_dir = "dbfs:/FileStore/fake_news_detection/models/lstm"
    logs_dir = "dbfs:/FileStore/fake_news_detection/logs"
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("FakeNewsLSTM") \
        .config("spark.driver.memory", "4g") \
        .enableHiveSupport() \
        .getOrCreate()
    
    # Load data from Hive table or parquet
    print("Loading data...")
    try:
        # Try to load from Hive table first
        df_spark = spark.table("sample_news")
        print("Loaded data from Hive table 'sample_news'")
    except Exception as e:
        print(f"Could not load from Hive table: {e}")
        # Fall back to loading from parquet
        try:
            df_spark = spark.read.parquet(data_path)
            print(f"Loaded data from {data_path}")
        except Exception as e2:
            print(f"Error loading data: {e2}")
            print("Exiting as no data sources are available.")
            return
    
    # Convert to pandas for model training
    df = df_spark.toPandas()
    
    # Preprocess text
    print("Preprocessing text...")
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    df['content'] = df['title'] + " " + df['text']
    
    # Print dataset info
    print(f"Dataset shape: {df.shape}")
    print("Class distribution:")
    print(df['label'].value_counts())
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
    )
    
    # Tokenize text
    print("Tokenizing text...")
    max_words = 10000
    max_len = 200
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    # Build model
    print("Building LSTM model...")
    vocab_size = min(len(tokenizer.word_index) + 1, max_words)
    embedding_dim = 100
    
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    # Train model
    print("Training LSTM model...")
    batch_size = 32
    epochs = 5
    
    history = model.fit(
        X_train_pad, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    y_pred_proba = model.predict(X_test_pad)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and tokenizer
    print("Saving model and tokenizer...")
    try:
        # In Databricks, we need to save locally first, then copy to DBFS
        local_model_path = "/tmp/lstm_model.keras"
        local_tokenizer_path = "/tmp/tokenizer.pickle"
        local_metrics_path = "/tmp/lstm_metrics.csv"
        local_cm_path = "/tmp/lstm_confusion_matrix.png"
        local_history_path = "/tmp/lstm_training_history.png"
        
        # Save model with .keras extension as required
        model.save(local_model_path)
        
        # Save tokenizer
        import pickle
        with open(local_tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Model': ['LSTM'],
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
            'Value': [accuracy, precision, recall, f1]
        })
        
        metrics_df.to_csv(local_metrics_path, index=False)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - LSTM Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(local_cm_path)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(local_history_path)
        
        # Copy files to DBFS
        dbutils.fs.cp(f"file:{local_model_path}", f"{models_dir}/lstm_model.keras")
        dbutils.fs.cp(f"file:{local_tokenizer_path}", f"{models_dir}/tokenizer.pickle")
        dbutils.fs.cp(f"file:{local_metrics_path}", f"{logs_dir}/lstm_metrics.csv")
        dbutils.fs.cp(f"file:{local_cm_path}", f"{logs_dir}/lstm_confusion_matrix.png")
        dbutils.fs.cp(f"file:{local_history_path}", f"{logs_dir}/lstm_training_history.png")
        
        print(f"Results saved to {logs_dir}")
        print(f"Model saved to {models_dir}")
        print("LSTM model implementation completed successfully.")
        
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    create_directory_structure()
    train_lstm_model()

# Last modified: May 29, 2025
