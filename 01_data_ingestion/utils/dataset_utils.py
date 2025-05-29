"""
Dataset preparation and augmentation utility functions for fake news detection project.
This module contains functions for preparing and supplementing datasets while preventing data leakage.
"""

import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, regexp_replace, lower, udf, concat
from pyspark.sql.types import StringType, ArrayType
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_and_clean_datasets(spark, fake_path, true_path):
    """
    Load and clean fake and true news datasets.
    
    Args:
        spark (SparkSession): Spark session
        fake_path (str): Path to fake news CSV file
        true_path (str): Path to true news CSV file
        
    Returns:
        DataFrame: Combined and cleaned dataset with labels
    """
    # Load datasets
    df_fake = spark.read.csv(fake_path, header=True, inferSchema=True)
    df_real = spark.read.csv(true_path, header=True, inferSchema=True)
    
    # Add labels (0 for fake, 1 for real)
    df_fake = df_fake.withColumn("label", lit(0))
    df_real = df_real.withColumn("label", lit(1))
    
    # Select relevant columns and handle missing values
    if "title" in df_fake.columns and "text" in df_fake.columns:
        df_fake = df_fake.select("title", "text", "label").na.drop()
        df_real = df_real.select("title", "text", "label").na.drop()
        
        # Combine title and text for better context
        df_fake = df_fake.withColumn("full_text", 
                                    concat(col("title"), lit(". "), col("text")))
        df_real = df_real.withColumn("full_text", 
                                    concat(col("title"), lit(". "), col("text")))
        
        # Select final columns
        df_fake = df_fake.select("full_text", "label")
        df_real = df_real.select("full_text", "label")
        
        # Rename column
        df_fake = df_fake.withColumnRenamed("full_text", "text")
        df_real = df_real.withColumnRenamed("full_text", "text")
    else:
        df_fake = df_fake.select("text", "label").na.drop()
        df_real = df_real.select("text", "label").na.drop()
    
    # Combine datasets
    df = df_fake.unionByName(df_real)
    
    # Clean text
    df = df.withColumn("text", lower(col("text")))
    df = df.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z0-9\\s]", " "))
    df = df.withColumn("text", regexp_replace(col("text"), "\\s+", " "))
    
    return df

def advanced_text_preprocessing(df):
    """
    Perform advanced text preprocessing including lemmatization and stopword removal.
    
    Args:
        df: DataFrame with text column
        
    Returns:
        DataFrame: DataFrame with preprocessed text
    """
    # Convert to pandas for more efficient text processing
    pandas_df = df.toPandas()
    
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
        
        # Join tokens back into text
        processed_text = ' '.join(processed_tokens)
        
        return processed_text
    
    # Apply preprocessing
    pandas_df['processed_text'] = pandas_df['text'].apply(preprocess_text)
    
    # Convert back to Spark DataFrame
    processed_df = df.sparkSession.createDataFrame(pandas_df)
    
    return processed_df

def augment_dataset_with_synonyms(df, augmentation_factor=0.2, seed=42):
    """
    Augment dataset by replacing words with synonyms.
    This does not introduce future data as it only uses existing vocabulary.
    
    Args:
        df: DataFrame with text column
        augmentation_factor (float): Fraction of data to augment
        seed (int): Random seed for reproducibility
        
    Returns:
        DataFrame: Augmented DataFrame
    """
    from nltk.corpus import wordnet
    nltk.download('wordnet', quiet=True)
    
    # Convert to pandas for more efficient text processing
    pandas_df = df.toPandas()
    
    # Set random seed
    np.random.seed(seed)
    
    # Select subset for augmentation
    n_samples = int(len(pandas_df) * augmentation_factor)
    augment_indices = np.random.choice(len(pandas_df), n_samples, replace=False)
    
    augmented_texts = []
    augmented_labels = []
    
    for idx in augment_indices:
        text = pandas_df.iloc[idx]['text']
        label = pandas_df.iloc[idx]['label']
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Replace some words with synonyms
        new_tokens = []
        for token in tokens:
            # 30% chance to replace with synonym if word is alphabetic
            if token.isalpha() and np.random.random() < 0.3:
                synonyms = []
                for syn in wordnet.synsets(token):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
                
                if synonyms:
                    # Remove duplicates and original word
                    synonyms = list(set(synonyms))
                    if token in synonyms:
                        synonyms.remove(token)
                    
                    if synonyms:
                        # Replace with random synonym
                        replacement = np.random.choice(synonyms)
                        new_tokens.append(replacement)
                    else:
                        new_tokens.append(token)
                else:
                    new_tokens.append(token)
            else:
                new_tokens.append(token)
        
        # Join tokens back into text
        augmented_text = ' '.join(new_tokens)
        
        augmented_texts.append(augmented_text)
        augmented_labels.append(label)
    
    # Create DataFrame with augmented data
    augmented_df = pd.DataFrame({
        'text': augmented_texts,
        'label': augmented_labels
    })
    
    # Combine original and augmented data
    combined_df = pd.concat([pandas_df, augmented_df], ignore_index=True)
    
    # Convert back to Spark DataFrame
    result_df = df.sparkSession.createDataFrame(combined_df)
    
    return result_df

def create_balanced_dataset(df, balance_method='undersample', seed=42):
    """
    Create a balanced dataset by undersampling or oversampling.
    
    Args:
        df: DataFrame with text and label columns
        balance_method (str): Method to balance dataset ('undersample' or 'oversample')
        seed (int): Random seed for reproducibility
        
    Returns:
        DataFrame: Balanced DataFrame
    """
    # Get counts by label
    label_counts = df.groupBy("label").count().collect()
    label_counts_dict = {row['label']: row['count'] for row in label_counts}
    
    # Determine minority and majority classes
    min_label = min(label_counts_dict.items(), key=lambda x: x[1])[0]
    max_label = max(label_counts_dict.items(), key=lambda x: x[1])[0]
    min_count = label_counts_dict[min_label]
    max_count = label_counts_dict[max_label]
    
    if balance_method == 'undersample':
        # Undersample majority class
        df_majority = df.filter(col("label") == max_label)
        df_minority = df.filter(col("label") == min_label)
        
        # Sample majority class to match minority class size
        df_majority_sampled = df_majority.sample(fraction=min_count/max_count, seed=seed)
        
        # Combine minority class with sampled majority class
        balanced_df = df_majority_sampled.unionByName(df_minority)
        
    elif balance_method == 'oversample':
        # Oversample minority class
        df_majority = df.filter(col("label") == max_label)
        df_minority = df.filter(col("label") == min_label)
        
        # Calculate oversampling ratio (with replacement)
        ratio = max_count / min_count
        
        # Oversample minority class
        df_minority_oversampled = df_minority.sample(fraction=ratio, withReplacement=True, seed=seed)
        
        # Combine majority class with oversampled minority class
        balanced_df = df_majority.unionByName(df_minority_oversampled)
    
    else:
        raise ValueError("balance_method must be 'undersample' or 'oversample'")
    
    return balanced_df

def split_dataset_by_time(df, date_col, train_ratio=0.8, seed=42):
    """
    Split dataset by time to prevent data leakage.
    
    Args:
        df: DataFrame with text and label columns
        date_col (str): Column containing date information
        train_ratio (float): Ratio of training data
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, test_df)
    """
    # Sort by date
    sorted_df = df.orderBy(date_col)
    
    # Calculate split point
    count = sorted_df.count()
    split_point = int(count * train_ratio)
    
    # Split data
    train_df = sorted_df.limit(split_point)
    test_df = sorted_df.subtract(train_df)
    
    return train_df, test_df

def save_prepared_datasets(train_df, test_df, output_dir):
    """
    Save prepared datasets to disk.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        output_dir (str): Output directory
        
    Returns:
        tuple: (train_path, test_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    train_path = os.path.join(output_dir, "train_data.parquet")
    test_path = os.path.join(output_dir, "test_data.parquet")
    
    # Save datasets
    train_df.write.mode("overwrite").parquet(train_path)
    test_df.write.mode("overwrite").parquet(test_path)
    
    print(f"Training data saved to {train_path}")
    print(f"Testing data saved to {test_path}")
    
    return train_path, test_path

def create_streaming_simulation_data(df, output_path, batch_size=10, num_batches=5, seed=42):
    """
    Create data for streaming simulation.
    
    Args:
        df: DataFrame with text and label columns
        output_path (str): Output path for streaming data
        batch_size (int): Number of records per batch
        num_batches (int): Number of batches to create
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of batch file paths
    """
    # Sample data for streaming simulation
    sample_size = batch_size * num_batches
    streaming_df = df.sample(fraction=min(1.0, sample_size/df.count()), seed=seed)
    
    # Convert to pandas for easier batch processing
    pandas_df = streaming_df.toPandas()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create batches
    batch_files = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(pandas_df))
        
        if start_idx >= len(pandas_df):
            break
        
        batch = pandas_df.iloc[start_idx:end_idx]
        
        # Add ID and timestamp columns
        batch['id'] = [f"doc_{i}_{j}" for j in range(len(batch))]
        batch['timestamp'] = pd.Timestamp.now()
        
        # Save batch
        batch_file = f"{output_path}_batch_{i}.csv"
        batch.to_csv(batch_file, index=False)
        batch_files.append(batch_file)
    
    print(f"Created {len(batch_files)} streaming data batches")
    
    return batch_files

def analyze_dataset(df, output_path=None):
    """
    Analyze dataset and generate statistics.
    
    Args:
        df: DataFrame with text and label columns
        output_path (str): Path to save analysis results
        
    Returns:
        dict: Dictionary of dataset statistics
    """
    # Count records by label
    label_counts = df.groupBy("label").count().collect()
    label_counts_dict = {row['label']: row['count'] for row in label_counts}
    
    # Calculate text length statistics
    from pyspark.sql.functions import length, mean, min, max, stddev
    
    text_length_stats = df.select(
        mean(length(col("text"))).alias("mean_length"),
        min(length(col("text"))).alias("min_length"),
        max(length(col("text"))).alias("max_length"),
        stddev(length(col("text"))).alias("stddev_length")
    ).collect()[0]
    
    # Convert to dictionary
    stats = {
        "total_records": df.count(),
        "label_distribution": label_counts_dict,
        "text_length_stats": {
            "mean": text_length_stats["mean_length"],
            "min": text_length_stats["min_length"],
            "max": text_length_stats["max_length"],
            "stddev": text_length_stats["stddev_length"]
        }
    }
    
    # Save statistics if output path is provided
    if output_path:
        import json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Dataset analysis saved to {output_path}")
    
    return stats

# Last modified: May 29, 2025
