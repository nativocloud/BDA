# %%
"""
Data utility functions for fake news detection project.
This module contains functions for loading, preprocessing, and transforming data.
"""

# %%
import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, lit
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer

# %%
def initialize_spark():
    """
    Initialize a Spark session for data processing.
    
    Returns:
        SparkSession: Initialized Spark session
    """
    spark = SparkSession.builder \
        .appName("FakeNewsDetection") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .getOrCreate()
    
    return spark

# %%
def load_data(spark, fake_path, true_path):
    """
    Load and combine fake and true news datasets.
    
    Args:
        spark (SparkSession): Spark session
        fake_path (str): Path to fake news CSV file
        true_path (str): Path to true news CSV file
        
    Returns:
        DataFrame: Combined dataset with labels
    """
    # Load datasets
    df_fake = spark.read.csv(fake_path, header=True, inferSchema=True)
    df_real = spark.read.csv(true_path, header=True, inferSchema=True)
    
    # Add labels (0 for fake, 1 for real)
    df_fake = df_fake.withColumn("label", lit(0))
    df_real = df_real.withColumn("label", lit(1))
    
    # Combine datasets
    df = df_fake.unionByName(df_real).select("text", "label").na.drop()
    
    return df

# %%
def preprocess_text(df):
    """
    Preprocess text data by converting to lowercase and removing special characters.
    
    Args:
        df (DataFrame): Input DataFrame with text column
        
    Returns:
        DataFrame: DataFrame with preprocessed text
    """
    # Convert to lowercase and remove special characters
    df = df.withColumn("text", lower(regexp_replace(col("text"), "[^a-zA-Z\\s]", "")))
    
    return df

# %%
def create_feature_pipeline(input_col="text", output_col="features", hash_size=10000):
    """
    Create a feature extraction pipeline for text data.
    
    Args:
        input_col (str): Input column name
        output_col (str): Output column name
        hash_size (int): Size of the feature vectors
        
    Returns:
        list: List of pipeline stages
    """
    # Create pipeline stages
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=hash_size)
    idf = IDF(inputCol="rawFeatures", outputCol=output_col)
    
    return [tokenizer, remover, hashingTF, idf]

# %%
def split_data(df, train_ratio=0.8, test_ratio=0.2, seed=42):
    """
    Split data into training and testing sets.
    
    Args:
        df (DataFrame): Input DataFrame
        train_ratio (float): Ratio of training data
        test_ratio (float): Ratio of testing data
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_data, test_data)
    """
    # Split data
    train_data, test_data = df.randomSplit([train_ratio, test_ratio], seed=seed)
    
    return train_data, test_data

# %%
def save_model(model, path):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        path (str): Path to save the model
    """
    model.write().overwrite().save(path)
    print(f"Model saved to {path}")

# %%
def load_model(spark, path):
    """
    Load a trained model from disk.
    
    Args:
        spark (SparkSession): Spark session
        path (str): Path to the saved model
        
    Returns:
        Model: Loaded model
    """
    from pyspark.ml.pipeline import PipelineModel
    
    model = PipelineModel.load(path)
    print(f"Model loaded from {path}")
    
    return model
