"""
Script to load and process the complete fake news dataset for analysis and model training.

This script loads data from Hive tables ('fake' and 'real'), combines them with labels,
and provides options to work with either the complete dataset or create samples for
development and testing purposes with limited resources.
"""

import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, when

def create_directory_structure():
    """Create directory structure for data storage in DBFS.
    
    This function creates the necessary directories for storing:
    - Combined data: The full dataset with both real and fake news
    - Sample data: Optional balanced subset for development and testing
    """
    # In Databricks, we use dbutils to interact with DBFS
    directories = [
        "dbfs:/FileStore/fake_news_detection/data/combined_data",
        "dbfs:/FileStore/fake_news_detection/data/sample_data"
    ]
    
    for directory in directories:
        # Remove dbfs: prefix for dbutils.fs.mkdirs
        dir_path = directory.replace("dbfs:", "")
        dbutils.fs.mkdirs(dir_path)
        print(f"Created directory: {directory}")

def load_and_process_data(create_sample=False, sample_size=None):
    """
    Load datasets from Hive tables, combine them with labels, and optionally create a sample.
    
    Args:
        create_sample (bool): Whether to create a sample dataset (default: False)
        sample_size (int): Number of records per class for the sample (default: None, which uses all data)
                          If create_sample is True and sample_size is None, defaults to 1000 per class
    
    Returns:
        None: Data is saved to DBFS and Hive tables
    """
    print("Initializing Spark session...")
    # Configuration optimized for Databricks Community Edition (15.3 GB Memory, 2 Cores)
    spark = SparkSession.builder \
        .appName("FakeNewsDataProcessing") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "8g") \
        .enableHiveSupport() \
        .getOrCreate()
    
    print("Loading datasets...")
    # Try to load from Hive tables first
    try:
        # Load from Hive tables
        true_df = spark.table("real")
        fake_df = spark.table("fake")
        
        print(f"True news dataset loaded from Hive table: {true_df.count()} records")
        print(f"Fake news dataset loaded from Hive table: {fake_df.count()} records")
    except Exception as e:
        print(f"Could not load from Hive tables: {e}")
        # Fall back to loading from DBFS if available
        try:
            true_path = "dbfs:/FileStore/fake_news_detection/data/raw/True.csv"
            fake_path = "dbfs:/FileStore/fake_news_detection/data/raw/Fake.csv"
            
            true_df = spark.read.csv(true_path, header=True, inferSchema=True)
            fake_df = spark.read.csv(fake_path, header=True, inferSchema=True)
            
            print(f"True news dataset loaded from DBFS: {true_df.count()} records")
            print(f"Fake news dataset loaded from DBFS: {fake_df.count()} records")
        except Exception as e2:
            print(f"Error loading datasets from DBFS: {str(e2)}")
            print("Exiting as no data sources are available.")
            return
    
    try:
        # Add labels (1 for real, 0 for fake)
        print("Adding labels...")
        true_df = true_df.withColumn("label", lit(1))
        fake_df = fake_df.withColumn("label", lit(0))
        
        # Combine datasets
        print("Combining datasets...")
        combined_df = true_df.unionByName(fake_df)
        
        # Get dataset statistics for reporting
        real_count = combined_df.filter(col('label') == 1).count()
        fake_count = combined_df.filter(col('label') == 0).count()
        total_count = real_count + fake_count
        
        # Save combined dataset to DBFS with partitioning and compression
        print("Saving combined dataset...")
        combined_df.write \
            .mode("overwrite") \
            .partitionBy("label") \
            .option("compression", "snappy") \
            .parquet("dbfs:/FileStore/fake_news_detection/data/combined_data/combined_news.parquet")
        
        # Save as Hive table for easier access
        combined_df.write.mode("overwrite").saveAsTable("combined_news")
        print("Combined dataset saved as Hive table: combined_news")
        
        # Print combined dataset statistics
        print("\nCombined Dataset Statistics:")
        print(f"Total records: {total_count}")
        print(f"Real news: {real_count}")
        print(f"Fake news: {fake_count}")
        
        # Analyze subject distribution
        print("\nAnalyzing subject distribution...")
        subject_by_label = combined_df.groupBy("subject", "label").count()
        subject_by_label_pivot = subject_by_label.groupBy("subject") \
            .pivot("label", [0, 1]) \
            .sum("count") \
            .na.fill(0) \
            .withColumnRenamed("0", "fake_count") \
            .withColumnRenamed("1", "real_count")
        
        # Show top subjects by count
        print("\nSubject distribution by label (top 10):")
        subject_by_label_pivot \
            .withColumn("total", col("fake_count") + col("real_count")) \
            .orderBy(col("total").desc()) \
            .select("subject", "real_count", "fake_count") \
            .show(10)
        
        # Check for potential data leakage in subject column
        print("\nChecking for potential data leakage in subject column...")
        # Count subjects that appear in both classes
        overlap_subjects = subject_by_label_pivot \
            .filter((col("fake_count") > 0) & (col("real_count") > 0)) \
            .count()
        
        print(f"Subjects appearing in both real and fake news: {overlap_subjects}")
        if overlap_subjects == 0:
            print("WARNING: No overlap in subjects between real and fake news.")
            print("This suggests potential data leakage - the 'subject' column may perfectly separate classes.")
            print("Consider removing this column from feature set or creating models with and without it.")
        
        # Create sample if requested (useful for Community Edition with limited resources)
        if create_sample:
            print("\nCreating sample dataset for development with limited resources...")
            # Default sample size if not specified
            if sample_size is None:
                sample_size = 1000
                
            print(f"Using sample size of {sample_size} records per class")
            
            # Use sampleBy for stratified sampling by label
            sample_fractions = {
                0: min(1.0, sample_size/fake_count),
                1: min(1.0, sample_size/real_count)
            }
            
            sample_df = combined_df.sampleBy("label", fractions=sample_fractions, seed=42)
            
            # Save sample dataset to DBFS with partitioning and compression
            print("Saving sample dataset...")
            sample_df.write \
                .mode("overwrite") \
                .partitionBy("label") \
                .option("compression", "snappy") \
                .parquet("dbfs:/FileStore/fake_news_detection/data/sample_data/sample_news.parquet")
            
            # Save as Hive table for easier access
            sample_df.write.mode("overwrite").saveAsTable("sample_news")
            print("Sample dataset saved as Hive table: sample_news")
            
            # Print sample statistics
            sample_real_count = sample_df.filter(col('label') == 1).count()
            sample_fake_count = sample_df.filter(col('label') == 0).count()
            
            print("\nSample Dataset Statistics:")
            print(f"Total records: {sample_real_count + sample_fake_count}")
            print(f"Real news: {sample_real_count}")
            print(f"Fake news: {sample_fake_count}")
            
            # Show sample of each class
            print("\nSample of real news:")
            sample_df.filter(col('label') == 1).select("title", "text").show(3, truncate=50)
            
            print("\nSample of fake news:")
            sample_df.filter(col('label') == 0).select("title", "text").show(3, truncate=50)
            
            print("\nNOTE: This sample is intended for development and testing with limited resources.")
            print("For production models, use the full dataset when resources permit.")
        
        # Release memory
        if 'subject_by_label' in locals():
            subject_by_label.unpersist()
        if 'subject_by_label_pivot' in locals():
            subject_by_label_pivot.unpersist()
        if create_sample and 'sample_df' in locals():
            sample_df.unpersist()
        
        print("\nData processing completed successfully!")
        print("The data is now ready for preprocessing and feature engineering.")
        
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
    finally:
        # Release memory before stopping Spark
        if 'combined_df' in locals():
            combined_df.unpersist()
        if 'true_df' in locals():
            true_df.unpersist()
        if 'fake_df' in locals():
            fake_df.unpersist()
            
        spark.stop()
        print("Spark session stopped.")

if __name__ == "__main__":
    print("Starting data processing...")
    create_directory_structure()
    
    # By default, process the complete dataset without creating a sample
    # To create a sample for development with limited resources, call with create_sample=True
    load_and_process_data(create_sample=False)
    
    print("\nResource Management Tips for Databricks Community Edition:")
    print("1. For complex models (LSTM, deep learning), consider using a sample")
    print("2. For simpler models (Naive Bayes, Logistic Regression), the full dataset can be used")
    print("3. Always use partitioned Parquet files with Snappy compression")
    print("4. Release memory with .unpersist() when DataFrames are no longer needed")

# Last modified: May 29, 2025
