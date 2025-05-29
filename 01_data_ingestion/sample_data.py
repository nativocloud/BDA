"""
Script to load and process the complete fake news dataset with optional sampling capabilities.

This script provides functionality to:
1. Process the complete fake news dataset (default behavior)
2. Create balanced samples for development and testing (optional)
3. Generate streaming samples for pipeline testing (optional)

The script prioritizes working with the full dataset while maintaining sampling
capabilities for specific development and testing scenarios.
"""

import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, when, rand

def create_directory_structure():
    """Create directory structure for data storage in DBFS.
    
    This function creates the necessary directories for storing:
    - Combined data: The full dataset with both real and fake news
    - Sample data: Optional balanced subset for development and testing
    - Streaming data: Optional streaming samples for pipeline testing
    """
    # In Databricks, we use dbutils to interact with DBFS
    directories = [
        "dbfs:/FileStore/fake_news_detection/data/combined_data",
        "dbfs:/FileStore/fake_news_detection/data/sample_data",
        "dbfs:/FileStore/fake_news_detection/data/streaming"
    ]
    
    for directory in directories:
        # Remove dbfs: prefix for dbutils.fs.mkdirs
        dir_path = directory.replace("dbfs:", "")
        dbutils.fs.mkdirs(dir_path)
        print(f"Created directory: {directory}")

def process_full_dataset(create_sample=False, sample_size=None, create_stream_sample=False, stream_size=20):
    """
    Process the complete fake news dataset with optional sampling capabilities.
    
    Args:
        create_sample (bool): Whether to create a balanced sample dataset (default: False)
        sample_size (int): Number of records per class for the sample (default: None, uses 100 if create_sample is True)
        create_stream_sample (bool): Whether to create a streaming sample (default: False)
        stream_size (int): Number of records for the streaming sample (default: 20)
    
    Returns:
        None: Data is saved to DBFS and Hive tables
    """
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("FakeNewsDataProcessing") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
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
        
        # Save combined dataset to DBFS
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
        total_count = combined_df.count()
        real_count = combined_df.filter(col('label') == 1).count()
        fake_count = combined_df.filter(col('label') == 0).count()
        
        print(f"Total records: {total_count}")
        print(f"Real news: {real_count}")
        print(f"Fake news: {fake_count}")
        
        # Analyze subject distribution to check for potential data leakage
        print("\nAnalyzing subject distribution (potential data leakage check)...")
        subject_counts = combined_df.groupBy("label").pivot("subject").count().na.fill(0)
        
        # Count unique subjects by label
        unique_subjects_by_label = combined_df.groupBy("label", "subject").count() \
            .groupBy("label").count() \
            .withColumnRenamed("count", "unique_subjects")
        
        print("Combined dataset statistics:")
        combined_df.groupBy("label").count() \
            .join(unique_subjects_by_label, "label") \
            .show()
        
        # Show top subjects by label
        print("\nSubject distribution by label (top 10):")
        combined_df.groupBy("subject", "label").count() \
            .withColumn("label_type", when(col("label") == 1, "real_count").otherwise("fake_count")) \
            .groupBy("subject").pivot("label_type").sum("count").na.fill(0) \
            .orderBy(col("real_count") + col("fake_count"), ascending=False) \
            .show(10)
        
        # Create balanced sample if requested
        if create_sample:
            print("\nCreating balanced sample dataset...")
            # Default sample size if not specified
            if sample_size is None:
                sample_size = 100
                
            print(f"Using sample size of {sample_size} records per class")
            
            # Use sampleBy for stratified sampling by label
            sample_fractions = {
                0: min(1.0, sample_size/fake_count),
                1: min(1.0, sample_size/real_count)
            }
            
            sample_df = combined_df.sampleBy("label", fractions=sample_fractions, seed=42)
            
            # Save sample dataset to DBFS
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
            print("\nSample Dataset Statistics:")
            print(f"Total records: {sample_df.count()}")
            print(f"Real news: {sample_df.filter(col('label') == 1).count()}")
            print(f"Fake news: {sample_df.filter(col('label') == 0).count()}")
            
            # Show sample of each class
            print("\nSample of real news:")
            sample_df.filter(col('label') == 1).select("title", "text").show(3, truncate=50)
            
            print("\nSample of fake news:")
            sample_df.filter(col('label') == 0).select("title", "text").show(3, truncate=50)
        
        # Create streaming sample if requested
        if create_stream_sample:
            print("\nCreating streaming sample...")
            
            # Create a small streaming sample with timestamp and ID
            stream_df = combined_df.orderBy(rand(seed=42)).limit(stream_size)
            
            # Add timestamp and ID columns
            from pyspark.sql.functions import current_timestamp, monotonically_increasing_id, date_format
            
            stream_df = stream_df \
                .withColumn("timestamp", current_timestamp()) \
                .withColumn("id", monotonically_increasing_id()) \
                .withColumn("id", date_format(col("timestamp"), "yyyyMMdd_HHmmss_") + col("id").cast("string"))
            
            # Save streaming sample to DBFS
            stream_df.write \
                .mode("overwrite") \
                .option("compression", "snappy") \
                .parquet("dbfs:/FileStore/fake_news_detection/data/streaming/stream_sample.parquet")
            
            # Save as Hive table for easier access
            stream_df.write.mode("overwrite").saveAsTable("stream_sample")
            print(f"Streaming sample with {stream_size} records saved to DBFS and Hive table: stream_sample")
        
        print("\nData processing completed successfully!")
        
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()
        print("Spark session stopped.")

if __name__ == "__main__":
    print("Starting data processing...")
    create_directory_structure()
    
    # By default, process the complete dataset without creating samples
    # To create samples, call with create_sample=True and/or create_stream_sample=True
    process_full_dataset(create_sample=False, create_stream_sample=False)
