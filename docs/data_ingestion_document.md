# Data Ingestion Document: Fake News Detection System

## 1. Introduction

This document outlines the data ingestion process for the Fake News Detection system. It details how data is acquired, processed, and prepared for both the offline model training pipeline and the online streaming pipeline. The document provides step-by-step instructions for implementing data ingestion in both local development and Databricks Community Edition environments.

## 2. Data Sources

### 2.1 Offline Training Data

The system uses two primary datasets for training and evaluation:

- **Fake.csv**: Contains fake news articles with the following columns:
  - `title`: The title of the news article
  - `text`: The content of the news article
  - Additional metadata columns (publication date, author, etc.)

- **True.csv**: Contains real news articles with the same structure as Fake.csv

### 2.2 Online Streaming Data

For the streaming pipeline, we use:

- **stream1.csv**: Sample streaming data for simulation
- **Generated batches**: Small CSV files created from the original datasets to simulate streaming data

## 3. Data Ingestion Workflow

### 3.1 Offline Data Ingestion

#### 3.1.1 Local Environment

1. **Data Loading**:
   ```python
   from utils.data_utils import initialize_spark, load_data
   
   # Initialize Spark session
   spark = initialize_spark()
   
   # Define data paths
   fake_path = "/home/ubuntu/fake_news_detection/data/Fake.csv"
   true_path = "/home/ubuntu/fake_news_detection/data/True.csv"
   
   # Load data
   df = load_data(spark, fake_path, true_path)
   ```

2. **Data Cleaning and Preprocessing**:
   ```python
   from utils.data_utils import preprocess_text
   from utils.dataset_utils import advanced_text_preprocessing
   
   # Basic preprocessing
   df = preprocess_text(df)
   
   # Advanced preprocessing (optional)
   df = advanced_text_preprocessing(df)
   ```

3. **Data Augmentation** (ensuring no future data leakage):
   ```python
   from utils.dataset_utils import augment_dataset_with_synonyms
   
   # Augment dataset with synonyms
   df_augmented = augment_dataset_with_synonyms(df, augmentation_factor=0.2, seed=42)
   ```

4. **Dataset Balancing**:
   ```python
   from utils.dataset_utils import create_balanced_dataset
   
   # Create balanced dataset
   df_balanced = create_balanced_dataset(df_augmented, balance_method='undersample', seed=42)
   ```

5. **Train-Test Split**:
   ```python
   from utils.data_utils import split_data
   
   # Split data into training and testing sets
   train_data, test_data = split_data(df_balanced, train_ratio=0.8, test_ratio=0.2, seed=42)
   ```

6. **Feature Extraction**:
   ```python
   from utils.data_utils import create_feature_pipeline
   from pyspark.ml import Pipeline
   
   # Create feature extraction pipeline
   feature_stages = create_feature_pipeline(input_col="text", output_col="features", hash_size=10000)
   feature_pipeline = Pipeline(stages=feature_stages)
   
   # Transform data
   train_features = feature_pipeline.fit(train_data).transform(train_data)
   test_features = feature_pipeline.fit(test_data).transform(test_data)
   ```

7. **Save Processed Data**:
   ```python
   from utils.dataset_utils import save_prepared_datasets
   
   # Save prepared datasets
   train_path, test_path = save_prepared_datasets(
       train_features, 
       test_features, 
       "/home/ubuntu/fake_news_detection/data/processed"
   )
   ```

#### 3.1.2 Databricks Environment

1. **Data Upload to DBFS**:
   - Upload Fake.csv and True.csv to DBFS using the Databricks UI or CLI:
     ```bash
     databricks fs cp /path/to/Fake.csv dbfs:/FileStore/tables/fake.csv
     databricks fs cp /path/to/True.csv dbfs:/FileStore/tables/real.csv
     ```

2. **Data Loading in Databricks**:
   ```python
   # Initialize Spark session (already available in Databricks)
   
   # Define data paths
   fake_path = "/FileStore/tables/fake.csv"
   true_path = "/FileStore/tables/real.csv"
   
   # Load data
   df_fake = spark.read.csv(fake_path, header=True, inferSchema=True)
   df_real = spark.read.csv(true_path, header=True, inferSchema=True)
   
   # Add labels
   df_fake = df_fake.withColumn("label", lit(0))
   df_real = df_real.withColumn("label", lit(1))
   
   # Combine datasets
   df = df_fake.unionByName(df_real).select("text", "label").na.drop()
   ```

3. **Follow the same preprocessing, augmentation, and feature extraction steps as in the local environment**

4. **Save to Delta Table**:
   ```python
   # Save to Delta table
   train_features.write.format("delta").mode("overwrite").save("/FileStore/tables/train_features")
   test_features.write.format("delta").mode("overwrite").save("/FileStore/tables/test_features")
   ```

### 3.2 Online Streaming Data Ingestion

#### 3.2.1 Local Environment

1. **Create Streaming Simulation Data**:
   ```python
   from utils.dataset_utils import create_streaming_simulation_data
   
   # Create streaming simulation data
   batch_files = create_streaming_simulation_data(
       df,
       "/home/ubuntu/fake_news_detection/data/streaming_input/stream",
       batch_size=10,
       num_batches=5,
       seed=42
   )
   ```

2. **Set Up Streaming Source**:
   ```python
   # Define schema for streaming data
   news_schema = StructType([
       StructField("id", StringType(), True),
       StructField("text", StringType(), True),
       StructField("timestamp", TimestampType(), True)
   ])
   
   # Define streaming source
   streaming_df = spark.readStream \
       .format("csv") \
       .option("header", "true") \
       .option("inferSchema", "true") \
       .schema(news_schema) \
       .load("/home/ubuntu/fake_news_detection/data/streaming_input")
   ```

3. **Process Streaming Data**:
   ```python
   # Preprocess streaming data
   streaming_df = streaming_df.withColumn("text", preprocess_text(col("text")))
   
   # Load model
   model = load_model(spark, "/home/ubuntu/fake_news_detection/models/best_model")
   
   # Apply model to streaming data
   predictions = model.transform(streaming_df) \
       .select("id", "text", "prediction", "timestamp") \
       .withColumn("prediction_time", current_timestamp())
   ```

4. **Write Streaming Results**:
   ```python
   # Write streaming predictions to output
   query = predictions.writeStream \
       .format("csv") \
       .option("path", "/home/ubuntu/fake_news_detection/data/streaming_output") \
       .option("checkpointLocation", "/home/ubuntu/fake_news_detection/data/checkpoints") \
       .outputMode("append") \
       .start()
   ```

#### 3.2.2 Databricks Environment

1. **Create Directory for Streaming Input**:
   ```python
   dbutils.fs.mkdirs("/FileStore/streaming_input")
   ```

2. **Upload Streaming Data**:
   ```python
   # Upload stream1.csv to DBFS
   dbutils.fs.cp("/path/to/stream1.csv", "dbfs:/FileStore/streaming_input/stream1.csv")
   ```

3. **Set Up Streaming Source**:
   ```python
   # Define schema for streaming data
   news_schema = StructType([
       StructField("id", StringType(), True),
       StructField("text", StringType(), True),
       StructField("timestamp", TimestampType(), True)
   ])
   
   # Define streaming source
   streaming_df = spark.readStream \
       .format("csv") \
       .option("header", "true") \
       .option("inferSchema", "true") \
       .schema(news_schema) \
       .load("/FileStore/streaming_input")
   ```

4. **Process Streaming Data**:
   ```python
   # Preprocess streaming data
   streaming_df = streaming_df.withColumn("text", lower(regexp_replace("text", "[^a-zA-Z\s]", "")))
   
   # Load model
   from pyspark.ml.pipeline import PipelineModel
   model = PipelineModel.load("/FileStore/models/fake_news_best_model")
   
   # Apply model to streaming data
   predictions = model.transform(streaming_df) \
       .select("id", "text", "prediction", "timestamp") \
       .withColumn("prediction_time", current_timestamp())
   ```

5. **Write to Delta Table**:
   ```python
   # Write streaming predictions to Delta table
   query = predictions.writeStream \
       .format("delta") \
       .option("checkpointLocation", "/FileStore/checkpoints/fake_news") \
       .outputMode("append") \
       .table("fake_news_predictions")
   ```

## 4. Data Quality and Validation

### 4.1 Data Quality Checks

1. **Check for Missing Values**:
   ```python
   # Count missing values
   missing_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
   missing_counts.show()
   ```

2. **Check Data Distribution**:
   ```python
   # Check label distribution
   df.groupBy("label").count().show()
   
   # Check text length distribution
   from pyspark.sql.functions import length
   df.select(length("text").alias("text_length")).summary().show()
   ```

3. **Validate Text Content**:
   ```python
   # Sample and display text content
   df.select("text", "label").sample(0.01).show(10, truncate=50)
   ```

### 4.2 Data Validation in Streaming Pipeline

1. **Schema Validation**:
   ```python
   # Validate schema
   assert streaming_df.schema.fieldNames() == ["id", "text", "timestamp"], "Schema mismatch"
   ```

2. **Data Type Validation**:
   ```python
   # Validate data types
   assert streaming_df.schema["id"].dataType == StringType(), "ID should be string"
   assert streaming_df.schema["text"].dataType == StringType(), "Text should be string"
   assert streaming_df.schema["timestamp"].dataType == TimestampType(), "Timestamp should be timestamp"
   ```

3. **Content Validation**:
   ```python
   # Validate text content
   empty_texts = streaming_df.filter(length("text") == 0).count()
   assert empty_texts == 0, "Empty texts detected"
   ```

## 5. Data Monitoring and Logging

### 5.1 Offline Data Monitoring

1. **Dataset Analysis**:
   ```python
   from utils.dataset_utils import analyze_dataset
   
   # Analyze dataset
   stats = analyze_dataset(df, "/home/ubuntu/fake_news_detection/logs/dataset_analysis.json")
   ```

2. **Data Processing Logs**:
   ```python
   # Log data processing steps
   with open("/home/ubuntu/fake_news_detection/logs/data_processing.log", "w") as f:
       f.write(f"Total records: {df.count()}\n")
       f.write(f"Records after preprocessing: {df_preprocessed.count()}\n")
       f.write(f"Records after augmentation: {df_augmented.count()}\n")
       f.write(f"Training records: {train_data.count()}\n")
       f.write(f"Testing records: {test_data.count()}\n")
   ```

### 5.2 Streaming Data Monitoring

1. **Monitor Streaming Query**:
   ```python
   # Print streaming query status
   print(query.status)
   
   # Print recent progress
   print(query.recentProgress)
   ```

2. **Log Streaming Metrics**:
   ```python
   # Log streaming metrics
   def log_streaming_metrics(batch_id, time):
       counts = spark.sql("SELECT prediction, COUNT(*) as count FROM fake_news_predictions GROUP BY prediction")
       with open("/home/ubuntu/fake_news_detection/logs/streaming_metrics.log", "a") as f:
           f.write(f"Batch ID: {batch_id}, Time: {time}\n")
           for row in counts.collect():
               label = "Real" if row["prediction"] == 1.0 else "Fake"
               f.write(f"  {label}: {row['count']}\n")
   
   # Register callback
   query.foreachBatch(log_streaming_metrics)
   ```

## 6. Data Retention and Archiving

### 6.1 Local Environment

1. **Archive Processed Data**:
   ```python
   import shutil
   import datetime
   
   # Archive processed data
   archive_date = datetime.datetime.now().strftime("%Y%m%d")
   archive_dir = f"/home/ubuntu/fake_news_detection/data/archive/{archive_date}"
   os.makedirs(archive_dir, exist_ok=True)
   
   # Copy processed data to archive
   shutil.copy("/home/ubuntu/fake_news_detection/data/processed/train_data.parquet", archive_dir)
   shutil.copy("/home/ubuntu/fake_news_detection/data/processed/test_data.parquet", archive_dir)
   ```

2. **Archive Streaming Results**:
   ```python
   # Archive streaming results
   streaming_archive_dir = f"/home/ubuntu/fake_news_detection/data/streaming_archive/{archive_date}"
   os.makedirs(streaming_archive_dir, exist_ok=True)
   
   # Copy streaming results to archive
   for file in os.listdir("/home/ubuntu/fake_news_detection/data/streaming_output"):
       if file.endswith(".csv"):
           shutil.copy(f"/home/ubuntu/fake_news_detection/data/streaming_output/{file}", streaming_archive_dir)
   ```

### 6.2 Databricks Environment

1. **Archive Delta Tables**:
   ```python
   # Archive Delta tables
   archive_date = datetime.datetime.now().strftime("%Y%m%d")
   
   # Create archive tables
   spark.sql(f"CREATE TABLE IF NOT EXISTS train_features_archive_{archive_date} AS SELECT * FROM train_features")
   spark.sql(f"CREATE TABLE IF NOT EXISTS test_features_archive_{archive_date} AS SELECT * FROM test_features")
   spark.sql(f"CREATE TABLE IF NOT EXISTS fake_news_predictions_archive_{archive_date} AS SELECT * FROM fake_news_predictions")
   ```

2. **Implement Retention Policy**:
   ```python
   # List archive tables
   archive_tables = [table for table in spark.sql("SHOW TABLES").collect() if "archive" in table["tableName"]]
   
   # Keep only the last 5 archives
   if len(archive_tables) > 5:
       # Sort by date
       archive_tables.sort(key=lambda x: x["tableName"].split("_")[-1], reverse=True)
       
       # Drop oldest archives
       for table in archive_tables[5:]:
           spark.sql(f"DROP TABLE IF EXISTS {table['tableName']}")
   ```

## 7. Troubleshooting Common Issues

### 7.1 Data Loading Issues

- **Issue**: CSV parsing errors
  - **Solution**: Check CSV format, ensure proper encoding, and handle special characters
  
- **Issue**: Memory errors when loading large datasets
  - **Solution**: Increase Spark driver and executor memory, or process data in chunks

### 7.2 Data Processing Issues

- **Issue**: Slow text preprocessing
  - **Solution**: Optimize preprocessing functions, use UDFs efficiently, or consider sampling for development

- **Issue**: Imbalanced dataset
  - **Solution**: Use `create_balanced_dataset` function with appropriate method

### 7.3 Streaming Issues

- **Issue**: Streaming source not found
  - **Solution**: Verify directory paths and permissions

- **Issue**: Streaming query fails
  - **Solution**: Check checkpoint directory, ensure schema consistency, and verify output path

## 8. Best Practices

1. **Data Versioning**: Keep track of dataset versions and transformations
2. **Incremental Processing**: Process new data incrementally when possible
3. **Schema Evolution**: Design for schema evolution in streaming pipelines
4. **Data Validation**: Implement thorough data validation checks
5. **Error Handling**: Implement robust error handling for data processing
6. **Documentation**: Document data sources, transformations, and quality checks
7. **Monitoring**: Set up monitoring for data quality and processing metrics
8. **Security**: Ensure data security and privacy compliance

## 9. Conclusion

This document provides a comprehensive guide to data ingestion for the Fake News Detection system. By following these steps and best practices, you can ensure reliable, reproducible, and efficient data processing for both offline model training and online streaming classification.

# Last modified: May 29, 2025
