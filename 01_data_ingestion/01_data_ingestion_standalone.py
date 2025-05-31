# %% [markdown]
# # Fake News Detection: Data Ingestion
# 
# This notebook contains all the necessary code to load, process, and prepare data for the fake news detection project. The code is organized into independent functions, without dependencies on external modules or classes, to facilitate execution in Databricks Community Edition.

# %% [markdown]
# ## Setup and Imports

# %%
# Import necessary libraries
import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, lower, regexp_replace, rand, when, concat
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Initialize Spark session with Hive support
spark = SparkSession.builder \
    .appName("FakeNewsDetection") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

# Show Spark version
print(f"Spark version: {spark.version}")
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")

# %% [markdown]
# ## Directory Structure Setup

# %%
def create_directory_structure(base_dir="/dbfs/FileStore/fake_news_detection"):
    """
    Creates the necessary directory structure for the fake news detection project.
    
    This function ensures all required directories exist in the Databricks environment.
    It's essential to run this function before executing the rest of the pipeline.
    
    Args:
        base_dir (str): Base directory for the project
        
    Returns:
        dict: Dictionary with paths to all created directories
    """
    print(f"Creating directory structure in {base_dir}...")
    
    # Define directory paths
    directories = {
        "data": f"{base_dir}/data",
        "raw_data": f"{base_dir}/data/raw",
        "processed_data": f"{base_dir}/data/processed",
        "sample_data": f"{base_dir}/data/sample",
        "models": f"{base_dir}/models",
        "logs": f"{base_dir}/logs",
        "visualizations": f"{base_dir}/visualizations",
        "temp": f"{base_dir}/temp"
    }
    
    # Create directories
    for dir_name, dir_path in directories.items():
        # Use dbutils in Databricks environment
        try:
            dbutils.fs.mkdirs(dir_path)
            print(f"Created directory: {dir_path}")
        except NameError:
            # Fallback for non-Databricks environments
            os.makedirs(dir_path.replace("/dbfs", ""), exist_ok=True)
            print(f"Created directory: {dir_path} (local mode)")
    
    print("Directory structure created successfully")
    return directories

# %% [markdown]
# ## Reusable Functions

# %% [markdown]
# ### Data Loading Functions

# %%
def load_csv_files(fake_path, true_path, cache=True):
    """
    Loads CSV files containing fake and true news articles.
    
    Args:
        fake_path (str): Path to the CSV file with fake news
        true_path (str): Path to the CSV file with true news
        cache (bool): Whether to cache the DataFrames in memory
        
    Returns:
        tuple: (fake_df, true_df) DataFrames with loaded data
    """
    print(f"Loading CSV files from {fake_path} and {true_path}...")
    
    # Load CSV files
    fake_df = spark.read.csv(fake_path, header=True, inferSchema=True)
    true_df = spark.read.csv(true_path, header=True, inferSchema=True)
    
    # Add labels (0 for fake, 1 for true)
    fake_df = fake_df.withColumn("label", lit(0))
    true_df = true_df.withColumn("label", lit(1))
    
    # Cache DataFrames if requested (improves performance for multiple operations)
    if cache:
        fake_df.cache()
        true_df.cache()
    
    # Show information about the DataFrames
    print(f"Fake news loaded: {fake_df.count()} records")
    print(f"True news loaded: {true_df.count()} records")
    
    # Analyze subject distribution for data leakage detection
    analyze_subject_distribution(fake_df, true_df)
    
    return fake_df, true_df

# %%
def analyze_subject_distribution(fake_df, true_df):
    """
    Analyzes the distribution of subjects in fake and true news datasets to detect potential data leakage.
    
    Args:
        fake_df: DataFrame with fake news
        true_df: DataFrame with true news
    """
    print("\nAnalyzing subject distribution for potential data leakage...")
    
    # Check if subject column exists in both DataFrames
    if "subject" in fake_df.columns and "subject" in true_df.columns:
        # Get subject distribution in fake news
        print("Subject distribution in fake news:")
        fake_subjects = fake_df.groupBy("subject").count().orderBy("count", ascending=False)
        fake_subjects.show(truncate=False)
        
        # Get subject distribution in true news
        print("Subject distribution in true news:")
        true_subjects = true_df.groupBy("subject").count().orderBy("count", ascending=False)
        true_subjects.show(truncate=False)
        
        # Check for potential data leakage
        print("\nDATA LEAKAGE WARNING:")
        print("The 'subject' column may cause data leakage as it perfectly separates fake from true news.")
        print("Fake news articles are predominantly labeled with subjects like 'News', while")
        print("true news articles are labeled with subjects like 'politicsNews'.")
        print("This column should be removed before model training to prevent unrealistic performance.")
    else:
        print("No 'subject' column found in the datasets.")

# %%
def create_hive_tables(fake_df, true_df, fake_table_name="fake", true_table_name="real"):
    """
    Creates Hive tables for fake and true news DataFrames.
    
    Args:
        fake_df: DataFrame with fake news
        true_df: DataFrame with true news
        fake_table_name (str): Name of the Hive table for fake news
        true_table_name (str): Name of the Hive table for true news
    """
    print(f"Creating Hive tables '{fake_table_name}' and '{true_table_name}'...")
    
    # Create table for fake news
    spark.sql(f"DROP TABLE IF EXISTS {fake_table_name}")
    fake_df.write.mode("overwrite").saveAsTable(fake_table_name)
    print(f"Table '{fake_table_name}' created successfully")
    
    # Create table for true news
    spark.sql(f"DROP TABLE IF EXISTS {true_table_name}")
    true_df.write.mode("overwrite").saveAsTable(true_table_name)
    print(f"Table '{true_table_name}' created successfully")
    
    # Verify that tables were created correctly
    print("\nAvailable tables in catalog:")
    spark.sql("SHOW TABLES").show()

# %%
def load_data_from_hive(fake_table_name="fake", true_table_name="real", cache=True):
    """
    Loads data from Hive tables.
    
    Args:
        fake_table_name (str): Name of the Hive table with fake news
        true_table_name (str): Name of the Hive table with true news
        cache (bool): Whether to cache the DataFrames in memory
        
    Returns:
        tuple: (true_df, fake_df) DataFrames with loaded data
    """
    print(f"Loading data from Hive tables '{true_table_name}' and '{fake_table_name}'...")
    
    # Check if tables exist
    tables = [row.tableName for row in spark.sql("SHOW TABLES").collect()]
    
    if true_table_name not in tables or fake_table_name not in tables:
        raise ValueError(f"Hive tables '{true_table_name}' and/or '{fake_table_name}' do not exist")
    
    # Load data from Hive tables
    true_df = spark.table(true_table_name)
    fake_df = spark.table(fake_table_name)
    
    # Cache DataFrames if requested
    if cache:
        true_df.cache()
        fake_df.cache()
    
    # Register as temporary views for SQL queries
    true_df.createOrReplaceTempView("true_news")
    fake_df.createOrReplaceTempView("fake_news")
    
    # Show information about the DataFrames
    print(f"True news loaded: {true_df.count()} records")
    print(f"Fake news loaded: {fake_df.count()} records")
    
    return true_df, fake_df

# %% [markdown]
# ### Data Processing Functions

# %%
def combine_datasets(true_df, fake_df, cache=True):
    """
    Combines DataFrames of true and fake news.
    
    Args:
        true_df: DataFrame with true news
        fake_df: DataFrame with fake news
        cache (bool): Whether to cache the combined DataFrame
        
    Returns:
        DataFrame: Combined DataFrame
    """
    print("Combining true and fake news datasets...")
    
    # Check available columns
    true_cols = set(true_df.columns)
    fake_cols = set(fake_df.columns)
    common_cols = true_cols.intersection(fake_cols)
    
    print(f"Common columns: {common_cols}")
    
    # Select common columns to ensure compatibility
    if "title" in common_cols and "text" in common_cols:
        # If we have title and text, combine for better context
        true_df = true_df.select("title", "text", "label")
        fake_df = fake_df.select("title", "text", "label")
        
        # Combine title and text for better context
        true_df = true_df.withColumn("full_text", 
                                    concat(col("title"), lit(". "), col("text")))
        fake_df = fake_df.withColumn("full_text", 
                                    concat(col("title"), lit(". "), col("text")))
        
        # Select final columns
        true_df = true_df.select("full_text", "label")
        fake_df = fake_df.select("full_text", "label")
        
        # Rename column
        true_df = true_df.withColumnRenamed("full_text", "text")
        fake_df = fake_df.withColumnRenamed("full_text", "text")
    else:
        # Otherwise, just use text and label
        true_df = true_df.select("text", "label")
        fake_df = fake_df.select("text", "label")
    
    # Combine datasets
    combined_df = true_df.unionByName(fake_df)
    
    # Cache the combined DataFrame if requested
    if cache:
        combined_df.cache()
    
    # Show information about the combined DataFrame
    print(f"Combined dataset: {combined_df.count()} records")
    print(f"Label distribution:")
    combined_df.groupBy("label").count().show()
    
    # Unpersist individual DataFrames to free up memory
    true_df.unpersist()
    fake_df.unpersist()
    
    return combined_df

# %%
def preprocess_text(df, cache=True):
    """
    Preprocesses text by converting to lowercase and removing special characters.
    Also checks for and removes problematic columns that may cause data leakage.
    
    Args:
        df: DataFrame with text column
        cache (bool): Whether to cache the preprocessed DataFrame
        
    Returns:
        DataFrame: DataFrame with preprocessed text
    """
    print("Preprocessing text...")
    
    # Convert to lowercase
    df = df.withColumn("text", lower(col("text")))
    
    # Remove special characters
    df = df.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z0-9\\s]", " "))
    
    # Remove multiple spaces
    df = df.withColumn("text", regexp_replace(col("text"), "\\s+", " "))
    
    # Check for problematic columns that may cause data leakage
    if "subject" in df.columns:
        print("\nWARNING: Removing 'subject' column to prevent data leakage")
        print("The 'subject' column perfectly discriminates between true and fake news")
        print("True news: subject='politicsNews', Fake news: subject='News'")
        df = df.drop("subject")
        print("'subject' column successfully removed")
    
    # Cache the preprocessed DataFrame if requested
    if cache:
        df.cache()
    
    return df

# %%
def create_balanced_sample(df, sample_size=1000, seed=42, cache=True):
    """
    Creates a balanced sample of the dataset.
    
    Args:
        df: DataFrame with data
        sample_size (int): Sample size for each class
        seed (int): Seed for reproducibility
        cache (bool): Whether to cache the sample DataFrame
        
    Returns:
        DataFrame: Balanced sample
    """
    print(f"Creating balanced sample with {sample_size} records per class...")
    
    # Sample of true news (label=1)
    real_sample = df.filter(col("label") == 1) \
                    .orderBy(rand(seed=seed)) \
                    .limit(sample_size)
    
    # Sample of fake news (label=0)
    fake_sample = df.filter(col("label") == 0) \
                    .orderBy(rand(seed=seed)) \
                    .limit(sample_size)
    
    # Combine the samples
    sample_df = real_sample.unionByName(fake_sample)
    
    # Cache the sample DataFrame if requested
    if cache:
        sample_df.cache()
    
    # Register the sample DataFrame as a temporary view
    sample_df.createOrReplaceTempView("sample_news")
    
    # Show sample statistics
    print("\nSample statistics:")
    spark.sql("""
        SELECT 
            label, 
            COUNT(*) as count
        FROM sample_news
        GROUP BY label
        ORDER BY label DESC
    """).show()
    
    return sample_df

# %% [markdown]
# ### Data Storage Functions

# %%
def save_to_parquet(df, path, partition_by=None):
    """
    Saves a DataFrame in Parquet format.
    
    Args:
        df: DataFrame to save
        path (str): Path where to save the DataFrame
        partition_by (str): Column to partition by (optional)
    """
    print(f"Saving DataFrame to {path}...")
    
    writer = df.write.mode("overwrite")
    
    if partition_by:
        writer = writer.partitionBy(partition_by)
    
    writer.parquet(path)
    print(f"DataFrame saved to {path}")

# %%
def save_to_hive_table(df, table_name, partition_by=None):
    """
    Saves a DataFrame to a Hive table.
    
    Args:
        df: DataFrame to save
        table_name (str): Name of the Hive table to create or replace
        partition_by (str): Column to partition by (optional)
    """
    print(f"Saving DataFrame to Hive table {table_name}...")
    
    writer = df.write.mode("overwrite").format("parquet")
    
    if partition_by:
        writer = writer.partitionBy(partition_by)
    
    writer.saveAsTable(table_name)
    print(f"DataFrame saved to Hive table: {table_name}")

# %% [markdown]
# ### Data Analysis Functions

# %%
def analyze_dataset_characteristics(df):
    """
    Analyzes dataset characteristics to identify potential issues.
    
    Args:
        df: DataFrame with text and label columns
        
    Returns:
        dict: Dictionary with analysis results
    """
    print("Analyzing dataset characteristics...")
    
    # Convert to pandas for easier analysis
    pandas_df = df.toPandas()
    
    # Calculate basic statistics
    total_samples = len(pandas_df)
    class_distribution = pandas_df['label'].value_counts().to_dict()
    class_balance = min(class_distribution.values()) / max(class_distribution.values())
    
    # Calculate text length statistics
    pandas_df['text_length'] = pandas_df['text'].apply(len)
    avg_text_length = pandas_df['text_length'].mean()
    min_text_length = pandas_df['text_length'].min()
    max_text_length = pandas_df['text_length'].max()
    
    # Check for empty or very short texts
    short_texts = (pandas_df['text_length'] < 10).sum()
    
    # Check for duplicate texts
    duplicate_texts = pandas_df['text'].duplicated().sum()
    
    # Compile results
    results = {
        'total_samples': total_samples,
        'class_distribution': class_distribution,
        'class_balance': class_balance,
        'avg_text_length': avg_text_length,
        'min_text_length': min_text_length,
        'max_text_length': max_text_length,
        'short_texts': short_texts,
        'duplicate_texts': duplicate_texts
    }
    
    # Print summary
    print("Dataset Characteristics:")
    print(f"Total samples: {total_samples}")
    print(f"Class distribution: {class_distribution}")
    print(f"Class balance ratio: {class_balance:.2f}")
    print(f"Average text length: {avg_text_length:.2f} characters")
    print(f"Text length range: {min_text_length} to {max_text_length} characters")
    print(f"Number of very short texts (<10 chars): {short_texts}")
    print(f"Number of duplicate texts: {duplicate_texts}")
    
    # Create plots
    plt.figure(figsize=(12, 5))
    
    # Class distribution plot
    plt.subplot(1, 2, 1)
    sns.countplot(x='label', data=pandas_df)
    plt.title('Class Distribution')
    plt.xlabel('Class (0=Fake, 1=True)')
    plt.ylabel('Count')
    
    # Text length distribution plot
    plt.subplot(1, 2, 2)
    sns.histplot(pandas_df['text_length'], bins=30)
    plt.title('Text Length Distribution')
    plt.xlabel('Length (characters)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    return results

# %% [markdown]
# ## Complete Data Ingestion Pipeline

# %%
def process_and_save_data(fake_path="/FileStore/tables/fake.csv", 
                         true_path="/FileStore/tables/real.csv",
                         output_dir="dbfs:/FileStore/fake_news_detection/data",
                         create_tables=True):
    """
    Processes and saves fake and true news data.
    
    This complete pipeline loads CSV data, combines datasets, creates samples,
    and saves results in Parquet format and as Hive tables.
    
    Args:
        fake_path (str): Path to the CSV file with fake news
        true_path (str): Path to the CSV file with true news
        output_dir (str): Directory to save processed data
        create_tables (bool): Whether to create Hive tables
        
    Returns:
        dict: Dictionary with references to processed DataFrames
    """
    print("Starting data processing pipeline...")
    
    # 0. Create directory structure
    directories = create_directory_structure()
    
    # 1. Load CSV files
    fake_df, true_df = load_csv_files(fake_path, true_path)
    
    # 2. Create Hive tables (optional)
    if create_tables:
        create_hive_tables(fake_df, true_df)
    
    # 3. Combine datasets
    combined_df = combine_datasets(true_df, fake_df)
    
    # 4. Preprocess text
    combined_df = preprocess_text(combined_df)
    
    # 5. Create balanced sample
    sample_df = create_balanced_sample(combined_df)
    
    # 6. Analyze dataset characteristics
    analyze_dataset_characteristics(combined_df)
    
    # 7. Save combined dataset to DBFS
    combined_path = f"{output_dir}/combined_data/combined_news.parquet"
    save_to_parquet(combined_df, combined_path, partition_by="label")
    
    # 8. Save sample to DBFS
    sample_path = f"{output_dir}/sample_data/sample_news.parquet"
    save_to_parquet(sample_df, sample_path)
    
    # 9. Save to Hive tables for easier access
    save_to_hive_table(combined_df, "combined_news", partition_by="label")
    save_to_hive_table(sample_df, "sample_news")
    
    # 10. Unpersist DataFrames to free up memory
    combined_df.unpersist()
    sample_df.unpersist()
    
    print("\nData processing pipeline completed successfully!")
    
    return {
        "true_df": true_df,
        "fake_df": fake_df,
        "combined_df": combined_df,
        "sample_df": sample_df,
        "directories": directories
    }

# %% [markdown]
# ## Memory Management Best Practices

# %%
def optimize_memory_usage():
    """
    Displays best practices for memory management in Databricks Community Edition.
    """
    print("Memory Management Best Practices for Databricks Community Edition:")
    print("\n1. Cache and Unpersist Strategy:")
    print("   - Cache DataFrames only when they will be reused multiple times")
    print("   - Always unpersist DataFrames when they are no longer needed")
    print("   - Monitor memory usage with Spark UI")
    
    print("\n2. Partition Management:")
    print("   - Use appropriate number of partitions (8-16 for Community Edition)")
    print("   - Repartition large DataFrames to avoid memory issues")
    print("   - Use coalesce() for reducing partitions without shuffle")
    
    print("\n3. Column Pruning:")
    print("   - Select only necessary columns as early as possible")
    print("   - Drop unnecessary columns to reduce memory footprint")
    
    print("\n4. Checkpointing:")
    print("   - Use checkpointing for complex operations to truncate lineage")
    print("   - Set checkpoint directory with spark.sparkContext.setCheckpointDir()")
    
    print("\n5. Broadcast Variables:")
    print("   - Use broadcast variables for small lookup tables")
    print("   - Example: broadcast(small_df).value for joins")
    
    print("\n6. Garbage Collection:")
    print("   - Monitor GC with spark.conf.get('spark.executor.extraJavaOptions')")
    print("   - Consider adding -XX:+PrintGCDetails to Java options")
    
    print("\nImplemented in this notebook:")
    print("- Strategic caching of DataFrames")
    print("- Explicit unpersist calls when DataFrames are no longer needed")
    print("- Early column selection to reduce memory footprint")
    print("- Appropriate partition management")

# %% [markdown]
# ## Step-by-Step Tutorial

# %% [markdown]
# ### 1. Set Up Directory Structure

# %%
# Create necessary directories
directories = create_directory_structure()
print(f"Working with directories: {directories}")

# %% [markdown]
# ### 2. Load CSV Data

# %%
# Define paths to CSV files
# Note: Adjust paths as needed for your environment
fake_path = "/FileStore/tables/fake.csv"
true_path = "/FileStore/tables/real.csv"

# Load the CSV files
fake_df, true_df = load_csv_files(fake_path, true_path)

# %% [markdown]
# ### 3. Create Hive Tables

# %%
# Create Hive tables
create_hive_tables(fake_df, true_df)

# %% [markdown]
# ### 4. Combine Datasets

# %%
# Combine true and fake news datasets
combined_df = combine_datasets(true_df, fake_df)

# %% [markdown]
# ### 5. Preprocess Text

# %%
# Preprocess text data
combined_df = preprocess_text(combined_df)

# %% [markdown]
# ### 6. Create Balanced Sample

# %%
# Create a balanced sample
sample_df = create_balanced_sample(combined_df, sample_size=1000)

# %% [markdown]
# ### 7. Analyze Dataset Characteristics

# %%
# Analyze dataset characteristics
analysis_results = analyze_dataset_characteristics(combined_df)

# %% [markdown]
# ### 8. Save Data to Parquet

# %%
# Save combined dataset to Parquet
save_to_parquet(combined_df, f"{directories['processed_data']}/combined_news.parquet", partition_by="label")

# Save sample to Parquet
save_to_parquet(sample_df, f"{directories['sample_data']}/sample_news.parquet")

# %% [markdown]
# ### 9. Save to Hive Tables

# %%
# Save to Hive tables
save_to_hive_table(combined_df, "combined_news", partition_by="label")
save_to_hive_table(sample_df, "sample_news")

# %% [markdown]
# ### 10. Clean Up Memory

# %%
# Unpersist DataFrames to free up memory
combined_df.unpersist()
sample_df.unpersist()
fake_df.unpersist()
true_df.unpersist()

# Display memory management best practices
optimize_memory_usage()

# %% [markdown]
# ## Run Complete Pipeline

# %%
# Run the complete data ingestion pipeline
results = process_and_save_data(
    fake_path="/FileStore/tables/fake.csv",
    true_path="/FileStore/tables/real.csv",
    output_dir="dbfs:/FileStore/fake_news_detection/data",
    create_tables=True
)

# %% [markdown]
# ## Important Notes
# 
# 1. **Directory Structure**: The `create_directory_structure()` function must be called before running the pipeline to ensure all necessary directories exist.
# 
# 2. **Data Leakage**: The 'subject' column in the original datasets perfectly separates fake from true news, which would cause data leakage. This column is automatically removed during preprocessing.
# 
# 3. **Memory Management**: Databricks Community Edition has limited memory. The functions in this notebook implement best practices for memory management:
#    - Strategic caching of DataFrames
#    - Explicit unpersist calls when DataFrames are no longer needed
#    - Early column selection to reduce memory footprint
# 
# 4. **File Paths**: The default paths assume files are uploaded to Databricks FileStore. Adjust paths as needed for your environment.
# 
# 5. **Hive Tables**: Creating Hive tables is optional but recommended for easier data access in subsequent notebooks.
# 
# 6. **Balanced Sample**: A balanced sample is created for exploratory analysis and initial model development. The full dataset should be used for final model training.
# 
# 7. **Partitioning**: Data is partitioned by label when saved to improve query performance when filtering by class.
# 
# 8. **Reproducibility**: A fixed seed is used for sampling to ensure reproducible results.
