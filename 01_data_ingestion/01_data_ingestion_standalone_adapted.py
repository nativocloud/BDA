# %% [markdown]
# # Data Ingestion for Fake News Detection

# %% [markdown]
# ## Why Data Ingestion is Important
# 
# In fake news detection, proper data ingestion ensures:
# 1. Data quality and consistency
# 2. Appropriate labeling of real and fake news articles
# 3. Balanced representation of both classes
# 4. Efficient storage for distributed processing

# %% [markdown]
# ## Setup and Imports
# 
# First, let's set up our Spark environment and import the necessary libraries.

# %%
# Import required libraries
import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, lower, regexp_replace, rand, when, concat
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Creating a Spark Session with Hive Support
# 
# We'll use Apache Spark for distributed data processing, with Hive support enabled to access the metastore tables. Let's create a properly configured Spark session optimized for the Databricks Community Edition limitations (1 driver, 15.3 GB Memory, 2 Cores).

# %%
# Create a Spark session with configuration optimized for Databricks Community Edition
# - appName: Identifies this application in the Spark UI and logs
# - spark.sql.shuffle.partitions: Set to 8 (4x number of cores) for Community Edition
# - spark.driver.memory: Set to 8g to utilize available memory while leaving room for system
# - enableHiveSupport: Enables access to Hive metastore tables
spark = SparkSession.builder \
    .appName("FakeNewsDetection") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

# %% [markdown]
# ## Create directory structure

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
# ## Custom functions

# %% [markdown]
# ### Data Loading functions

# %% [markdown]
# #### Load CSV files

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
        # Force materialization
        fake_count = fake_df.count()
        true_count = true_df.count()
    
    # Show information about the DataFrames
    print(f"Fake news loaded: {fake_df.count()} records")
    print(f"True news loaded: {true_df.count()} records")
    
    return fake_df, true_df

# %% [markdown]
# #### Analyse subject distribution

# %%
from pyspark.sql.functions import col, count, lit
# Alias PySpark's min/max to avoid conflict with Python's built-in min/max
from pyspark.sql.functions import min as spark_min, max as spark_max
from pyspark.sql.types import StringType # For type checking
import builtins  # For Python's built-in min/max functions

# spark (SparkSession) is assumed to be globally available in Databricks

def analyze_subject_distribution(fake_df, true_df):
    """
    Analyzes the distribution of subjects in fake and true news datasets to detect potential data leakage.
    Provides essential checks using native PySpark functionality.
    Optimized for Databricks environment with display() visualizations by minimizing collect() calls.

    Args:
        fake_df: DataFrame with fake news.
        true_df: DataFrame with true news.

    Returns:
        None (This function prints analysis directly to the Databricks notebook output).
    """
    print("\n" + "="*80)
    print("📊 SUBJECT DISTRIBUTION ANALYSIS")
    print("="*80)

    # --- Initial Checks for Robustness ---
    # Calculate total counts upfront (this is a necessary action)
    fake_total = fake_df.count()
    true_total = true_df.count()

    if fake_total == 0:
        print("\n⚠️ Fake news DataFrame is empty. Analysis cannot proceed.")
        return
    if true_total == 0:
        print("\n⚠️ True news DataFrame is empty. Analysis cannot proceed.")
        return

    # Check if 'subject' column exists in both DataFrames
    if "subject" not in fake_df.columns or "subject" not in true_df.columns:
        print("\n⚠️ 'subject' column not found in one or both datasets. Analysis cannot proceed.")
        print(f"  Fake DF columns: {fake_df.columns}")
        print(f"  True DF columns: {true_df.columns}")
        return
    
    # Check if 'subject' column is of a string type for proper analysis
    fake_subject_type = fake_df.schema["subject"].dataType
    true_subject_type = true_df.schema["subject"].dataType
    if not isinstance(fake_subject_type, StringType) or not isinstance(true_subject_type, StringType):
        print(f"\n⚠️ 'subject' column expected to be 'string' type for distribution analysis, but found '{fake_subject_type.typeName()}' in fake_df and '{true_subject_type.typeName()}' in true_df. Analysis cannot proceed.")
        return

    # --- Step 1 & 2: Get and Display Subject Distributions ---
   
    print("\n1️⃣ FAKE NEWS SUBJECT DISTRIBUTION")
    fake_subjects_df = fake_df.groupBy("subject").count().orderBy(col("count").desc())
    print("• Subject distribution in fake news:")
    display(fake_subjects_df)

    print("\n2️⃣ TRUE NEWS SUBJECT DISTRIBUTION")
    true_subjects_df = true_df.groupBy("subject").count().orderBy(col("count").desc())
    print("• Subject distribution in true news:")
    display(true_subjects_df)

    # --- Step 3: Subject Overlap Analysis ---
    print("\n3️⃣ SUBJECT OVERLAP ANALYSIS")

    # Get total unique subjects in each dataset directly in Spark
    num_fake_unique_subjects = fake_subjects_df.count()
    num_true_unique_subjects = true_subjects_df.count()

    # Find common subjects 
    common_subjects_df = fake_subjects_df.join(true_subjects_df, on="subject", how="inner")
    num_common_subjects = common_subjects_df.count()

    # Find subjects exclusive to fake news using left_anti join
    fake_exclusive_df = fake_subjects_df.join(true_subjects_df, on="subject", how="left_anti")
    num_fake_exclusive = fake_exclusive_df.count()

    # Find subjects exclusive to true news using right_anti join (or left_anti with roles swapped)
    true_exclusive_df = true_subjects_df.join(fake_subjects_df, on="subject", how="left_anti") 
    # returns only the rows from the left DataFrame that have no match in the right DataFrame.
    num_true_exclusive = true_exclusive_df.count()

    print(f"• Total unique subjects in fake news: {num_fake_unique_subjects}")
    print(f"• Total unique subjects in true news: {num_true_unique_subjects}")
    print(f"• Subjects common to both datasets: {num_common_subjects}")
    print(f"• Subjects exclusive to fake news: {num_fake_exclusive}")
    print(f"• Subjects exclusive to true news: {num_true_exclusive}")

    # Create a comparison view for common subjects
    if num_common_subjects > 0:
        print("\n• Distribution of common subjects (count and percentage):")

        # Create temporary views for SQL query 
        # Note: Using distinct temp view names to avoid conflicts if the notebook runs multiple times
        fake_df.createOrReplaceTempView("fake_news_temp_view_subject_analysis")
        true_df.createOrReplaceTempView("true_news_temp_view_subject_analysis")

        # SQL query to compare subject distributions and their percentages
        comparison_query = f"""
        SELECT
            f.subject,
            f.count AS fake_count,
            t.count AS true_count,
            CAST(f.count AS DOUBLE) / {fake_total} * 100 AS fake_percentage,
            CAST(t.count AS DOUBLE) / {true_total} * 100 AS true_percentage
        FROM
            (SELECT subject, COUNT(*) AS count FROM fake_news_temp_view_subject_analysis GROUP BY subject) f
        JOIN
            (SELECT subject, COUNT(*) AS count FROM true_news_temp_view_subject_analysis GROUP BY subject) t
        ON
            f.subject = t.subject
        ORDER BY
            ABS((CAST(f.count AS DOUBLE) / {fake_total} * 100) - (CAST(t.count AS DOUBLE) / {true_total} * 100)) DESC
        """

        comparison_df = spark.sql(comparison_query)
        display(comparison_df)
    else:
        print("\n• No common subjects found between fake and true news datasets, skipping detailed comparison table.")

    # --- Step 4: Data Leakage Assessment ---
    print("\n4️⃣ DATA LEAKAGE ASSESSMENT")
    
    # Check if there's a perfect separation by subject
    if num_common_subjects == 0 and num_fake_unique_subjects > 0 and num_true_unique_subjects > 0:
        print("\n🚨 HIGH RISK OF DATA LEAKAGE DETECTED!")
        print("• The 'subject' column perfectly separates fake and true news articles.")
        print("• This is a clear case of data leakage that would artificially inflate model performance.")
        print("• RECOMMENDATION: Remove the 'subject' column before model training.")
    elif num_common_subjects > 0:
        # Get the most biased subjects (those that appear predominantly in one class)
        if comparison_df.count() > 0:
            # Calculate the absolute difference between fake and true percentages
            comparison_df = comparison_df.withColumn(
                "percentage_difference", 
                abs(col("fake_percentage") - col("true_percentage"))
            )
            
            # Find subjects with high bias (>80% difference)
            highly_biased = comparison_df.filter(col("percentage_difference") > 80).count()
            
            if highly_biased > 0:
                print("\n🔶 MODERATE RISK OF DATA LEAKAGE DETECTED!")
                print(f"• Found {highly_biased} subjects with >80% difference in distribution between classes.")
                print("• These subjects may create partial data leakage.")
                print("• RECOMMENDATION: Consider removing the 'subject' column or perform careful cross-validation.")
            else:
                print("\n✅ LOW RISK OF DATA LEAKAGE")
                print("• Subject distributions do not show strong bias toward either class.")
                print("• The 'subject' column may be used as a feature with caution.")
    
    print("\n" + "="*80)

# %% [markdown]
# #### Analyze dataset characteristics

# %%
def analyze_dataset_characteristics(df):
    """
    Very basic dataset analysis function with minimal Spark operations.
    Only performs essential checks to avoid overwhelming the cluster.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        dict: Dictionary with basic analysis results
    """
    from pyspark.sql.functions import col, count, when, length
    
    print("\n" + "="*50)
    print("📊 BASIC DATASET ANALYSIS")
    print("="*50)
    
    # Get column names
    columns = df.columns
    print(f"• Columns: {', '.join(columns)}")
    
    # Get total count (single Spark action)
    total_count = df.count()
    print(f"• Total records: {total_count}")
    
    # Check for required columns
    has_text = "text" in columns
    has_label = "label" in columns
    
    # Basic class distribution if label exists
    if has_label:
        print("\n• Class distribution:")
        # Use a single SQL query instead of multiple DataFrame operations
        df.createOrReplaceTempView("temp_data")
        class_dist = spark.sql("""
            SELECT label, COUNT(*) as count
            FROM temp_data
            GROUP BY label
            ORDER BY label
        """)
        display(class_dist)
    
    # Check for null values in important columns
    print("\n• Null value check:")
    null_counts = {}
    
    # Only check a few important columns to minimize operations
    columns_to_check = []
    if has_text:
        columns_to_check.append("text")
    if has_label:
        columns_to_check.append("label")
    if "location" in columns:
        columns_to_check.append("location")
    if "news_source" in columns:
        columns_to_check.append("news_source")
    
    for column_name in columns_to_check:
        null_count = df.filter(col(column_name).isNull()).count()
        null_counts[column_name] = null_count
        print(f"  - Null values in '{column_name}': {null_count}")
    
    # Check for duplicates in text column if it exists
    if has_text:
        print("\n• Duplicate check:")
        unique_count = df.select("text").distinct().count()
        duplicate_count = total_count - unique_count
        print(f"  - Duplicate texts: {duplicate_count}")
    
    print("\n" + "="*50)
    
    # Return minimal results
    return {
        "total_count": total_count,
        "columns": columns,
        "null_counts": null_counts
    }

# %% [markdown]
# #### Check random records with SQL

# %%
def check_random_records(df, num_records=10):
    """
    Uses SQL to select random records from a DataFrame for inspection.
    
    Args:
        df: DataFrame to sample
        num_records: Number of random records to return
        
    Returns:
        DataFrame: Sample of random records
    """
    print(f"Selecting {num_records} random records for inspection...")
    
    # Create a temporary view
    df.createOrReplaceTempView("temp_data")
    
    # Use SQL to select random records
    query = f"""
    SELECT *
    FROM temp_data
    ORDER BY rand()
    LIMIT {num_records}
    """
    
    # Execute the query
    result_df = spark.sql(query)
    
    # Display the results
    print("\nRandom sample of records:")
    display(result_df)
    
    # Show schema information
    print("\nSchema information:")
    result_df.printSchema()
    
    return result_df

# %% [markdown]
# #### Preprocess text

# %%
def preprocess_text(df, cache=True):
    """
    Preprocesses text by converting to lowercase, normalizing acronyms, and extracting metadata.
    Also checks for and removes problematic columns that may cause data leakage.
    
    Args:
        df: DataFrame with text column
        cache (bool): Whether to cache the preprocessed DataFrame
        
    Returns:
        DataFrame: DataFrame with preprocessed text
    """
    print("Preprocessing text...")
    
    # Define acronym replacements
    acronyms = {
        "U.S.": "US",
        "U.S.A.": "USA",
        "U.K.": "UK",
        "U.N.": "UN",
        "F.B.I.": "FBI",
        "C.I.A.": "CIA",
        "D.C.": "DC",
        "E.U.": "EU",
        "N.A.T.O.": "NATO",
        "W.H.O.": "WHO"
    }
    
    # Function to replace acronyms
    def replace_acronyms(text):
        if text is None:
            return None
        
        result = text
        for acronym, replacement in acronyms.items():
            # Use word boundaries to ensure we're replacing complete acronyms
            result = result.replace(acronym, replacement)
        
        return result
    
    # Register the UDF
    from pyspark.sql.functions import udf
    replace_acronyms_udf = udf(replace_acronyms, StringType())
    
    # Apply acronym normalization first
    df = df.withColumn("text", replace_acronyms_udf(col("text")))
    
    # Extract location information if present in text (e.g., "WASHINGTON —" or "NEW YORK (Reuters) -")
    df = df.withColumn(
        "location",
        when(
            regexp_replace(col("text"), "[^a-zA-Z0-9\\s]", " ").rlike("^[A-Z]{2,}\\s+—"),
            regexp_replace(regexp_replace(col("text"), "^([A-Z]{2,})\\s+—.*", "$1"), "\\s+", "")
        ).otherwise(None)
    )
    
    # Extract news source information if present (e.g., "(Reuters)" or "(AP)")
    df = df.withColumn(
        "news_source",
        when(
            col("text").rlike("\\([A-Za-z]+\\)"),
            regexp_replace(regexp_replace(col("text"), ".*\\(([A-Za-z]+)\\).*", "$1"), "\\s+", "")
        ).otherwise(None)
    )
    
    # Convert to lowercase
    df = df.withColumn("text", lower(col("text")))
    
    # Remove special characters but preserve hashtags and mentions
    df = df.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z0-9\\s#@]", " "))
    
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
        # Force materialization
        df.count()
    
    return df

# %% [markdown]
# #### Save to Hive tables safely

# %%
def save_to_hive_table_safely(df, table_name, partition_by=None, mode="overwrite"):
    """
    Safely saves a DataFrame to a Hive table, handling existing tables and locations.
    
    Args:
        df: DataFrame to save
        table_name: Name of the Hive table
        partition_by: Column(s) to partition by (optional)
        mode: Write mode (default: "overwrite")
    
    Returns:
        bool: True if successful
    """
    print(f"Safely saving DataFrame to Hive table: {table_name}")
    
    # Check if table exists
    tables = spark.sql("SHOW TABLES").select("tableName").rdd.flatMap(lambda x: x).collect()
    table_exists = table_name in tables
    
    if table_exists:
        print(f"Table '{table_name}' already exists. Dropping it...")
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        print(f"Table '{table_name}' dropped successfully.")
    
    # Check if the location exists and remove it if necessary
    # This is needed because dropping the table might not always remove the underlying data
    try:
        location_path = f"dbfs:/user/hive/warehouse/{table_name}"
        print(f"Checking if location exists: {location_path}")
        
        # Use dbutils to check if path exists and delete it if it does
        if dbutils.fs.ls(location_path):
            print(f"Location exists. Removing directory: {location_path}")
            dbutils.fs.rm(location_path, recurse=True)
            print(f"Directory removed successfully.")
    except Exception as e:
        # Path might not exist, which is fine
        print(f"Note: {str(e)}")
    
    # Save the DataFrame to the Hive table
    print(f"Saving DataFrame to table '{table_name}'...")
    
    if partition_by:
        df.write.format("parquet").partitionBy(partition_by).mode(mode).saveAsTable(table_name)
        print(f"DataFrame saved to table '{table_name}' with partitioning on '{partition_by}'.")
    else:
        df.write.format("parquet").mode(mode).saveAsTable(table_name)
        print(f"DataFrame saved to table '{table_name}'.")
    
    # Verify the table was created
    tables = spark.sql("SHOW TABLES").select("tableName").rdd.flatMap(lambda x: x).collect()
    if table_name in tables:
        print(f"Verified: Table '{table_name}' exists.")
        
        # Show table information
        print("\nTable information:")
        spark.sql(f"DESCRIBE TABLE {table_name}").show(truncate=False)
        
        # Show record count
        count = spark.sql(f"SELECT COUNT(*) as count FROM {table_name}").collect()[0]['count']
        print(f"\nRecord count: {count:,}")
        
        return True
    else:
        print(f"Error: Failed to create table '{table_name}'.")
        return False

# %% [markdown]
# ## Pipeline Execution

# %% [markdown]
# ### Step 1: Create Directory Structure

# %%
# Create the directory structure
directories = create_directory_structure()

# %% [markdown]
# ### Step 2: Load Data

# %%
# Define paths to the CSV files
fake_news_path = "/dbfs/FileStore/fake_news_detection/data/raw/Fake.csv"
true_news_path = "/dbfs/FileStore/fake_news_detection/data/raw/True.csv"

# Load the CSV files
fake_df, true_df = load_csv_files(fake_news_path, true_news_path, cache=True)

# %% [markdown]
# ### Step 3: Analyze Subject Distribution (Data Leakage Detection)

# %%
# Analyze the distribution of subjects to detect potential data leakage
analyze_subject_distribution(fake_df, true_df)

# %% [markdown]
# ### Step 4: Combine and Preprocess Data

# %%
# Combine fake and true news DataFrames
combined_df = fake_df.union(true_df)

# Analyze the combined dataset
analyze_dataset_characteristics(combined_df)

# Check a random sample of records
check_random_records(combined_df, num_records=5)

# Preprocess the text
combined_df = preprocess_text(combined_df, cache=True)

# %% [markdown]
# ### Step 5: Analyze Preprocessed Data

# %%
# Analyze the preprocessed dataset
analyze_dataset_characteristics(combined_df)

# Check a random sample of preprocessed records
check_random_records(combined_df, num_records=5)

# %% [markdown]
# ### Step 6: Save to Hive Tables

# %%
# Save the combined dataset to a Hive table
save_to_hive_table_safely(combined_df, "combined_news", partition_by="label")

# %% [markdown]
# ## Memory Management

# %%
# Unpersist DataFrames to free up memory
if fake_df._jdf.storageLevel().useMemory():
    fake_df.unpersist()
    print("Unpersisted fake_df from memory")

if true_df._jdf.storageLevel().useMemory():
    true_df.unpersist()
    print("Unpersisted true_df from memory")

if combined_df._jdf.storageLevel().useMemory():
    combined_df.unpersist()
    print("Unpersisted combined_df from memory")

print("Memory management completed")

# %% [markdown]
# ## Summary
# 
# In this notebook, we have:
# 
# 1. Created a directory structure for the fake news detection project
# 2. Loaded fake and true news datasets from CSV files
# 3. Analyzed the subject distribution to detect potential data leakage
# 4. Combined and preprocessed the data, including:
#    - Acronym normalization
#    - Location and news source extraction
#    - Text cleaning and normalization
#    - Removal of the 'subject' column to prevent data leakage
# 5. Saved the preprocessed data to a Hive table for use in subsequent phases
# 6. Implemented proper memory management by unpersisting DataFrames when no longer needed
# 
# The data is now ready for the next phase: feature engineering.
