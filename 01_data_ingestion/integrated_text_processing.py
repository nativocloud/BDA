# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Integrated Text Processing for Fake News Detection
#
# This notebook demonstrates the integrated approach to text processing, combining data ingestion and preprocessing in a single phase. This optimized pipeline improves efficiency and reduces redundancy in the fake news detection workflow.

# %% [markdown]
# ## Setup and Imports

# %%
# Import required libraries
import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, lower, regexp_replace, regexp_extract, trim, when, rand, concat
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Import NLTK for text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# %% [markdown]
# ## Create Spark Session

# %%
# Create a Spark session with configuration optimized for Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsDetection_IntegratedProcessing") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

# Display Spark configuration
print(f"Spark version: {spark.version}")
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")

# %% [markdown]
# ## Create Directory Structure

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

# Create directories
directories = create_directory_structure()

# %% [markdown]
# ## Data Loading Functions

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

def combine_datasets(fake_df, true_df, cache=True):
    """
    Combines fake and true news datasets into a single DataFrame.
    
    Args:
        fake_df: DataFrame with fake news
        true_df: DataFrame with true news
        cache (bool): Whether to cache the combined DataFrame
        
    Returns:
        DataFrame: Combined DataFrame with both fake and true news
    """
    print("Combining fake and true news datasets...")
    
    # Combine datasets
    combined_df = fake_df.union(true_df)
    
    # Cache the combined DataFrame if requested
    if cache:
        combined_df.cache()
        # Force materialization
        combined_count = combined_df.count()
    
    print(f"Combined dataset created with {combined_df.count()} records")
    
    return combined_df

# %% [markdown]
# ## Integrated Text Processing Functions
#
# These functions combine preprocessing, tokenization, and stopword removal in a single pipeline.

# %%
from pyspark.sql.functions import col, lower, regexp_replace, regexp_extract, trim, when, lit, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover

def preprocess_text(df, cache=True):
    """
    Optimized text preprocessing function for Spark performance with fixed acronym handling.
    Preprocesses text by extracting optional location(s) and news source,
    normalizing acronyms, converting to lowercase, removing special characters,
    and handling multiple spaces.
    
    This version fixes the issue with acronyms like "U.S." being incorrectly converted to "u s".
    
    Args:
        df: DataFrame with text and potentially title columns.
        cache (bool): Whether to cache the preprocessed DataFrame.

    Returns:
        DataFrame: DataFrame with preprocessed text (and title if applicable),
                   plus new 'location' and 'news_source' columns.
    """
    print("Starting text preprocessing...")
    
    # Create a list to track columns that need preprocessing
    columns_to_preprocess = []
    
    # Check for text and title columns upfront to minimize schema lookups
    has_text = "text" in df.columns
    has_title = "title" in df.columns
    
    # Get column types once to avoid repeated schema lookups
    if has_text:
        text_is_string = isinstance(df.schema["text"].dataType, StringType)
        if text_is_string:
            columns_to_preprocess.append("text")
    
    if has_title:
        title_is_string = isinstance(df.schema["title"].dataType, StringType)
        if title_is_string:
            columns_to_preprocess.append("title")
    
    # --- 1. Extract Optional Location(s) and News Source from 'text' column ---
    if has_text and text_is_string:
        print("• Extracting 'location' and 'news_source' from 'text' column...")
        
        # Optimize regex pattern with non-capturing groups where possible
        news_header_pattern = r"^(?:([A-Z][a-zA-Z\s\./,]*)\s*)?\(([^)]+)\)\s*-\s*(.*)"
        
        # Apply all extractions in a single transformation to minimize passes over the data
        df = df.withColumn("location", regexp_extract(col("text"), news_header_pattern, 1)) \
               .withColumn("news_source", regexp_extract(col("text"), news_header_pattern, 2)) \
               .withColumn("text_cleaned", regexp_extract(col("text"), news_header_pattern, 3))
        
        # Update text column and handle empty extractions in a single transformation
        df = df.withColumn("text", 
                          when(col("text_cleaned") != "", col("text_cleaned"))
                          .otherwise(col("text"))) \
               .withColumn("location", 
                          when(col("location") == "", lit(None))
                          .otherwise(trim(col("location")))) \
               .withColumn("news_source", 
                          when(col("news_source") == "", lit(None))
                          .otherwise(trim(col("news_source")))) \
               .drop("text_cleaned")
        
        print("• 'location' and 'news_source' columns added (if pattern found).")
    else:
        if has_text:
            print(f"• Skipping location/source extraction: 'text' column is not a string type.")
        else:
            print("• 'text' column not found, skipping location/source extraction.")
    
    # --- 2. Apply acronym normalization BEFORE any other text processing ---
    if columns_to_preprocess:
        print(f"• Applying acronym normalization to {len(columns_to_preprocess)} column(s): {', '.join(columns_to_preprocess)}")
        
        # Define a function to normalize acronyms
        def normalize_acronyms(text):
            if text is None:
                return None
                
            # Replace common acronyms with their normalized forms
            # The order is important - longer patterns first
            replacements = [
                ("U.S.A.", "USA"),
                ("U.S.", "US"),
                ("U.N.", "UN"),
                ("F.B.I.", "FBI"),
                ("C.I.A.", "CIA"),
                ("D.C.", "DC"),
                ("U.K.", "UK"),
                ("E.U.", "EU"),
                ("N.Y.", "NY"),
                ("L.A.", "LA"),
                ("N.A.T.O.", "NATO"),
                ("W.H.O.", "WHO")
            ]
            
            for pattern, replacement in replacements:
                # Use Python's replace method which is more reliable for exact string replacement
                text = text.replace(pattern, replacement)
                
            return text
        
        # Register the UDF
        normalize_acronyms_udf = udf(normalize_acronyms, StringType())
        
        # Apply the UDF to each column that needs preprocessing
        for col_name in columns_to_preprocess:
            # First normalize acronyms using the UDF
            df = df.withColumn(col_name, normalize_acronyms_udf(col(col_name)))
            print(f"  - Applied acronym normalization to '{col_name}'.")
        
        # --- 3. Now apply the rest of the text preprocessing ---
        print(f"• Applying general text preprocessing to {len(columns_to_preprocess)} column(s)")
        
        for col_name in columns_to_preprocess:
            # Apply remaining transformations in a single chain
            df = df.withColumn(
                col_name,
                # Step 3: Trim and normalize spaces
                trim(
                    regexp_replace(
                        # Step 2: Remove special characters (keep #@)
                        regexp_replace(
                            # Step 1: Convert to lowercase
                            lower(col(col_name)),
                            "[^a-z0-9\\s#@]", " "  # Remove special chars
                        ),
                        "\\s+", " "  # Normalize spaces
                    )
                )
            )
            
            print(f"  - Applied full preprocessing chain to '{col_name}'.")
    else:
        print("• No suitable text columns found for preprocessing.")
    
    # --- 4. Data Leakage Check and Removal ('subject' column) ---
    has_subject = "subject" in df.columns
    if has_subject:
        print("\nWARNING: Removing 'subject' column to prevent data leakage.")
        df = df.drop("subject")
        print("'subject' column successfully removed.")
    else:
        print("\n'subject' column not found, no data leakage prevention needed for this column.")
    
    # --- 5. Cache the preprocessed DataFrame if requested ---
    if cache:
        print("• Caching the preprocessed DataFrame for optimized downstream operations.")
        df.cache()
        # Force materialization of the cache to ensure transformations are computed
        df.count()
    else:
        print("• Caching of the preprocessed DataFrame is disabled.")
    
    print("Text preprocessing complete.")
    return df

# %%
def tokenize_text(df, text_column="text", output_column="tokens"):
    """
    Tokenize text into words.
    
    Args:
        df: DataFrame with text column
        text_column (str): Name of the text column
        output_column (str): Name of the output column for tokens
        
    Returns:
        DataFrame: DataFrame with tokenized text
    """
    print("Tokenizing text...")
    
    # Create a tokenizer
    tokenizer = Tokenizer(inputCol=text_column, outputCol=output_column)
    
    # Apply tokenization
    tokenized_df = tokenizer.transform(df)
    
    return tokenized_df

# %%
def remove_stopwords(df, tokens_column="tokens", output_column="filtered_tokens"):
    """
    Remove stopwords from tokenized text.
    
    Args:
        df: DataFrame with tokens column
        tokens_column (str): Name of the tokens column
        output_column (str): Name of the output column for filtered tokens
        
    Returns:
        DataFrame: DataFrame with stopwords removed
    """
    print("Removing stopwords...")
    
    # Create a stopwords remover
    remover = StopWordsRemover(inputCol=tokens_column, outputCol=output_column)
    
    # Apply stopwords removal
    filtered_df = remover.transform(df)
    
    return filtered_df

# %%
def complete_text_processing(df, cache=True):
    """
    Performs complete text processing in a single pass:
    1. Text preprocessing (acronym normalization, lowercase, etc.)
    2. Tokenization
    3. Stopword removal
    
    Args:
        df: DataFrame with text column
        cache (bool): Whether to cache intermediate DataFrames
        
    Returns:
        DataFrame: Fully processed DataFrame with tokens and filtered tokens
    """
    print("Starting complete text processing pipeline...")
    
    # Step 1: Preprocess text
    preprocessed_df = preprocess_text(df, cache=cache)
    
    # Step 2: Tokenize text
    tokenized_df = tokenize_text(preprocessed_df, text_column="text", output_column="tokens")
    
    # Step 3: Remove stopwords
    processed_df = remove_stopwords(tokenized_df, tokens_column="tokens", output_column="filtered_tokens")
    
    # Unpersist intermediate DataFrame if it was cached
    if cache and preprocessed_df != df:  # Only if it's a different DataFrame
        try:
            preprocessed_df.unpersist()
            print("Unpersisted intermediate preprocessed DataFrame to free memory.")
        except:
            print("Note: Could not unpersist intermediate DataFrame.")
    
    print("Complete text processing pipeline finished.")
    return processed_df

# %% [markdown]
# ## Data Validation Functions

# %%
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
    
    # Use display() in Databricks, otherwise print
    try:
        display(fake_subjects_df)
    except NameError:
        print(fake_subjects_df.toPandas())

    print("\n2️⃣ TRUE NEWS SUBJECT DISTRIBUTION")
    true_subjects_df = true_df.groupBy("subject").count().orderBy(col("count").desc())
    print("• Subject distribution in true news:")
    
    # Use display() in Databricks, otherwise print
    try:
        display(true_subjects_df)
    except NameError:
        print(true_subjects_df.toPandas())

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
        
        # Use display() in Databricks, otherwise print
        try:
            display(comparison_df)
        except NameError:
            print(comparison_df.toPandas())
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
# ## Complete Integrated Pipeline

# %%
def run_integrated_pipeline(fake_path, true_path, output_path=None, cache=True):
    """
    Runs the complete integrated pipeline for fake news detection.
    This combines data ingestion and preprocessing in a single phase.
    
    Args:
        fake_path (str): Path to the CSV file with fake news
        true_path (str): Path to the CSV file with true news
        output_path (str): Path to save the processed data (optional)
        cache (bool): Whether to cache DataFrames during processing
        
    Returns:
        DataFrame: Fully processed DataFrame ready for feature engineering
    """
    print("Starting integrated pipeline for fake news detection...")
    
    # Step 1: Load data
    fake_df, true_df = load_csv_files(fake_path, true_path, cache=cache)
    
    # Step 2: Analyze subject distribution to detect data leakage
    analyze_subject_distribution(fake_df, true_df)
    
    # Step 3: Combine datasets
    combined_df = combine_datasets(fake_df, true_df, cache=cache)
    
    # Step 4: Complete text processing (preprocessing + tokenization + stopword removal)
    processed_df = complete_text_processing(combined_df, cache=cache)
    
    # Step 5: Save processed data if output path is provided
    if output_path:
        print(f"Saving processed data to {output_path}...")
        processed_df.write.mode("overwrite").parquet(output_path)
        print("Data saved successfully.")
    
    # Step 6: Unpersist DataFrames that are no longer needed
    if cache:
        print("Cleaning up memory...")
        try:
            fake_df.unpersist()
            true_df.unpersist()
            combined_df.unpersist()
            print("Memory cleanup complete.")
        except:
            print("Note: Could not unpersist some DataFrames.")
    
    print("Integrated pipeline completed successfully.")
    return processed_df

# %% [markdown]
# ## Example Usage
#
# Here's how to use the integrated pipeline:

# %%
# Define paths
fake_path = "/path/to/Fake.csv"  # Update with your actual path
true_path = "/path/to/True.csv"  # Update with your actual path
output_path = "/path/to/processed_data"  # Update with your desired output path

# Run the integrated pipeline
# Uncomment the following lines to execute
# processed_df = run_integrated_pipeline(
#     fake_path=fake_path,
#     true_path=true_path,
#     output_path=output_path,
#     cache=True
# )

# %% [markdown]
# ## Examine the Results

# %%
# Display schema
# processed_df.printSchema()

# Show sample data
# display(processed_df.select("text", "tokens", "filtered_tokens", "label", "location", "news_source").limit(5))

# Count records by label
# display(processed_df.groupBy("label").count().orderBy("label"))

# %% [markdown]
# ## Pipeline API Approach
#
# An alternative approach is to use the Spark ML Pipeline API:

# %%
from pyspark.ml import Pipeline

def create_pipeline_api_approach(include_features=True):
    """
    Creates a text processing pipeline using Spark ML Pipeline API.
    
    Args:
        include_features (bool): Whether to include feature extraction steps
        
    Returns:
        Pipeline: Spark ML Pipeline for text processing
    """
    # Define transformers
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    
    # Create pipeline stages
    stages = [tokenizer, remover]
    
    # Optionally add feature extraction
    if include_features:
        hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        stages.extend([hashingTF, idf])
    
    # Create and return the pipeline
    return Pipeline(stages=stages)

# Example usage (commented out)
# pipeline = create_pipeline_api_approach(include_features=True)
# model = pipeline.fit(preprocessed_df)
# processed_df = model.transform(preprocessed_df)

# %% [markdown]
# ## Conclusion
#
# This notebook demonstrates the integrated approach to text processing for fake news detection, combining data ingestion and preprocessing in a single phase. This optimized pipeline improves efficiency and reduces redundancy in the workflow.
#
# Key benefits of this approach:
# 1. Reduced computation by eliminating redundant processing
# 2. Improved memory efficiency through strategic caching and unpersisting
# 3. Simplified workflow with fewer steps
# 4. Enhanced performance in resource-constrained environments like Databricks Community Edition
#
# The processed data is now ready for feature engineering and model training.
