# %% [markdown]
# # Fake News Detection: Feature Engineering
# 
# This notebook contains all the necessary code for feature engineering in the fake news detection project. The code is organized into independent functions, without dependencies on external modules or classes, to facilitate execution in Databricks Community Edition.

# %% [markdown]
# ## Setup and Imports

# %%
# Import necessary libraries
import os
import time
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, lower, regexp_extract, when, count, desc, lit, array, 
    size, split, explode, collect_list, struct, expr
)
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, IntegerType, FloatType
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, StopWordsRemover, LDA
from pyspark.ml import Pipeline

# %%
# Initialize Spark session optimized for Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsFeatureEngineering") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

# Display Spark configuration
print(f"Spark version: {spark.version}")
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")

# %%
# Start timer for performance tracking
start_time = time.time()

# %% [markdown]
# ## Reusable Functions

# %% [markdown]
# ### Data Loading Functions

# %%
def load_preprocessed_data(path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet"):
    """
    Load preprocessed data from Parquet file.
    
    Args:
        path (str): Path to the preprocessed data Parquet file
        
    Returns:
        DataFrame: Spark DataFrame with preprocessed data
    """
    print(f"Loading preprocessed data from {path}...")
    
    try:
        # Load data from Parquet file
        df = spark.read.parquet(path)
        
        # Display basic information
        print(f"Successfully loaded {df.count()} records.")
        df.printSchema()
        
        # Cache the DataFrame for better performance
        df.cache()
        print("Preprocessed DataFrame cached.")
        
        return df
    
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        print("Please ensure the preprocessing notebook ran successfully and saved data to the correct path.")
        return None

# %% [markdown]
# ### Source Extraction Functions

# %%
def extract_source_from_text(text):
    """
    Extract news source from text using regex patterns.
    
    Args:
        text (str): The news article text
        
    Returns:
        str: Extracted source name or None if not found
    """
    if text is None:
        return None
    
    # Define list of common news sources
    common_sources = [
        "Reuters", "AP", "Associated Press", "CNN", "Fox News", "MSNBC", "BBC", 
        "New York Times", "Washington Post", "USA Today", "NPR", "CBS", "NBC", 
        "ABC News", "The Guardian", "Bloomberg", "Wall Street Journal", "WSJ",
        "Huffington Post", "Breitbart", "BuzzFeed", "Daily Mail", "The Hill"
    ]
    
    # Pattern: Optional Location (SOURCE) - Text
    match = re.match(r"^\s*\w*\s*\(([^)]+)\)\s*-", text)
    if match:
        potential_source = match.group(1).strip()
        # Check against common sources
        for src in common_sources:
            if src.lower() == potential_source.lower():
                return src
    
    # Fallback: Check if text starts with a known source name
    for src in common_sources:
        if text and text.lower().startswith(src.lower()):
            return src
    
    # Try to find source at the end of the text with pattern "- Source"
    if text:
        source_match = re.search(r'-\s*([^-\n]+?)$', text)
        if source_match:
            potential_source = source_match.group(1).strip()
            # Check if the extracted text contains a known source
            for known_source in common_sources:
                if known_source.lower() in potential_source.lower():
                    return known_source
    
    # Try to find source in the text
    if text:
        for source in common_sources:
            if source.lower() in text.lower():
                return source
    
    return None

# %%
def extract_source_feature(df, text_column="text"):
    """
    Extract source feature from text column in a DataFrame.
    
    Args:
        df (DataFrame): Input DataFrame with text column
        text_column (str): Name of the column containing text
        
    Returns:
        DataFrame: DataFrame with extracted source feature
    """
    print("Extracting news source feature...")
    
    # Register UDF for source extraction
    extract_source_udf = udf(extract_source_from_text, StringType())
    
    # Apply UDF to extract source
    result_df = df.withColumn("extracted_source", extract_source_udf(col(text_column)))
    
    # Show some results
    result_df.select(text_column, "extracted_source").show(10, truncate=80)
    
    # Analyze extracted sources
    print("\nDistribution of extracted sources:")
    source_counts = result_df.groupBy("extracted_source").count().orderBy(desc("count"))
    source_counts.show()
    
    # Analyze source distribution by label
    if "label" in df.columns:
        print("\nExtracted source distribution by label:")
        source_by_label = result_df.groupBy("extracted_source", "label").count()
        source_by_label_pivot = source_by_label.groupBy("extracted_source")\
            .pivot("label", [0, 1])\
            .agg(count("count").alias("count"))\
            .na.fill(0)\
            .withColumnRenamed("0", "fake_count")\
            .withColumnRenamed("1", "real_count")\
            .withColumn("total_count", col("fake_count") + col("real_count"))\
            .orderBy(desc("total_count"))
            
        source_by_label_pivot.show()
    
    return result_df, source_by_label_pivot if "label" in df.columns else None

# %% [markdown]
# ### Topic Modeling Functions

# %%
def create_topic_modeling_pipeline(num_topics=10, max_iterations=10, vocab_size=10000, min_doc_freq=5.0):
    """
    Create a topic modeling pipeline using Spark MLlib.
    
    Args:
        num_topics (int): Number of topics for LDA
        max_iterations (int): Maximum iterations for LDA
        vocab_size (int): Vocabulary size for CountVectorizer
        min_doc_freq (float): Minimum document frequency for CountVectorizer
        
    Returns:
        Pipeline: Spark ML Pipeline for topic modeling
    """
    print("Setting up topic modeling pipeline...")
    
    # 1. Tokenizer: Split processed text into words
    tokenizer = Tokenizer(inputCol="processed_text", outputCol="raw_tokens")
    
    # 2. StopWordsRemover: Remove common English stop words
    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="tokens")
    
    # 3. CountVectorizer: Convert tokens into frequency vectors
    cv = CountVectorizer(
        inputCol="tokens", 
        outputCol="rawFeatures", 
        vocabSize=vocab_size, 
        minDF=min_doc_freq
    )
    
    # 4. IDF: Down-weight common terms across documents
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    
    # 5. LDA: Discover latent topics
    lda = LDA(
        k=num_topics, 
        maxIter=max_iterations, 
        featuresCol="features", 
        topicDistributionCol="topicDistribution"
    )
    
    # Create the pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lda])
    
    print(f"Topic modeling pipeline created with {num_topics} topics.")
    return pipeline

# %%
def fit_topic_model(df, pipeline, text_column="processed_text"):
    """
    Fit topic modeling pipeline to data.
    
    Args:
        df (DataFrame): Input DataFrame with processed text
        pipeline (Pipeline): Spark ML Pipeline for topic modeling
        text_column (str): Name of the column containing processed text
        
    Returns:
        tuple: (pipeline_model, lda_results_df) - Fitted pipeline model and transformed DataFrame
    """
    print("Fitting topic modeling pipeline... This may take some time on the full dataset.")
    start_lda_time = time.time()
    
    # Ensure the text column exists
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in DataFrame.")
        return None, None
    
    # Fit the pipeline
    pipeline_model = pipeline.fit(df)
    
    # Transform the data to get topic distributions
    lda_results_df = pipeline_model.transform(df)
    
    print(f"Pipeline fitting and transformation completed in {time.time() - start_lda_time:.2f} seconds.")
    
    # Display schema with new columns
    lda_results_df.printSchema()
    
    # Show sample results with topic distribution
    if "id" in lda_results_df.columns:
        lda_results_df.select("id", "label", "topicDistribution").show(5, truncate=False)
    else:
        lda_results_df.select("label", "topicDistribution").show(5, truncate=False)
    
    return pipeline_model, lda_results_df

# %%
def analyze_topics(pipeline_model, max_terms_per_topic=10):
    """
    Analyze topics discovered by the LDA model.
    
    Args:
        pipeline_model (PipelineModel): Fitted pipeline model containing LDA
        max_terms_per_topic (int): Maximum number of terms to show per topic
        
    Returns:
        DataFrame: DataFrame with topic descriptions
    """
    print("Analyzing discovered topics...")
    
    # Extract the LDA model and vocabulary from the pipeline
    lda_model = pipeline_model.stages[-1]  # LDA is the last stage
    cv_model = pipeline_model.stages[2]    # CountVectorizer is the third stage
    vocabulary = cv_model.vocabulary
    
    # Get the topic descriptions (top words per topic)
    topics = lda_model.describeTopics(maxTermsPerTopic=max_terms_per_topic)
    
    print("Top terms per topic:")
    topics_with_terms = []
    for row in topics.collect():
        topic_idx = row[0]
        term_indices = row[1]
        term_weights = row[2]
        topic_terms = [vocabulary[i] for i in term_indices]
        print(f"Topic {topic_idx}: {topic_terms}")
        
        # Create a row for the topics DataFrame
        topics_with_terms.append((topic_idx, topic_terms))
    
    # Convert topics summary to DataFrame for easier analysis/saving
    topics_schema = StructType([
        StructField("topic_id", StringType(), False),
        StructField("top_terms", ArrayType(StringType()), False)
    ])
    
    # Create DataFrame using createDataFrame with explicit schema
    topics_df = spark.createDataFrame(topics_with_terms, ["topic_id", "top_terms"])
    topics_df.show(truncate=False)
    
    return topics_df

# %%
def analyze_topic_distribution_by_label(lda_results_df):
    """
    Analyze how topics are distributed across fake and real news.
    
    Args:
        lda_results_df (DataFrame): DataFrame with topic distributions and labels
        
    Returns:
        DataFrame: DataFrame with topic distribution by label
    """
    print("Analyzing topic distribution by label...")
    
    # Ensure required columns exist
    if "topicDistribution" not in lda_results_df.columns or "label" not in lda_results_df.columns:
        print("Error: Required columns 'topicDistribution' or 'label' not found.")
        return None
    
    # Extract dominant topic for each document
    dominant_topic_udf = udf(lambda v: float(v.argmax()), FloatType())
    with_dominant_topic = lda_results_df.withColumn("dominant_topic", dominant_topic_udf(col("topicDistribution")))
    
    # Analyze topic distribution by label
    topic_by_label = with_dominant_topic.groupBy("dominant_topic", "label").count()
    
    # Create pivot table
    topic_by_label_pivot = topic_by_label.groupBy("dominant_topic")\
        .pivot("label", [0, 1])\
        .agg(count("count").alias("count"))\
        .na.fill(0)\
        .withColumnRenamed("0", "fake_count")\
        .withColumnRenamed("1", "real_count")\
        .withColumn("total_count", col("fake_count") + col("real_count"))\
        .withColumn("fake_ratio", col("fake_count") / col("total_count"))\
        .withColumn("real_ratio", col("real_count") / col("total_count"))\
        .orderBy("dominant_topic")
    
    # Show results
    print("Topic distribution by label:")
    topic_by_label_pivot.show()
    
    return topic_by_label_pivot

# %% [markdown]
# ### Text Feature Extraction Functions

# %%
def extract_text_features(df, text_column="text"):
    """
    Extract various text features from the text column.
    
    Args:
        df (DataFrame): Input DataFrame with text column
        text_column (str): Name of the column containing text
        
    Returns:
        DataFrame: DataFrame with extracted text features
    """
    print("Extracting text features...")
    
    # Ensure text column exists
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in DataFrame.")
        return df
    
    # Text length (character count)
    result_df = df.withColumn("text_length", length(col(text_column)))
    
    # Word count
    result_df = result_df.withColumn("word_count", size(split(col(text_column), " ")))
    
    # Average word length
    result_df = result_df.withColumn(
        "avg_word_length", 
        when(col("word_count") > 0, col("text_length") / col("word_count")).otherwise(0)
    )
    
    # Count of special characters
    special_chars_udf = udf(lambda text: len(re.findall(r'[^\w\s]', text)) if text else 0, IntegerType())
    result_df = result_df.withColumn("special_char_count", special_chars_udf(col(text_column)))
    
    # Count of uppercase words
    uppercase_words_udf = udf(lambda text: len(re.findall(r'\b[A-Z]{2,}\b', text)) if text else 0, IntegerType())
    result_df = result_df.withColumn("uppercase_word_count", uppercase_words_udf(col(text_column)))
    
    # Show sample of extracted features
    result_df.select(
        text_column, "text_length", "word_count", 
        "avg_word_length", "special_char_count", "uppercase_word_count"
    ).show(5, truncate=80)
    
    return result_df

# %%
def calculate_text_statistics(df):
    """
    Calculate statistics for text features.
    
    Args:
        df (DataFrame): Input DataFrame with text features
        
    Returns:
        DataFrame: DataFrame with text feature statistics by label
    """
    print("Calculating text feature statistics...")
    
    # Required text feature columns
    text_features = ["text_length", "word_count", "avg_word_length", "special_char_count", "uppercase_word_count"]
    
    # Check if all required columns exist
    missing_cols = [col for col in text_features if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Please run extract_text_features first.")
        return None
    
    # Calculate statistics for each feature
    stats = []
    for feature in text_features:
        # Overall statistics
        overall_stats = df.select(feature).summary("min", "25%", "mean", "75%", "max").collect()
        overall_dict = {row["summary"]: float(row[feature]) for row in overall_stats}
        
        # Statistics by label if label column exists
        if "label" in df.columns:
            # Real news statistics (label=1)
            real_stats = df.filter(col("label") == 1).select(feature).summary("min", "25%", "mean", "75%", "max").collect()
            real_dict = {row["summary"]: float(row[feature]) for row in real_stats}
            
            # Fake news statistics (label=0)
            fake_stats = df.filter(col("label") == 0).select(feature).summary("min", "25%", "mean", "75%", "max").collect()
            fake_dict = {row["summary"]: float(row[feature]) for row in fake_stats}
            
            stats.append((feature, overall_dict, real_dict, fake_dict))
        else:
            stats.append((feature, overall_dict, None, None))
    
    # Print statistics
    for feature, overall, real, fake in stats:
        print(f"\nStatistics for {feature}:")
        print(f"Overall: {overall}")
        if real and fake:
            print(f"Real news: {real}")
            print(f"Fake news: {fake}")
    
    return stats

# %% [markdown]
# ### Data Storage Functions

# %%
def save_to_parquet(df, path, partition_by=None):
    """
    Save a DataFrame in Parquet format.
    
    Args:
        df (DataFrame): DataFrame to save
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
    Save a DataFrame to a Hive table.
    
    Args:
        df (DataFrame): DataFrame to save
        table_name (str): Name of the Hive table to create or replace
        partition_by (str): Column to partition by (optional)
    """
    print(f"Saving DataFrame to Hive table {table_name}...")
    
    writer = df.write.mode("overwrite").format("parquet")
    
    if partition_by:
        writer = writer.partitionBy(partition_by)
    
    writer.saveAsTable(table_name)
    print(f"DataFrame saved to Hive table: {table_name}")

# %%
def save_pipeline_model(pipeline_model, path):
    """
    Save a pipeline model to disk.
    
    Args:
        pipeline_model (PipelineModel): Fitted pipeline model to save
        path (str): Path where to save the model
    """
    print(f"Saving pipeline model to {path}...")
    
    try:
        pipeline_model.write().overwrite().save(path)
        print("Pipeline model saved successfully.")
    except Exception as e:
        print(f"Error saving pipeline model: {e}")

# %% [markdown]
# ## Complete Feature Engineering Pipeline

# %%
def engineer_features_and_save(
    input_path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet",
    output_dir="dbfs:/FileStore/fake_news_detection/feature_data",
    model_save_dir="dbfs:/FileStore/fake_news_detection/models/feature_engineering",
    num_topics=10,
    create_tables=True
):
    """
    Complete feature engineering pipeline for fake news detection.
    
    This pipeline loads preprocessed data, extracts source and text features,
    performs topic modeling, and saves the results.
    
    Args:
        input_path (str): Path to preprocessed data
        output_dir (str): Directory to save feature data
        model_save_dir (str): Directory to save models
        num_topics (int): Number of topics for LDA
        create_tables (bool): Whether to create Hive tables
        
    Returns:
        dict: Dictionary with references to processed DataFrames
    """
    print("Starting feature engineering pipeline...")
    start_time = time.time()
    
    # Create output directories
    try:
        dbutils.fs.mkdirs(output_dir.replace("dbfs:", ""))
        dbutils.fs.mkdirs(model_save_dir.replace("dbfs:", ""))
    except:
        print("Warning: Could not create directories. This is expected in local environments.")
    
    # 1. Load preprocessed data
    df = load_preprocessed_data(input_path)
    if df is None:
        print("Error: Could not load preprocessed data. Pipeline aborted.")
        return None
    
    # 2. Extract source feature
    df_with_source, source_by_label = extract_source_feature(df, "text")
    
    # 3. Extract text features
    df_with_text_features = extract_text_features(df_with_source, "text")
    
    # 4. Calculate text statistics
    text_stats = calculate_text_statistics(df_with_text_features)
    
    # 5. Create topic modeling pipeline
    topic_pipeline = create_topic_modeling_pipeline(num_topics=num_topics)
    
    # 6. Fit topic model
    pipeline_model, df_with_topics = fit_topic_model(df_with_text_features, topic_pipeline)
    
    # 7. Analyze topics
    if pipeline_model:
        topics_df = analyze_topics(pipeline_model)
        
        # 8. Analyze topic distribution by label
        topic_by_label = analyze_topic_distribution_by_label(df_with_topics)
        
        # 9. Save pipeline model
        model_path = f"{model_save_dir}/lda_pipeline_model"
        save_pipeline_model(pipeline_model, model_path)
    else:
        print("Warning: Topic modeling failed. Continuing with other features.")
        df_with_topics = df_with_text_features
        topics_df = None
        topic_by_label = None
    
    # 10. Save feature data
    features_path = f"{output_dir}/features.parquet"
    save_to_parquet(df_with_topics, features_path, partition_by="label")
    
    # 11. Save to Hive table for easier access
    if create_tables:
        save_to_hive_table(df_with_topics, "news_features", partition_by="label")
        
        # Save source distribution
        if source_by_label is not None:
            save_to_hive_table(source_by_label, "source_distribution")
        
        # Save topic distribution
        if topic_by_label is not None:
            save_to_hive_table(topic_by_label, "topic_distribution")
    
    print(f"\nFeature engineering pipeline completed in {time.time() - start_time:.2f} seconds!")
    
    return {
        "preprocessed_df": df,
        "df_with_source": df_with_source,
        "df_with_text_features": df_with_text_features,
        "df_with_topics": df_with_topics,
        "source_by_label": source_by_label,
        "topic_by_label": topic_by_label,
        "topics_df": topics_df,
        "pipeline_model": pipeline_model
    }

# %% [markdown]
# ## Step-by-Step Tutorial

# %% [markdown]
# ### 1. Load Preprocessed Data

# %%
# Load preprocessed data
preprocessed_df = load_preprocessed_data()

# Display sample data
if preprocessed_df:
    print("Preprocessed data sample:")
    preprocessed_df.show(5, truncate=80)

# %% [markdown]
# ### 2. Extract Source Feature

# %%
# Extract source feature
if preprocessed_df:
    df_with_source, source_by_label = extract_source_feature(preprocessed_df)
    
    # Visualize source distribution (using Databricks display function)
    if source_by_label:
        print("Source distribution by label:")
        display(source_by_label.limit(15))

# %% [markdown]
# ### 3. Extract Text Features

# %%
# Extract text features
if 'df_with_source' in locals():
    df_with_text_features = extract_text_features(df_with_source)
    
    # Calculate text statistics
    text_stats = calculate_text_statistics(df_with_text_features)

# %% [markdown]
# ### 4. Topic Modeling

# %%
# Create topic modeling pipeline
if 'df_with_text_features' in locals():
    # Create pipeline with 10 topics
    topic_pipeline = create_topic_modeling_pipeline(num_topics=10)
    
    # Fit topic model
    pipeline_model, df_with_topics = fit_topic_model(df_with_text_features, topic_pipeline)

# %% [markdown]
# ### 5. Analyze Topics

# %%
# Analyze topics
if 'pipeline_model' in locals() and pipeline_model:
    # Get topic descriptions
    topics_df = analyze_topics(pipeline_model)
    
    # Analyze topic distribution by label
    topic_by_label = analyze_topic_distribution_by_label(df_with_topics)
    
    # Visualize topic distribution (using Databricks display function)
    print("Topic distribution by label:")
    display(topic_by_label)

# %% [markdown]
# ### 6. Complete Feature Engineering Pipeline

# %%
# Run the complete feature engineering pipeline
results = engineer_features_and_save(
    input_path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet",
    output_dir="dbfs:/FileStore/fake_news_detection/feature_data",
    model_save_dir="dbfs:/FileStore/fake_news_detection/models/feature_engineering",
    num_topics=10,
    create_tables=True
)

# %% [markdown]
# ## Important Notes
# 
# 1. **Feature Engineering Importance**: Feature engineering is crucial for fake news detection as it helps extract meaningful signals from text and creates numerical representations suitable for machine learning models.
# 
# 2. **Source Extraction**: We extract the news source from the text, which can be a valuable feature as some sources may be more reliable than others.
# 
# 3. **Text Features**: We extract various text features like text length, word count, average word length, and counts of special characters and uppercase words, which can help identify patterns in fake vs. real news.
# 
# 4. **Topic Modeling**: We use Latent Dirichlet Allocation (LDA) to discover latent topics in the news articles, which can reveal thematic differences between fake and real news.
# 
# 5. **Spark Optimization**: The code is optimized for Spark's distributed processing capabilities, making it suitable for large datasets.
# 
# 6. **Databricks Integration**: The pipeline is designed to work seamlessly in Databricks, with appropriate configurations for the Community Edition.
# 
# 7. **Pipeline Model Saving**: We save the fitted pipeline model for later use in prediction or further analysis.
# 
# 8. **Feature Storage**: All extracted features are saved in Parquet format and as Hive tables for easy access in subsequent steps of the pipeline.
