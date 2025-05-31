# Tutorial: Integrated Text Processing Pipeline for Fake News Detection

*Last updated: May 31, 2025*

## Introduction

This tutorial explains how to implement an integrated text processing pipeline for fake news detection that combines data ingestion and preprocessing in a single phase. This approach improves efficiency, reduces redundancy, and optimizes performance in resource-constrained environments like Databricks Community Edition.

## Table of Contents

1. [Why Integrate Data Ingestion and Preprocessing?](#why-integrate-data-ingestion-and-preprocessing)
2. [Pipeline Overview](#pipeline-overview)
3. [Implementation Approaches](#implementation-approaches)
   - [Functional Approach](#functional-approach)
   - [Pipeline API Approach](#pipeline-api-approach)
   - [Choosing the Right Approach](#choosing-the-right-approach)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Memory Management](#memory-management)
6. [Common Issues and Solutions](#common-issues-and-solutions)
7. [Next Steps](#next-steps)

## Why Integrate Data Ingestion and Preprocessing?

Traditionally, data processing pipelines separate ingestion and preprocessing into distinct phases. However, for text-based tasks like fake news detection, integrating these phases offers several advantages:

1. **Reduced Computation**: Eliminates redundant processing of the same data
2. **Improved Memory Efficiency**: Reduces overall memory usage by processing data once
3. **Simplified Workflow**: Fewer steps to manage and execute
4. **Enhanced Performance**: Particularly beneficial in resource-constrained environments
5. **Consistent Processing**: Ensures all data undergoes identical preprocessing

## Pipeline Overview

The integrated pipeline consists of the following steps:

1. **Data Loading**: Load raw data from CSV files or other sources
2. **Data Leakage Detection**: Analyze subject distribution to identify potential data leakage
3. **Text Preprocessing**: 
   - Extract location and news source information
   - Normalize acronyms (e.g., "U.S." → "US")
   - Convert text to lowercase
   - Remove special characters
   - Normalize spaces
4. **Tokenization**: Split text into individual tokens (words)
5. **Stopword Removal**: Remove common words that don't carry significant meaning
6. **Feature Extraction** (optional): Create numerical features from text using TF-IDF or other techniques
7. **Data Storage**: Save processed data for subsequent phases

## Implementation Approaches

There are two main approaches to implementing the integrated pipeline: functional and Pipeline API.

### Functional Approach

The functional approach uses separate functions for each step and a main function to orchestrate the entire process:

```python
def run_integrated_pipeline(fake_path, true_path, output_path=None, cache=True):
    """
    Runs the complete integrated pipeline for fake news detection.
    """
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
        processed_df.write.mode("overwrite").parquet(output_path)
    
    # Step 6: Unpersist DataFrames that are no longer needed
    if cache:
        fake_df.unpersist()
        true_df.unpersist()
        combined_df.unpersist()
    
    return processed_df
```

**Advantages**:
- Explicit control over each step
- Flexibility to modify individual components
- Easier to debug and understand
- Fine-grained memory management

**Disadvantages**:
- More verbose code
- Potential for inconsistency between steps
- May be less optimized than Pipeline API

### Pipeline API Approach

The Pipeline API approach uses Spark ML's Pipeline to define a sequence of transformations:

```python
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline

def create_text_processing_pipeline(include_features=True):
    """
    Creates a text processing pipeline using Spark ML Pipeline API.
    """
    # Define custom transformer for text preprocessing
    preprocessor = TextPreprocessor(
        inputCols=["text", "title"],
        outputCols=["text_processed", "title_processed"]
    )
    
    # Define standard ML transformers
    tokenizer = Tokenizer(inputCol="text_processed", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    
    # Create pipeline stages
    stages = [preprocessor, tokenizer, remover]
    
    # Optionally add feature extraction
    if include_features:
        hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        stages.extend([hashingTF, idf])
    
    # Create and return the pipeline
    return Pipeline(stages=stages)

def run_pipeline_api_approach(fake_path, true_path, output_path=None):
    """
    Runs the integrated pipeline using Pipeline API approach.
    """
    # Load and combine data
    fake_df, true_df = load_csv_files(fake_path, true_path)
    combined_df = combine_datasets(fake_df, true_df)
    
    # Create and fit the pipeline
    pipeline = create_text_processing_pipeline()
    model = pipeline.fit(combined_df)
    
    # Transform the data
    processed_df = model.transform(combined_df)
    
    # Save if needed
    if output_path:
        processed_df.write.mode("overwrite").parquet(output_path)
    
    return processed_df
```

**Advantages**:
- Concise and elegant code
- Consistent application of transformations
- Optimized execution by Spark
- Reusable pipeline model

**Disadvantages**:
- Less explicit control over individual steps
- May require custom transformers for complex operations
- Less flexibility for memory management

### Choosing the Right Approach

Choose the approach based on your specific needs:

- **Functional Approach**: Better for complex preprocessing with many custom steps and when memory management is critical
- **Pipeline API Approach**: Better for standard NLP tasks and when you need to reuse the same pipeline on multiple datasets

You can also combine both approaches by using custom transformers within the Pipeline API.

## Step-by-Step Implementation

### 1. Set Up Environment

```python
# Import required libraries
import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Create Spark session
spark = SparkSession.builder \
    .appName("FakeNewsDetection_IntegratedProcessing") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()
```

### 2. Implement Text Preprocessing

```python
def preprocess_text(df, cache=True):
    """
    Optimized text preprocessing function with acronym handling.
    """
    # Create a list to track columns that need preprocessing
    columns_to_preprocess = []
    
    # Check for text and title columns
    has_text = "text" in df.columns
    has_title = "title" in df.columns
    
    # Get column types
    if has_text:
        text_is_string = isinstance(df.schema["text"].dataType, StringType)
        if text_is_string:
            columns_to_preprocess.append("text")
    
    if has_title:
        title_is_string = isinstance(df.schema["title"].dataType, StringType)
        if title_is_string:
            columns_to_preprocess.append("title")
    
    # Extract location and news source
    if has_text and text_is_string:
        news_header_pattern = r"^(?:([A-Z][a-zA-Z\s\./,]*)\s*)?\(([^)]+)\)\s*-\s*(.*)"
        
        df = df.withColumn("location", regexp_extract(col("text"), news_header_pattern, 1)) \
               .withColumn("news_source", regexp_extract(col("text"), news_header_pattern, 2)) \
               .withColumn("text_cleaned", regexp_extract(col("text"), news_header_pattern, 3))
        
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
    
    # Apply acronym normalization
    if columns_to_preprocess:
        # Define function to normalize acronyms
        def normalize_acronyms(text):
            if text is None:
                return None
                
            replacements = [
                ("U.S.A.", "USA"),
                ("U.S.", "US"),
                ("U.N.", "UN"),
                # Add more acronyms as needed
            ]
            
            for pattern, replacement in replacements:
                text = text.replace(pattern, replacement)
                
            return text
        
        # Register UDF
        normalize_acronyms_udf = udf(normalize_acronyms, StringType())
        
        # Apply to each column
        for col_name in columns_to_preprocess:
            df = df.withColumn(col_name, normalize_acronyms_udf(col(col_name)))
            
            # Apply text transformations
            df = df.withColumn(
                col_name,
                trim(
                    regexp_replace(
                        regexp_replace(
                            lower(col(col_name)),
                            "[^a-z0-9\\s#@]", " "
                        ),
                        "\\s+", " "
                    )
                )
            )
    
    # Remove 'subject' column to prevent data leakage
    if "subject" in df.columns:
        df = df.drop("subject")
    
    # Cache if requested
    if cache:
        df.cache()
        df.count()  # Force materialization
    
    return df
```

### 3. Implement Tokenization and Stopword Removal

```python
def tokenize_text(df, text_column="text", output_column="tokens"):
    """
    Tokenize text into words.
    """
    tokenizer = Tokenizer(inputCol=text_column, outputCol=output_column)
    return tokenizer.transform(df)

def remove_stopwords(df, tokens_column="tokens", output_column="filtered_tokens"):
    """
    Remove stopwords from tokenized text.
    """
    remover = StopWordsRemover(inputCol=tokens_column, outputCol=output_column)
    return remover.transform(df)
```

### 4. Implement Complete Processing Function

```python
def complete_text_processing(df, cache=True):
    """
    Performs complete text processing in a single pass.
    """
    # Step 1: Preprocess text
    preprocessed_df = preprocess_text(df, cache=cache)
    
    # Step 2: Tokenize text
    tokenized_df = tokenize_text(preprocessed_df, text_column="text", output_column="tokens")
    
    # Step 3: Remove stopwords
    processed_df = remove_stopwords(tokenized_df, tokens_column="tokens", output_column="filtered_tokens")
    
    # Unpersist intermediate DataFrame if it was cached
    if cache and preprocessed_df != df:
        try:
            preprocessed_df.unpersist()
        except:
            pass
    
    return processed_df
```

### 5. Run the Complete Pipeline

```python
# Define paths
fake_path = "/path/to/Fake.csv"
true_path = "/path/to/True.csv"
output_path = "/path/to/processed_data"

# Run the integrated pipeline
processed_df = run_integrated_pipeline(
    fake_path=fake_path,
    true_path=true_path,
    output_path=output_path,
    cache=True
)

# Examine results
processed_df.printSchema()
display(processed_df.select("text", "tokens", "filtered_tokens", "label").limit(5))
```

## Memory Management

Effective memory management is crucial, especially in Databricks Community Edition:

### Strategic Caching

```python
# Cache DataFrame only when it will be used multiple times
df.cache()

# Force materialization to ensure caching is complete
df.count()
```

### Explicit Unpersisting

```python
# Unpersist DataFrame when it's no longer needed
df.unpersist()

# Verify it's been removed from memory
print(f"Is cached: {df.is_cached}")
```

### Column Pruning

```python
# Select only the columns you need
df = df.select("id", "text", "label")
```

### Partition Management

```python
# Set appropriate number of partitions based on cluster size
spark.conf.set("spark.sql.shuffle.partitions", "8")
```

## Common Issues and Solutions

### 1. Out of Memory Errors

**Problem**: Spark driver runs out of memory when processing large datasets.

**Solution**:
```python
# Reduce driver memory usage
spark.conf.set("spark.driver.memory", "8g")

# Increase number of partitions to distribute work
spark.conf.set("spark.sql.shuffle.partitions", "16")

# Use disk spill if necessary
spark.conf.set("spark.memory.storageFraction", "0.3")
```

### 2. Slow Text Processing

**Problem**: Text preprocessing operations are slow on large datasets.

**Solution**:
```python
# Use built-in functions instead of UDFs when possible
df = df.withColumn("text_lower", lower(col("text")))  # Faster than UDF

# Combine multiple operations in a single transformation
df = df.withColumn("text_clean", 
                  regexp_replace(
                      lower(
                          regexp_replace(col("text"), "http\\S+", "")
                      ), 
                      "[^a-zA-Z0-9\\s]", " "
                  ))
```

### 3. Pipeline API vs. Functional Approach

**Problem**: Deciding between Pipeline API and functional approach.

**Solution**: Create a custom transformer that encapsulates your preprocessing logic:

```python
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol

class TextPreprocessor(Transformer, HasInputCol, HasOutputCol):
    """
    Custom transformer for text preprocessing.
    """
    def __init__(self, inputCol=None, outputCol=None):
        super(TextPreprocessor, self).__init__()
        self.setInputCol(inputCol)
        self.setOutputCol(outputCol)
    
    def _transform(self, dataset):
        # Implement your preprocessing logic here
        return preprocess_text(dataset)
```

Then use it in a Pipeline:

```python
preprocessor = TextPreprocessor(inputCol="text", outputCol="text_processed")
tokenizer = Tokenizer(inputCol="text_processed", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")

pipeline = Pipeline(stages=[preprocessor, tokenizer, remover])
model = pipeline.fit(df)
result = model.transform(df)
```

## Next Steps

After completing the integrated text processing:

1. **Feature Engineering**: Create numerical features from the processed text
2. **Model Training**: Train machine learning models on the processed data
3. **Evaluation**: Evaluate model performance using appropriate metrics
4. **Deployment**: Deploy the model for real-time or batch prediction

By integrating data ingestion and preprocessing, you've created a more efficient pipeline that reduces redundancy and improves performance. This approach is particularly valuable in resource-constrained environments like Databricks Community Edition.

---

*Last updated: May 31, 2025*
