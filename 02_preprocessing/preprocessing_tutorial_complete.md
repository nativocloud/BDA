# Fake News Detection: Preprocessing Tutorial (Complete Guide)

*Last updated: May 31, 2025*

This comprehensive tutorial explains the preprocessing steps for the fake news detection project, with a focus on the unique characteristics of news datasets and implementation in Databricks Community Edition.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Analysis](#dataset-analysis)
3. [Data Leakage: The Subject Column Problem](#data-leakage-the-subject-column-problem)
4. [Preprocessing Pipeline Overview](#preprocessing-pipeline-overview)
5. [Implementation in PySpark](#implementation-in-pyspark)
6. [Running the Standalone Version](#running-the-standalone-version)
7. [Memory Management Best Practices](#memory-management-best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Cross-Platform Compatibility](#cross-platform-compatibility)
10. [Common Issues and Solutions](#common-issues-and-solutions)

## Introduction

Preprocessing is a critical step in any machine learning pipeline, but it's especially important for text-based tasks like fake news detection. This tutorial covers the preprocessing steps specifically tailored to news datasets, explaining why each step is necessary and how it's implemented in a distributed environment.

### Why Preprocessing Matters for Fake News Detection

Effective preprocessing for fake news detection:

1. **Removes noise** that could mislead the model
2. **Standardizes text** across different sources and formats
3. **Extracts meaningful features** from raw text
4. **Prevents data leakage** from metadata fields
5. **Optimizes performance** for large-scale processing

## Dataset Analysis

Before implementing preprocessing, we analyze the datasets to understand their characteristics:

### Basic Statistics
- **True.csv**: 21,418 articles labeled as real news
- **Fake.csv**: 23,490 articles labeled as fake news
- Both datasets have the same columns: title, text, subject, date

### Key Findings

1. **Subject Column Distribution**:
   - True.csv: All articles have subject = "politicsNews"
   - Fake.csv: All articles have subject = "News"
   - This creates a perfect data leakage problem (explained in detail below)

2. **URL Presence**:
   - Fake.csv: Contains numerous Twitter URLs (https://t.co/...) and other web links
   - True.csv: Contains fewer URLs but still requires consistent handling

3. **Social Media Content**:
   - Fake.csv: Contains embedded tweets with usernames, hashtags, and @mentions
   - True.csv: Contains minimal social media artifacts

4. **Date Format**:
   - True.csv: Dates have trailing spaces (e.g., "December 31, 2017 ")
   - Fake.csv: Dates are clean without trailing spaces (e.g., "December 31, 2017")

5. **Text Length**:
   - Articles vary significantly in length, with some extremely long articles

## Data Leakage: The Subject Column Problem

One of the most critical findings from our analysis is the perfect correlation between the 'subject' column and the article labels:

- **True.csv**: All 21,418 articles have subject = "politicsNews"
- **Fake.csv**: All 23,490 articles have subject = "News"

This creates a severe data leakage problem:

1. **Perfect Prediction**: If we include the 'subject' column as a feature, any model would learn to simply check if subject="politicsNews" to determine if an article is real or fake, achieving nearly 100% accuracy.

2. **No Generalization**: This "perfect" accuracy is misleading because:
   - It doesn't reflect the model's ability to detect fake news based on content
   - It won't generalize to new articles where the subject field might be different
   - It creates a false sense of model performance

3. **Scientific Integrity**: Including this column would invalidate any research findings since the model wouldn't be learning meaningful patterns in the text content.

**Solution**: We must drop the 'subject' column from our preprocessing pipeline to ensure our model learns from the actual content of the news articles rather than this artificial distinction.

## Preprocessing Pipeline Overview

Our preprocessing pipeline consists of the following steps:

### 1. Data Loading and Initial Validation
```python
# Load data from Hive tables or CSV files
real_df, fake_df = load_data_from_hive("real_news", "fake_news")

# Check for missing values
missing_values = check_missing_values(real_df)
display(missing_values)

# Check for duplicates
duplicates = check_duplicates(real_df)
print(f"Number of duplicate records: {duplicates}")
```

### 2. Data Cleaning and Standardization
```python
# Combine datasets and add label column
combined_df = combine_datasets(real_df, fake_df)

# Remove the 'subject' column to prevent data leakage
combined_df = combined_df.drop("subject")

# Standardize date format
combined_df = standardize_dates(combined_df, "date")
```

### 3. Text Preprocessing
```python
# Preprocess text fields
preprocessed_df = preprocess_text(combined_df, cache=True)
```

The `preprocess_text` function performs the following operations:
- Converts text to lowercase
- Removes URLs and special characters
- Extracts location and news source information
- Normalizes acronyms (e.g., "U.S." → "US")
- Removes multiple spaces

### 4. Feature Extraction Preparation
```python
# Tokenize text
tokenized_df = tokenize_text(preprocessed_df, "text")

# Remove stopwords
filtered_df = remove_stopwords(tokenized_df, "tokens")
```

### 5. Data Validation and Quality Checks
```python
# Validate the preprocessed data
validation_results = validate_preprocessed_data(filtered_df)
```

### 6. Memory Management
```python
# Unpersist DataFrames that are no longer needed
real_df.unpersist()
fake_df.unpersist()
combined_df.unpersist()
```

## Implementation in PySpark

Our implementation uses PySpark for distributed processing, which is essential for handling large news datasets efficiently. Here's how we implement key preprocessing steps:

### Text Preprocessing with PySpark

```python
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
        # Force materialization
        df.count()
    
    return df
```

### Tokenization and Stopword Removal

```python
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
```

### Acronym Handling

```python
def normalize_acronyms(text):
    """
    Normalize common acronyms in text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized acronyms
    """
    # Dictionary of acronyms and their normalized forms
    acronyms = {
        r'\bU\.S\.A\.\b': 'USA',
        r'\bU\.S\.\b': 'US',
        r'\bU\.K\.\b': 'UK',
        r'\bU\.N\.\b': 'UN',
        r'\bF\.B\.I\.\b': 'FBI',
        r'\bC\.I\.A\.\b': 'CIA',
        r'\bD\.C\.\b': 'DC',
        r'\bN\.Y\.\b': 'NY',
        r'\bL\.A\.\b': 'LA'
    }
    
    # Apply replacements
    for pattern, replacement in acronyms.items():
        text = re.sub(pattern, replacement, text)
    
    return text
```

## Running the Standalone Version

The standalone version of the preprocessing pipeline is designed to work in both script mode and interactive mode, with compatibility for Databricks Community Edition and local environments.

### Option 1: Running as a Script

```bash
# In a terminal or command prompt
python 02_preprocessing_standalone.py
```

### Option 2: Running in Databricks

1. Upload the `02_preprocessing_standalone.py` file to Databricks
2. Import it as a notebook (it will be converted to `02_preprocessing_standalone.ipynb`)
3. Run the cells sequentially

### Option 3: Running the Complete Pipeline

```python
# Import the standalone script
%run "./02_preprocessing_standalone"

# Run the complete preprocessing pipeline
preprocessed_df = run_preprocessing_pipeline(
    fake_path="/path/to/Fake.csv",
    true_path="/path/to/True.csv",
    output_path="/path/to/output",
    cache=True
)
```

## Memory Management Best Practices

When working with large news datasets, proper memory management is essential, especially in environments with limited resources like Databricks Community Edition.

### 1. Strategic Caching

```python
# Cache DataFrame only when it will be used multiple times
df.cache()

# Force materialization to ensure caching is complete
df.count()
```

### 2. Explicit Unpersisting

```python
# Unpersist DataFrame when it's no longer needed
df.unpersist()

# Verify it's been removed from memory
print(f"Is cached: {df.is_cached}")
```

### 3. Column Pruning

```python
# Select only the columns you need
df = df.select("id", "text", "label")
```

### 4. Partition Management

```python
# Set appropriate number of partitions based on cluster size
spark.conf.set("spark.sql.shuffle.partitions", "8")
```

## Performance Optimization

To optimize performance in Databricks Community Edition:

### 1. Spark Configuration

```python
# Optimize Spark configuration for Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsDetection_Preprocessing") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()
```

### 2. Broadcast Variables

```python
# Broadcast small lookup tables or dictionaries
acronyms_dict = {"U.S.": "US", "U.K.": "UK"}
acronyms_broadcast = spark.sparkContext.broadcast(acronyms_dict)
```

### 3. Avoid Collect on Large DataFrames

```python
# Instead of collecting the entire DataFrame
# df_collect = df.collect()  # Bad practice for large DataFrames

# Sample a small subset
sample_df = df.limit(10).collect()
```

### 4. Use SQL for Complex Operations

```python
# Register DataFrame as a temporary view
df.createOrReplaceTempView("news_data")

# Use SQL for complex operations
result = spark.sql("""
    SELECT label, COUNT(*) as count
    FROM news_data
    GROUP BY label
    ORDER BY label
""")
```

## Cross-Platform Compatibility

The standalone preprocessing script is designed to work across different environments:

### Environment Detection

```python
def is_databricks():
    """Check if running in Databricks environment"""
    try:
        return 'dbutils' in globals()
    except:
        return False

def get_file_system_path(path):
    """Get the appropriate file system path based on environment"""
    if is_databricks():
        return f"dbfs:{path}"
    else:
        return path.replace("/dbfs", "")
```

### Display Function Compatibility

```python
def compatible_display(df, n=10):
    """Display DataFrame in a way that works in both environments"""
    if is_databricks():
        display(df)
    else:
        print(df.limit(n).toPandas())
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

### 3. Data Leakage Detection

**Problem**: Difficult to identify potential data leakage sources.

**Solution**:
```python
# Analyze column distributions by label
def check_column_distribution(df, column_name):
    df.groupBy(column_name, "label") \
      .count() \
      .orderBy(column_name, "label") \
      .show()
    
# Check all categorical columns
for col_name in categorical_columns:
    check_column_distribution(df, col_name)
```

---

By following this comprehensive preprocessing tutorial, you'll ensure that your fake news detection model learns from the actual content of the articles rather than artificial patterns or irrelevant features, while optimizing performance in Databricks Community Edition.

# Last modified: May 31, 2025
