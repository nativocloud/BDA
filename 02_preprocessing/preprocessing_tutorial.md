# Fake News Detection: Preprocessing Tutorial

*Last updated: May 29, 2025*

This tutorial explains the dataset-specific preprocessing steps for the fake news detection project, with a focus on the unique characteristics of the True.csv and Fake.csv datasets.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Analysis](#dataset-analysis)
3. [Data Leakage: The Subject Column Problem](#data-leakage-the-subject-column-problem)
4. [Dataset-Specific Preprocessing Steps](#dataset-specific-preprocessing-steps)
5. [Implementation in PySpark](#implementation-in-pyspark)
6. [Usage Examples](#usage-examples)
7. [Performance Considerations](#performance-considerations)

## Introduction

Preprocessing is a critical step in any machine learning pipeline, but it's especially important for text-based tasks like fake news detection. This tutorial covers the preprocessing steps specifically tailored to our fake news datasets, explaining why each step is necessary and how it's implemented.

## Dataset Analysis

Before implementing preprocessing, we analyzed the True.csv and Fake.csv datasets to understand their characteristics:

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

## Dataset-Specific Preprocessing Steps

Based on our analysis, we've implemented the following dataset-specific preprocessing steps:

### 1. Subject Column Removal
```python
# Drop the subject column to prevent data leakage
df = df.drop('subject')
```

### 2. Enhanced URL Removal
```python
# Remove URLs with enhanced pattern for Twitter URLs
text = re.sub(r'https?://t\.co/\w+|http\S+|www\S+|https\S+', '', text)
```

### 3. Social Media Content Handling
```python
# Remove Twitter handles
text = re.sub(r'@\w+', '', text)

# Convert hashtags to plain text
text = re.sub(r'#(\w+)', r'\1', text)
```

### 4. Special Character Handling
```python
# Remove emojis and special unicode characters
text = text.encode('ascii', 'ignore').decode('ascii')
```

### 5. Date Standardization
```python
# Trim whitespace from date fields
df = df.withColumn("date", trim(col("date")))
```

### 6. Text Length Normalization
```python
# Truncate to max 500 words for consistency
words = text.split()
if len(words) > 500:
    text = ' '.join(words[:500])
```

### 7. Quoted Content Extraction
```python
# Identify quoted content
quoted_text = re.findall(r'"([^"]*)"', text)
```

## Implementation in PySpark

We've implemented these preprocessing steps in the `EnhancedTextPreprocessor` class, which extends the functionality of the original `TextPreprocessor` with dataset-specific operations.

Key features of the implementation:

1. **Configurable Options**: All preprocessing steps can be enabled/disabled through a configuration dictionary.

2. **PySpark Integration**: The preprocessor works with both pandas DataFrames and Spark DataFrames for scalability.

3. **Optimized for Databricks**: The implementation is optimized for the Databricks Community Edition's resource constraints.

4. **Pipeline Integration**: The preprocessor can be easily integrated into the full fake news detection pipeline.

## Usage Examples

### Basic Usage

```python
from BDA.02_preprocessing.enhanced_text_preprocessor import EnhancedTextPreprocessor

# Create a preprocessor with default configuration
preprocessor = EnhancedTextPreprocessor()

# Preprocess a Spark DataFrame
preprocessed_df = preprocessor.preprocess_spark_df(
    spark_df, 
    text_column="text", 
    title_column="title", 
    combine_title_text=True
)
```

### Custom Configuration

```python
# Custom configuration
config = {
    'remove_stopwords': True,
    'stemming': False,
    'lemmatization': True,
    'lowercase': True,
    'remove_punctuation': True,
    'remove_numbers': False,
    'remove_urls': True,
    'remove_twitter_handles': True,
    'remove_hashtags': False,
    'convert_hashtags': True,
    'remove_special_chars': True,
    'normalize_whitespace': True,
    'max_text_length': 500,
    'language': 'english'
}

# Create a preprocessor with custom configuration
preprocessor = EnhancedTextPreprocessor(config)
```

## Performance Considerations

When working with large datasets in the Databricks Community Edition, consider these performance optimizations:

1. **Partition Management**: Adjust the number of partitions based on your dataset size:
   ```python
   spark.conf.set("spark.sql.shuffle.partitions", "8")
   ```

2. **Memory Management**: Limit the driver memory to avoid OOM errors:
   ```python
   spark.conf.set("spark.driver.memory", "8g")
   ```

3. **Text Length Limitation**: Consider limiting the maximum text length for very large datasets:
   ```python
   config['max_text_length'] = 300  # Reduce from default 500
   ```

4. **Caching Strategy**: Cache intermediate results when performing multiple operations:
   ```python
   preprocessed_df.cache()
   ```

By following these dataset-specific preprocessing steps, you'll ensure that your fake news detection model learns from the actual content of the articles rather than artificial patterns or irrelevant features.

# Last modified: May 29, 2025
