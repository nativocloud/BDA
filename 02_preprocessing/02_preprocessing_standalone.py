# %% [markdown]
# # Fake News Detection: Data Preprocessing
# 
# This notebook contains all the necessary code for preprocessing data in the fake news detection project. The code is organized into independent functions, without dependencies on external modules or classes, to facilitate execution in Databricks Community Edition.

# %% [markdown]
# ## Setup and Imports

# %%
# Import necessary libraries
import re
import string
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, when, isnan, isnull, desc, expr, lit, trim, 
    length, lower, upper, regexp_replace, udf, to_date, year, 
    month, dayofmonth, dayofweek, date_format, concat
)
from pyspark.sql.types import StringType, BooleanType, IntegerType, DateType, ArrayType
from pyspark.ml.feature import Tokenizer, StopWordsRemover

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

# %%
# Initialize Spark session optimized for Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsDetection_Preprocessing") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

# Display Spark configuration
print(f"Spark version: {spark.version}")
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")

# %% [markdown]
# ## Reusable Functions

# %% [markdown]
# ### Data Loading Functions

# %%
def load_data_from_hive(fake_table_name="fake", true_table_name="real"):
    """
    Load data from Hive tables.
    
    Args:
        fake_table_name (str): Name of the Hive table with fake news
        true_table_name (str): Name of the Hive table with real news
        
    Returns:
        tuple: (real_df, fake_df) DataFrames with loaded data
    """
    print(f"Loading data from Hive tables '{true_table_name}' and '{fake_table_name}'...")
    
    # Check if tables exist
    tables = [row.tableName for row in spark.sql("SHOW TABLES").collect()]
    
    if true_table_name not in tables or fake_table_name not in tables:
        raise ValueError(f"Hive tables '{true_table_name}' and/or '{fake_table_name}' do not exist")
    
    # Load data from Hive tables
    real_df = spark.table(true_table_name)
    fake_df = spark.table(fake_table_name)
    
    # Register as temporary views for SQL queries
    real_df.createOrReplaceTempView("real_news")
    fake_df.createOrReplaceTempView("fake_news")
    
    # Display information about DataFrames
    print(f"Real news loaded: {real_df.count()} records")
    print(f"Fake news loaded: {fake_df.count()} records")
    
    return real_df, fake_df

# %% [markdown]
# ### Data Validation Functions

# %%
def check_missing_values(df):
    """
    Check for missing values in all columns of a DataFrame.
    
    Args:
        df: Spark DataFrame to check
        
    Returns:
        DataFrame: DataFrame with missing value counts for each column
    """
    # Create expressions for each column to count nulls and empty strings
    exprs = []
    for col_name in df.columns:
        # Count nulls
        exprs.append(count(when(col(col_name).isNull() | 
                               (col(col_name) == "") | 
                               isnan(col_name), 
                               col_name)).alias(col_name))
    
    # Apply expressions to get missing value counts
    missing_values = df.select(*exprs)
    
    return missing_values

# %%
def check_duplicates(df):
    """
    Check for duplicate records in a DataFrame.
    
    Args:
        df: Spark DataFrame to check
        
    Returns:
        int: Number of duplicate records
    """
    # Count total records
    total_records = df.count()
    
    # Count distinct records
    distinct_records = df.distinct().count()
    
    # Calculate duplicates
    duplicates = total_records - distinct_records
    
    return duplicates

# %%
def has_suspicious_pattern(text):
    """
    Check if text contains suspicious patterns.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if suspicious patterns are found, False otherwise
    """
    if not text:
        return False
    
    # Common patterns for problematic content
    suspicious_patterns = [
        r'<\s*script.*?>.*?<\s*/\s*script\s*>', # Script tags
        r'<\s*style.*?>.*?<\s*/\s*style\s*>',   # Style tags
        r'<\s*iframe.*?>.*?<\s*/\s*iframe\s*>', # iFrame tags
        r'javascript:',                         # JavaScript protocol
        r'data:',                               # Data URI scheme
        r'[\u0080-\u00ff]{10,}',                # Long sequences of non-ASCII chars
        r'(\w)\1{10,}'                          # Repeated characters (e.g., "aaaaaaaaaa")
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return True
    
    return False

# %%
def validate_text_fields(df, text_columns, min_title_length=5, min_text_length=50):
    """
    Validate text fields in the DataFrame.
    
    Args:
        df (DataFrame): The input DataFrame
        text_columns (list): List of text column names to validate
        min_title_length (int): Minimum acceptable length for title fields
        min_text_length (int): Minimum acceptable length for text fields
        
    Returns:
        DataFrame: The DataFrame with additional validation flag columns
    """
    result_df = df
    
    for column in text_columns:
        # Check for null or empty values
        result_df = result_df.withColumn(
            f"{column}_present", 
            when(
                (col(column).isNull()) | (trim(col(column)) == ""),
                lit(False)
            ).otherwise(lit(True))
        )
        
        # Check for minimum length
        min_length = min_title_length if "title" in column else min_text_length
        result_df = result_df.withColumn(
            f"{column}_length_valid",
            when(
                col(f"{column}_present") & (length(trim(col(column))) >= min_length),
                lit(True)
            ).otherwise(lit(False))
        )
        
        # Check for suspicious patterns
        has_suspicious_pattern_udf = udf(has_suspicious_pattern, BooleanType())
        result_df = result_df.withColumn(
            f"{column}_content_valid",
            when(
                col(f"{column}_present") & (~has_suspicious_pattern_udf(col(column))),
                lit(True)
            ).otherwise(lit(False))
        )
        
        # Overall validation flag for this column
        result_df = result_df.withColumn(
            f"{column}_valid",
            col(f"{column}_present") & 
            col(f"{column}_length_valid") & 
            col(f"{column}_content_valid")
        )
    
    return result_df

# %%
def standardize_text_fields(df, text_columns):
    """
    Standardize text fields in the DataFrame.
    
    Args:
        df (DataFrame): The input DataFrame
        text_columns (list): List of text column names to standardize
        
    Returns:
        DataFrame: The DataFrame with standardized text columns
    """
    result_df = df
    
    for column in text_columns:
        # Create standardized version of the column
        std_column = f"std_{column}"
        
        # Clean whitespace and normalize case
        result_df = result_df.withColumn(
            std_column,
            when(
                col(column).isNull(),
                None
            ).otherwise(
                trim(regexp_replace(col(column), r'\s+', ' '))
            )
        )
        
        # Remove HTML tags
        result_df = result_df.withColumn(
            std_column,
            regexp_replace(col(std_column), r'<[^>]+>', ' ')
        )
        
        # Remove URLs
        result_df = result_df.withColumn(
            std_column,
            regexp_replace(col(std_column), r'https?://\S+', '[URL]')
        )
        
        # Normalize case based on column type
        if "title" in column:
            # For titles, use title case
            result_df = result_df.withColumn(
                std_column,
                expr("initcap(" + std_column + ")")
            )
        elif "author" in column or "source" in column:
            # For author and source, use title case
            result_df = result_df.withColumn(
                std_column,
                expr("initcap(" + std_column + ")")
            )
        else:
            # For other text, leave as is
            pass
    
    return result_df

# %%
def calculate_data_quality_metrics(df):
    """
    Calculate data quality metrics for the DataFrame.
    
    Args:
        df (DataFrame): The input DataFrame with validation columns
        
    Returns:
        dict: A dictionary of data quality metrics
    """
    # Get total number of rows
    total_rows = df.count()
    
    # Initialize metrics dictionary
    metrics = {
        "total_rows": total_rows,
        "completeness": {},
        "validity": {}
    }
    
    # Calculate completeness metrics for each column
    for column in df.columns:
        if f"{column}_present" in df.columns:
            present_count = df.filter(col(f"{column}_present") == True).count()
            completeness = present_count / total_rows if total_rows > 0 else 0
            metrics["completeness"][column] = completeness
        
        if f"{column}_valid" in df.columns:
            valid_count = df.filter(col(f"{column}_valid") == True).count()
            validity = valid_count / total_rows if total_rows > 0 else 0
            metrics["validity"][column] = validity
    
    # Calculate overall data quality score
    if metrics["validity"]:
        metrics["overall_quality"] = sum(metrics["validity"].values()) / len(metrics["validity"])
    else:
        metrics["overall_quality"] = 0
    
    return metrics

# %% [markdown]
# ### Date Processing Functions

# %%
def parse_complex_date(date_str):
    """
    Parse complex date formats that aren't handled by standard Spark functions.
    
    Args:
        date_str (str): The date string to parse
        
    Returns:
        datetime: The parsed date as a datetime object, or None if parsing fails
    """
    if not date_str:
        return None
    
    # Clean the date string
    date_str = date_str.strip()
    
    # Try common Python datetime formats
    formats = [
        "%B %d, %Y",      # December 25, 2017
        "%b %d, %Y",      # Dec 25, 2017
        "%Y-%m-%d",       # 2017-12-25
        "%m/%d/%Y",       # 12/25/2017
        "%d/%m/%Y",       # 25/12/2017
        "%Y/%m/%d",       # 2017/12/25
        "%m-%d-%Y",       # 12-25-2017
        "%d-%m-%Y"        # 25-12-2017
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try to extract date components using regex
    # This handles formats like "25th December 2017" or "December 2017"
    try:
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if not year_match:
            return None
        year = int(year_match.group(0))
        
        # Extract month
        month = None
        month_names = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        
        for name, num in month_names.items():
            if name in date_str.lower():
                month = num
                break
        
        if not month:
            # Try to find numeric month
            month_match = re.search(r'\b(0?[1-9]|1[0-2])\b', date_str)
            if month_match:
                month = int(month_match.group(0))
            else:
                month = 1  # Default to January if no month found
        
        # Extract day
        day = None
        day_match = re.search(r'\b(0?[1-9]|[12][0-9]|3[01])(st|nd|rd|th)?\b', date_str)
        if day_match:
            day = int(re.sub(r'(st|nd|rd|th)', '', day_match.group(0)))
        else:
            day = 1  # Default to 1st if no day found
        
        return datetime(year, month, day)
    
    except (ValueError, TypeError):
        return None

# %%
def standardize_date(df, date_column="publish_date"):
    """
    Standardize the date format in the DataFrame.
    
    Args:
        df (DataFrame): The input DataFrame with a date column
        date_column (str): The name of the date column to process
        
    Returns:
        DataFrame: The DataFrame with a standardized date column
    """
    # Common date formats to try when parsing dates
    date_formats = [
        "MMMM d, yyyy",       # December 25, 2017
        "MMM d, yyyy",        # Dec 25, 2017
        "yyyy-MM-dd",         # 2017-12-25
        "MM/dd/yyyy",         # 12/25/2017
        "dd/MM/yyyy",         # 25/12/2017
        "yyyy/MM/dd",         # 2017/12/25
        "MM-dd-yyyy",         # 12-25-2017
        "dd-MM-yyyy"          # 25-12-2017
    ]
    
    # Start with the original DataFrame
    result_df = df.withColumn("std_date", lit(None).cast(DateType()))
    
    # Try to parse the date using each format
    for date_format in date_formats:
        result_df = result_df.withColumn(
            "std_date",
            when(
                col("std_date").isNull(),
                to_date(col(date_column), date_format)
            ).otherwise(col("std_date"))
        )
    
    # For any remaining null values, use a more complex approach with UDFs
    # This is a fallback for unusual formats
    parse_complex_date_udf = udf(parse_complex_date, DateType())
    
    result_df = result_df.withColumn(
        "std_date",
        when(
            col("std_date").isNull() & col(date_column).isNotNull(),
            parse_complex_date_udf(col(date_column))
        ).otherwise(col("std_date"))
    )
    
    return result_df

# %%
def extract_date_features(df):
    """
    Extract useful date features from the standardized date column.
    
    Args:
        df (DataFrame): The input DataFrame with a standardized date column
        
    Returns:
        DataFrame: The DataFrame with additional date feature columns
    """
    # Extract date components
    result_df = df.withColumn("year", year(col("std_date")))
    result_df = result_df.withColumn("month", month(col("std_date")))
    result_df = result_df.withColumn("day", dayofmonth(col("std_date")))
    result_df = result_df.withColumn("day_of_week", dayofweek(col("std_date")))
    
    # Create YYYYMMDD format
    result_df = result_df.withColumn(
        "date_yyyymmdd", 
        date_format(col("std_date"), "yyyyMMdd").cast(IntegerType())
    )
    
    # Add validation flag
    result_df = result_df.withColumn(
        "date_valid",
        col("std_date").isNotNull()
    )
    
    return result_df

# %%
def process_date_column(df, date_column="publish_date"):
    """
    Process the date column in the DataFrame.
    
    Args:
        df (DataFrame): The input DataFrame with a date column
        date_column (str): The name of the date column to process
        
    Returns:
        DataFrame: The DataFrame with processed date columns
    """
    # Standardize the date
    result_df = standardize_date(df, date_column)
    
    # Extract date features
    result_df = extract_date_features(result_df)
    
    return result_df

# %% [markdown]
# ### Text Preprocessing Functions

# %%
def preprocess_text(text, config=None):
    """
    Apply all configured preprocessing steps to a text string.
    
    Args:
        text (str): The input text to preprocess
        config (dict): Configuration parameters for preprocessing
            
    Returns:
        str: The preprocessed text
    """
    # Default configuration
    default_config = {
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
        'max_text_length': 500,  # 0 means no limit
        'language': 'english'
    }
    
    # Use provided config or default
    if config is None:
        config = default_config
    
    if not text or not isinstance(text, str):
        return ""
    
    # Initialize tools based on configuration
    if config['remove_stopwords']:
        stop_words = set(stopwords.words(config['language']))
    
    if config['stemming']:
        stemmer = PorterStemmer()
        
    if config['lemmatization']:
        lemmatizer = WordNetLemmatizer()
    
    # Normalize whitespace
    if config['normalize_whitespace']:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove URLs (enhanced pattern for Twitter URLs)
    if config['remove_urls']:
        text = re.sub(r'https?://t\.co/\w+|http\S+|www\S+|https\S+', '', text)
    
    # Remove Twitter handles
    if config['remove_twitter_handles']:
        text = re.sub(r'@\w+', '', text)
    
    # Handle hashtags
    if config['remove_hashtags']:
        text = re.sub(r'#\w+', '', text)
    elif config['convert_hashtags']:
        text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove special characters and emojis
    if config['remove_special_chars']:
        text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Convert to lowercase
    if config['lowercase']:
        text = text.lower()
    
    # Remove punctuation
    if config['remove_punctuation']:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    if config['remove_numbers']:
        text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Limit text length if configured
    if config['max_text_length'] > 0 and len(tokens) > config['max_text_length']:
        tokens = tokens[:config['max_text_length']]
    
    # Remove stopwords
    if config['remove_stopwords']:
        tokens = [word for word in tokens if word not in stop_words]
    
    # Apply stemming
    if config['stemming']:
        tokens = [stemmer.stem(word) for word in tokens]
    
    # Apply lemmatization
    if config['lemmatization']:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

# %%
def preprocess_spark_df(spark_df, text_column, title_column=None, combine_title_text=False, config=None):
    """
    Preprocess text in a Spark DataFrame.
    
    Args:
        spark_df (DataFrame): The input Spark DataFrame
        text_column (str): The name of the column containing text to preprocess
        title_column (str, optional): The name of the column containing title text
        combine_title_text (bool): Whether to combine title and text for preprocessing
        config (dict, optional): Configuration parameters for preprocessing
            
    Returns:
        DataFrame: DataFrame with preprocessed text in a new column named 'processed_text'
    """
    # Register UDF for preprocessing
    preprocess_udf = udf(lambda text: preprocess_text(text, config), StringType())
    
    # Create a copy to work with
    result_df = spark_df
    
    # Trim whitespace from string columns
    for column in result_df.schema.fields:
        if isinstance(column.dataType, StringType):
            result_df = result_df.withColumn(column.name, trim(col(column.name)))
    
    # Drop the subject column if it exists (with explanation in the notebook)
    if 'subject' in result_df.columns:
        result_df = result_df.drop('subject')
    
    # Combine title and text if requested
    if combine_title_text and title_column and title_column in result_df.columns:
        result_df = result_df.withColumn(
            'combined_text', 
            concat(col(title_column).cast(StringType()).fillna(''), lit(' '), col(text_column).cast(StringType()).fillna(''))
        )
        result_df = result_df.withColumn('processed_text', preprocess_udf(col('combined_text')))
    else:
        result_df = result_df.withColumn('processed_text', preprocess_udf(col(text_column)))
    
    return result_df

# %%
def tokenize_text(text, config=None):
    """
    Tokenize text into words.
    
    Args:
        text (str): The input text to tokenize
        config (dict, optional): Configuration parameters for preprocessing
            
    Returns:
        list: List of tokens
    """
    if not text or not isinstance(text, str):
        return []
    
    # First apply preprocessing to get clean text
    clean_text = preprocess_text(text, config)
    
    # Tokenize the clean text
    tokens = word_tokenize(clean_text)
    
    return tokens

# %%
def tokenize_spark_df(spark_df, text_column, config=None):
    """
    Tokenize text in a Spark DataFrame.
    
    Args:
        spark_df (DataFrame): The input Spark DataFrame
        text_column (str): The name of the column containing text to tokenize
        config (dict, optional): Configuration parameters for preprocessing
            
    Returns:
        DataFrame: DataFrame with tokenized text in a new column named 'tokens'
    """
    # Register UDF for tokenization
    tokenize_udf = udf(lambda text: tokenize_text(text, config), ArrayType(StringType()))
    
    # Apply tokenization to the specified column
    tokenized_df = spark_df.withColumn('tokens', tokenize_udf(col(text_column)))
    
    return tokenized_df

# %%
def extract_quoted_content(text):
    """
    Extract content within quotation marks.
    
    Args:
        text (str): The input text
        
    Returns:
        list: List of quoted content
    """
    if not text or not isinstance(text, str):
        return []
    
    # Find all quoted content
    quoted_content = re.findall(r'"([^"]*)"', text)
    
    return quoted_content

# %%
def extract_quoted_content_spark_df(spark_df, text_column):
    """
    Extract quoted content from text in a Spark DataFrame.
    
    Args:
        spark_df (DataFrame): The input Spark DataFrame
        text_column (str): The name of the column containing text
        
    Returns:
        DataFrame: DataFrame with quoted content in a new column named 'quoted_content'
    """
    # Register UDF for extracting quoted content
    extract_quoted_udf = udf(extract_quoted_content, ArrayType(StringType()))
    
    # Apply extraction to the specified column
    result_df = spark_df.withColumn('quoted_content', extract_quoted_udf(col(text_column)))
    
    return result_df

# %% [markdown]
# ### Dataset Analysis Functions

# %%
def analyze_dataset_characteristics(df, sample_size=1000):
    """
    Analyze dataset characteristics to inform preprocessing decisions.
    
    Args:
        df (DataFrame): The input DataFrame
        sample_size (int): Number of rows to sample
        
    Returns:
        dict: Dictionary of dataset characteristics
    """
    # Sample the data
    sample_df = df.limit(sample_size)
    
    # Collect basic statistics
    stats = {
        "total_rows": df.count(),
        "columns": df.columns,
        "sample_size": sample_size
    }
    
    # Check for subject column distribution
    if "subject" in df.columns:
        subject_counts = df.groupBy("subject").count().collect()
        stats["subject_distribution"] = {row["subject"]: row["count"] for row in subject_counts}
    
    # Check for URL presence in text
    if "text" in df.columns:
        # Register UDF to check for URLs
        has_url_udf = udf(lambda text: 1 if re.search(r'http\S+|www\S+|https\S+', text) else 0, StringType())
        url_count = sample_df.withColumn("has_url", has_url_udf(col("text"))).filter(col("has_url") == 1).count()
        stats["url_presence_percentage"] = (url_count / sample_size) * 100
    
    # Check for Twitter handles
    if "text" in df.columns:
        has_handle_udf = udf(lambda text: 1 if re.search(r'@\w+', text) else 0, StringType())
        handle_count = sample_df.withColumn("has_handle", has_handle_udf(col("text"))).filter(col("has_handle") == 1).count()
        stats["twitter_handle_percentage"] = (handle_count / sample_size) * 100
    
    # Check for hashtags
    if "text" in df.columns:
        has_hashtag_udf = udf(lambda text: 1 if re.search(r'#\w+', text) else 0, StringType())
        hashtag_count = sample_df.withColumn("has_hashtag", has_hashtag_udf(col("text"))).filter(col("has_hashtag") == 1).count()
        stats["hashtag_percentage"] = (hashtag_count / sample_size) * 100
    
    # Check text length distribution
    if "text" in df.columns:
        text_length_df = sample_df.withColumn("text_length", length(col("text")))
        text_length_stats = text_length_df.select("text_length").summary("min", "25%", "mean", "75%", "max").collect()
        stats["text_length"] = {row["summary"]: float(row["text_length"]) for row in text_length_stats}
    
    # Print summary
    print("Dataset Characteristics:")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Columns: {stats['columns']}")
    
    if "subject_distribution" in stats:
        print("\nSubject Distribution:")
        for subject, count in stats["subject_distribution"].items():
            print(f"  {subject}: {count}")
    
    if "url_presence_percentage" in stats:
        print(f"\nURL presence: {stats['url_presence_percentage']:.2f}%")
    
    if "twitter_handle_percentage" in stats:
        print(f"Twitter handle presence: {stats['twitter_handle_percentage']:.2f}%")
    
    if "hashtag_percentage" in stats:
        print(f"Hashtag presence: {stats['hashtag_percentage']:.2f}%")
    
    if "text_length" in stats:
        print("\nText Length Statistics:")
        for stat, value in stats["text_length"].items():
            print(f"  {stat}: {value}")
    
    return stats

# %%
def contains_url(text):
    """
    Check if text contains URLs.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if URLs are found, False otherwise
    """
    if text is None:
        return False
    url_pattern = r'https?://t\.co/\w+|http\S+|www\S+|https\S+'
    return bool(re.search(url_pattern, text))

# %%
def contains_twitter_handle(text):
    """
    Check if text contains Twitter handles.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if Twitter handles are found, False otherwise
    """
    if text is None:
        return False
    pattern = r'@\w+'
    return bool(re.search(pattern, text))

# %%
def contains_hashtag(text):
    """
    Check if text contains hashtags.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if hashtags are found, False otherwise
    """
    if text is None:
        return False
    pattern = r'#\w+'
    return bool(re.search(pattern, text))

# %% [markdown]
# ### Data Storage Functions

# %%
def save_to_parquet(df, path, partition_by=None):
    """
    Save a DataFrame in Parquet format.
    
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
    Save a DataFrame to a Hive table.
    
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
# ## Complete Preprocessing Pipeline

# %%
def preprocess_and_save_data(output_dir="dbfs:/FileStore/fake_news_detection/preprocessed_data",
                            fake_table_name="fake", 
                            true_table_name="real",
                            create_tables=True):
    """
    Preprocess and save fake and real news data.
    
    This complete pipeline loads data from Hive tables, validates and cleans the data,
    preprocesses text, and saves the results in Parquet format and as Hive tables.
    
    Args:
        output_dir (str): Directory to save processed data
        fake_table_name (str): Name of the Hive table with fake news
        true_table_name (str): Name of the Hive table with real news
        create_tables (bool): Whether to create Hive tables
        
    Returns:
        dict: Dictionary with references to processed DataFrames
    """
    print("Starting data preprocessing pipeline...")
    
    # 1. Load data from Hive tables
    real_df, fake_df = load_data_from_hive(fake_table_name, true_table_name)
    
    # 2. Combine datasets with labels
    real_with_label = real_df.withColumn("label", lit(1))
    fake_with_label = fake_df.withColumn("label", lit(0))
    combined_df = real_with_label.union(fake_with_label)
    
    # 3. Analyze dataset characteristics
    analyze_dataset_characteristics(combined_df)
    
    # 4. Validate text fields
    text_columns = ["title", "text"]
    validated_df = validate_text_fields(combined_df, text_columns)
    
    # 5. Process date column if it exists
    if "date" in combined_df.columns or "publish_date" in combined_df.columns:
        date_column = "date" if "date" in combined_df.columns else "publish_date"
        validated_df = process_date_column(validated_df, date_column)
    
    # 6. Preprocess text
    preprocessed_df = preprocess_spark_df(validated_df, "text", "title", combine_title_text=True)
    
    # 7. Tokenize text
    tokenized_df = tokenize_spark_df(preprocessed_df, "processed_text")
    
    # 8. Extract quoted content
    final_df = extract_quoted_content_spark_df(tokenized_df, "text")
    
    # 9. Save preprocessed data
    preprocessed_path = f"{output_dir}/preprocessed_news.parquet"
    save_to_parquet(final_df, preprocessed_path, partition_by="label")
    
    # 10. Save to Hive table for easier access
    if create_tables:
        save_to_hive_table(final_df, "preprocessed_news", partition_by="label")
    
    print("\nData preprocessing pipeline completed successfully!")
    
    return {
        "real_df": real_df,
        "fake_df": fake_df,
        "combined_df": combined_df,
        "preprocessed_df": preprocessed_df,
        "final_df": final_df
    }

# %% [markdown]
# ## Step-by-Step Tutorial

# %% [markdown]
# ### 1. Load Data from Hive Tables

# %%
# Load data from Hive tables
real_df, fake_df = load_data_from_hive()

# Display sample data
print("Real news sample:")
real_df.limit(3).show(truncate=False)

print("\nFake news sample:")
fake_df.limit(3).show(truncate=False)

# %% [markdown]
# ### 2. Data Exploration and Validation

# %%
# Check for missing values in both datasets
print("Missing values in real news dataset:")
check_missing_values(real_df).show()

print("\nMissing values in fake news dataset:")
check_missing_values(fake_df).show()

# %%
# Check for duplicates
print(f"Duplicates in real news dataset: {check_duplicates(real_df)}")
print(f"Duplicates in fake news dataset: {check_duplicates(fake_df)}")

# %%
# Check text length distribution
real_text_lengths = real_df.select(expr("size(split(text, ' '))").alias("length"))
fake_text_lengths = fake_df.select(expr("size(split(text, ' '))").alias("length"))

# Convert to pandas for visualization
real_lengths_pd = real_text_lengths.toPandas()
fake_lengths_pd = fake_text_lengths.toPandas()

# Plot text length distributions
plt.figure(figsize=(12, 6))
plt.hist(real_lengths_pd['length'], bins=50, alpha=0.5, label='Real News')
plt.hist(fake_lengths_pd['length'], bins=50, alpha=0.5, label='Fake News')
plt.xlabel('Text Length (words)')
plt.ylabel('Frequency')
plt.title('Text Length Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print summary statistics
print("Real news text length statistics:")
real_text_lengths.summary().show()

print("\nFake news text length statistics:")
fake_text_lengths.summary().show()

# %% [markdown]
# ### 3. Analyzing the 'subject' Column - Data Leakage Issue

# %%
# Check subject distribution in real news
print("Subject distribution in real news:")
if "subject" in real_df.columns:
    real_subjects = real_df.groupBy("subject").count().orderBy(desc("count"))
    real_subjects.show()
else:
    print("No 'subject' column in real news dataset")

# Check subject distribution in fake news
print("\nSubject distribution in fake news:")
if "subject" in fake_df.columns:
    fake_subjects = fake_df.groupBy("subject").count().orderBy(desc("count"))
    fake_subjects.show()
else:
    print("No 'subject' column in fake news dataset")

# %% [markdown]
# ### Data Leakage Visualization
# 
# The analysis above reveals a critical issue: the 'subject' column perfectly discriminates between real and fake news. This creates a data leakage problem that would invalidate our model.

# %%
# Create a combined dataset with labels
real_with_label = real_df.withColumn("label", lit(1))
fake_with_label = fake_df.withColumn("label", lit(0))
combined_df = real_with_label.union(fake_with_label)

# Check if subject column exists
if "subject" in combined_df.columns:
    # Get subject distribution by label
    subject_by_label = combined_df.groupBy("subject", "label").count().orderBy("subject", "label")
    subject_by_label.show()
    
    # Convert to pandas for visualization
    subject_label_pd = subject_by_label.toPandas()
    
    # Create a pivot table
    pivot_data = subject_label_pd.pivot(index='subject', columns='label', values='count').fillna(0)
    pivot_data.columns = ['Fake News', 'Real News']
    
    # Plot the distribution
    ax = pivot_data.plot(kind='bar', figsize=(10, 6), color=['#FF6B6B', '#4ECDC4'])
    plt.title('Subject Distribution by News Type')
    plt.xlabel('Subject')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add text labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    
    plt.tight_layout()
    plt.show()
else:
    print("No 'subject' column in the dataset")

# %% [markdown]
# ### Why We Must Drop the 'subject' Column
# 
# As demonstrated above, the 'subject' column creates a perfect data leakage problem:
# 
# 1. **Perfect Correlation with Labels**: 
#    - Real news articles are all labeled as "politicsNews"
#    - Fake news articles are all labeled as "News"
# 
# 2. **Model Impact**: If we include this column:
#    - The model would achieve nearly 100% accuracy in training and validation
#    - But this accuracy would be misleading and wouldn't generalize to real-world data
#    - The model would simply learn "if subject='politicsNews' then real, else fake"
# 
# 3. **Real-World Failure**: In production, new articles wouldn't follow this artificial pattern, causing the model to make incorrect predictions based on an irrelevant feature.
# 
# 4. **Scientific Integrity**: Including this column would invalidate any research findings since the model wouldn't be learning meaningful patterns in the text content.
# 
# Therefore, we must drop the 'subject' column from our preprocessing pipeline to ensure our model learns from the actual content of the news articles rather than this artificial distinction.

# %% [markdown]
# ### 4. Analyzing URL Presence

# %%
# Register the function as a UDF
contains_url_udf = udf(contains_url, BooleanType())

# Check URL presence in real news
real_with_url = real_df.withColumn("contains_url", contains_url_udf(col("text")))
real_url_count = real_with_url.filter(col("contains_url") == True).count()
real_total = real_df.count()
real_url_percentage = (real_url_count / real_total) * 100

# Check URL presence in fake news
fake_with_url = fake_df.withColumn("contains_url", contains_url_udf(col("text")))
fake_url_count = fake_with_url.filter(col("contains_url") == True).count()
fake_total = fake_df.count()
fake_url_percentage = (fake_url_count / fake_total) * 100

# Print results
print(f"Real news articles containing URLs: {real_url_count} out of {real_total} ({real_url_percentage:.2f}%)")
print(f"Fake news articles containing URLs: {fake_url_count} out of {fake_total} ({fake_url_percentage:.2f}%)")

# Plot the results
labels = ['Real News', 'Fake News']
url_percentages = [real_url_percentage, fake_url_percentage]
no_url_percentages = [100 - real_url_percentage, 100 - fake_url_percentage]

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = np.arange(len(labels))

ax.bar(x, url_percentages, width, label='Contains URLs', color='#FF6B6B')
ax.bar(x, no_url_percentages, width, bottom=url_percentages, label='No URLs', color='#4ECDC4')

ax.set_ylabel('Percentage')
ax.set_title('URL Presence in News Articles')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add percentage labels
for i, v in enumerate(url_percentages):
    ax.text(i, v/2, f"{v:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    ax.text(i, v + (no_url_percentages[i]/2), f"{no_url_percentages[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5. Analyzing Social Media Content

# %%
# Register the functions as UDFs
contains_handle_udf = udf(contains_twitter_handle, BooleanType())
contains_hashtag_udf = udf(contains_hashtag, BooleanType())

# Check Twitter handle presence
real_with_handle = real_df.withColumn("contains_handle", contains_handle_udf(col("text")))
real_handle_count = real_with_handle.filter(col("contains_handle") == True).count()
real_handle_percentage = (real_handle_count / real_total) * 100

fake_with_handle = fake_df.withColumn("contains_handle", contains_handle_udf(col("text")))
fake_handle_count = fake_with_handle.filter(col("contains_handle") == True).count()
fake_handle_percentage = (fake_handle_count / fake_total) * 100

# Check hashtag presence
real_with_hashtag = real_df.withColumn("contains_hashtag", contains_hashtag_udf(col("text")))
real_hashtag_count = real_with_hashtag.filter(col("contains_hashtag") == True).count()
real_hashtag_percentage = (real_hashtag_count / real_total) * 100

fake_with_hashtag = fake_df.withColumn("contains_hashtag", contains_hashtag_udf(col("text")))
fake_hashtag_count = fake_with_hashtag.filter(col("contains_hashtag") == True).count()
fake_hashtag_percentage = (fake_hashtag_count / fake_total) * 100

# Print results
print(f"Real news articles containing Twitter handles: {real_handle_count} out of {real_total} ({real_handle_percentage:.2f}%)")
print(f"Fake news articles containing Twitter handles: {fake_handle_count} out of {fake_total} ({fake_handle_percentage:.2f}%)")
print(f"\nReal news articles containing hashtags: {real_hashtag_count} out of {real_total} ({real_hashtag_percentage:.2f}%)")
print(f"Fake news articles containing hashtags: {fake_hashtag_count} out of {fake_total} ({fake_hashtag_percentage:.2f}%)")

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Twitter handles plot
handle_percentages = [real_handle_percentage, fake_handle_percentage]
no_handle_percentages = [100 - real_handle_percentage, 100 - fake_handle_percentage]

ax1.bar(x, handle_percentages, width, label='Contains Twitter Handles', color='#FF6B6B')
ax1.bar(x, no_handle_percentages, width, bottom=handle_percentages, label='No Twitter Handles', color='#4ECDC4')
ax1.set_ylabel('Percentage')
ax1.set_title('Twitter Handle Presence in News Articles')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

# Add percentage labels
for i, v in enumerate(handle_percentages):
    ax1.text(i, v/2, f"{v:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    ax1.text(i, v + (no_handle_percentages[i]/2), f"{no_handle_percentages[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')

# Hashtags plot
hashtag_percentages = [real_hashtag_percentage, fake_hashtag_percentage]
no_hashtag_percentages = [100 - real_hashtag_percentage, 100 - fake_hashtag_percentage]

ax2.bar(x, hashtag_percentages, width, label='Contains Hashtags', color='#FF6B6B')
ax2.bar(x, no_hashtag_percentages, width, bottom=hashtag_percentages, label='No Hashtags', color='#4ECDC4')
ax2.set_ylabel('Percentage')
ax2.set_title('Hashtag Presence in News Articles')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()

# Add percentage labels
for i, v in enumerate(hashtag_percentages):
    ax2.text(i, v/2, f"{v:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    ax2.text(i, v + (no_hashtag_percentages[i]/2), f"{no_hashtag_percentages[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6. Text Preprocessing

# %%
# Define preprocessing configuration
preprocessing_config = {
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

# Preprocess a sample text to demonstrate
sample_text = real_df.select("text").limit(1).collect()[0][0]
print("Original text:")
print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)

# Preprocess the text
preprocessed_text = preprocess_text(sample_text, preprocessing_config)
print("\nPreprocessed text:")
print(preprocessed_text)

# %% [markdown]
# ### 7. Apply Preprocessing to DataFrames

# %%
# Preprocess both datasets
real_preprocessed = preprocess_spark_df(real_df, "text", "title", combine_title_text=True, config=preprocessing_config)
fake_preprocessed = preprocess_spark_df(fake_df, "text", "title", combine_title_text=True, config=preprocessing_config)

# Display sample of preprocessed data
print("Sample of preprocessed real news:")
real_preprocessed.select("title", "text", "processed_text").limit(2).show(truncate=50)

print("\nSample of preprocessed fake news:")
fake_preprocessed.select("title", "text", "processed_text").limit(2).show(truncate=50)

# %% [markdown]
# ### 8. Tokenize Text

# %%
# Tokenize preprocessed text
real_tokenized = tokenize_spark_df(real_preprocessed, "processed_text", config=preprocessing_config)
fake_tokenized = tokenize_spark_df(fake_preprocessed, "processed_text", config=preprocessing_config)

# Display sample of tokenized data
print("Sample of tokenized real news:")
real_tokenized.select("processed_text", "tokens").limit(2).show(truncate=50)

print("\nSample of tokenized fake news:")
fake_tokenized.select("processed_text", "tokens").limit(2).show(truncate=50)

# %% [markdown]
# ### 9. Extract Quoted Content

# %%
# Extract quoted content
real_with_quotes = extract_quoted_content_spark_df(real_tokenized, "text")
fake_with_quotes = extract_quoted_content_spark_df(fake_tokenized, "text")

# Display sample of data with quoted content
print("Sample of real news with quoted content:")
real_with_quotes.select("text", "quoted_content").limit(2).show(truncate=50)

print("\nSample of fake news with quoted content:")
fake_with_quotes.select("text", "quoted_content").limit(2).show(truncate=50)

# %% [markdown]
# ### 10. Complete Preprocessing Pipeline

# %%
# Run the complete preprocessing pipeline
results = preprocess_and_save_data(
    output_dir="dbfs:/FileStore/fake_news_detection/preprocessed_data",
    fake_table_name="fake", 
    true_table_name="real",
    create_tables=True
)

# Display sample of final preprocessed data
print("Sample of final preprocessed data:")
results["final_df"].select("label", "processed_text", "tokens").limit(3).show(truncate=50)

# %% [markdown]
# ## Important Notes
# 
# 1. **Data Leakage**: We identified and removed the 'subject' column to prevent data leakage, as it perfectly discriminates between real and fake news in our dataset.
# 
# 2. **Text Preprocessing**: Our preprocessing pipeline includes multiple steps:
#    - Removing URLs, Twitter handles, and special characters
#    - Converting text to lowercase
#    - Removing punctuation
#    - Removing stopwords
#    - Lemmatization
#    - Tokenization
# 
# 3. **Social Media Content**: We found significant differences in the presence of URLs, Twitter handles, and hashtags between real and fake news, which can be useful features for classification.
# 
# 4. **Date Processing**: If date information is available, we standardize it and extract useful features like year, month, day, and day of week.
# 
# 5. **Quoted Content**: We extract content within quotation marks, which can be useful for analyzing quoted sources in news articles.
# 
# 6. **Configuration**: All preprocessing steps are configurable through a dictionary, allowing for easy experimentation with different preprocessing strategies.
# 
# 7. **Spark Optimization**: The code is optimized for Spark's distributed processing capabilities, making it suitable for large datasets.
