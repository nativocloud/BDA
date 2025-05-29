"""
Enhanced Text Preprocessor for Fake News Detection

This module provides advanced text preprocessing functionality for the fake news detection pipeline,
specifically tailored to the characteristics of the True.csv and Fake.csv datasets.
It includes functions for text cleaning, normalization, tokenization, and other NLP preprocessing steps.

Author: BDA Team
Date: May 28, 2025
Last updated: May 29, 2025
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, trim, length
from pyspark.sql.types import StringType, ArrayType

# Download required NLTK resources if not already present
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

# Explicitly download punkt_tab if used by any NLTK dependency
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        # If punkt_tab doesn't exist, use standard punkt instead
        print("Warning: punkt_tab resource not found in NLTK. Using standard punkt tokenizer instead.")
        # No action needed as punkt is already downloaded above


class EnhancedTextPreprocessor:
    """
    An enhanced class for preprocessing text data in the fake news detection pipeline,
    specifically tailored to the characteristics of the True.csv and Fake.csv datasets.
    """
    
    def __init__(self, config=None):
        """
        Initialize the EnhancedTextPreprocessor with optional configuration.
        
        Args:
            config (dict, optional): Configuration parameters for preprocessing.
                Supported keys:
                - remove_stopwords (bool): Whether to remove stopwords. Default: True
                - stemming (bool): Whether to apply stemming. Default: False
                - lemmatization (bool): Whether to apply lemmatization. Default: True
                - lowercase (bool): Whether to convert text to lowercase. Default: True
                - remove_punctuation (bool): Whether to remove punctuation. Default: True
                - remove_numbers (bool): Whether to remove numbers. Default: False
                - remove_urls (bool): Whether to remove URLs. Default: True
                - remove_twitter_handles (bool): Whether to remove Twitter handles. Default: True
                - remove_hashtags (bool): Whether to remove hashtags. Default: False
                - convert_hashtags (bool): Whether to convert hashtags to plain text. Default: True
                - remove_special_chars (bool): Whether to remove special characters. Default: True
                - normalize_whitespace (bool): Whether to normalize whitespace. Default: True
                - max_text_length (int): Maximum text length in words. Default: 500 (0 means no limit)
                - language (str): Language for stopwords. Default: 'english'
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
        self.config = config if config else default_config
        
        # Initialize tools based on configuration
        if self.config['remove_stopwords']:
            self.stop_words = set(stopwords.words(self.config['language']))
        
        if self.config['stemming']:
            self.stemmer = PorterStemmer()
            
        if self.config['lemmatization']:
            self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """
        Apply all configured preprocessing steps to a text string.
        
        Args:
            text (str): The input text to preprocess
            
        Returns:
            str: The preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize whitespace
        if self.config['normalize_whitespace']:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs (enhanced pattern for Twitter URLs)
        if self.config['remove_urls']:
            text = re.sub(r'https?://t\.co/\w+|http\S+|www\S+|https\S+', '', text)
        
        # Remove Twitter handles
        if self.config['remove_twitter_handles']:
            text = re.sub(r'@\w+', '', text)
        
        # Handle hashtags
        if self.config['remove_hashtags']:
            text = re.sub(r'#\w+', '', text)
        elif self.config['convert_hashtags']:
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters and emojis
        if self.config['remove_special_chars']:
            text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Convert to lowercase
        if self.config['lowercase']:
            text = text.lower()
        
        # Remove punctuation
        if self.config['remove_punctuation']:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.config['remove_numbers']:
            text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Limit text length if configured
        if self.config['max_text_length'] > 0 and len(tokens) > self.config['max_text_length']:
            tokens = tokens[:self.config['max_text_length']]
        
        # Remove stopwords
        if self.config['remove_stopwords']:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Apply stemming
        if self.config['stemming']:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        # Apply lemmatization
        if self.config['lemmatization']:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into text
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_pandas_df(self, df, text_column, title_column=None, combine_title_text=False):
        """
        Preprocess text in a pandas DataFrame.
        
        Args:
            df (pandas.DataFrame): The input DataFrame
            text_column (str): The name of the column containing text to preprocess
            title_column (str, optional): The name of the column containing title text
            combine_title_text (bool): Whether to combine title and text for preprocessing
            
        Returns:
            pandas.DataFrame: DataFrame with preprocessed text in a new column named 'processed_text'
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Trim whitespace from all string columns
        for col in result_df.select_dtypes(include=['object']).columns:
            result_df[col] = result_df[col].str.strip()
        
        # Drop the subject column if it exists (with explanation in the notebook)
        if 'subject' in result_df.columns:
            result_df = result_df.drop('subject', axis=1)
        
        # Combine title and text if requested
        if combine_title_text and title_column and title_column in result_df.columns:
            result_df['combined_text'] = result_df[title_column].fillna('') + ' ' + result_df[text_column].fillna('')
            result_df['processed_text'] = result_df['combined_text'].apply(self.preprocess_text)
        else:
            result_df['processed_text'] = result_df[text_column].apply(self.preprocess_text)
        
        return result_df
    
    def preprocess_spark_df(self, spark_df, text_column, title_column=None, combine_title_text=False):
        """
        Preprocess text in a Spark DataFrame.
        
        Args:
            spark_df (pyspark.sql.DataFrame): The input Spark DataFrame
            text_column (str): The name of the column containing text to preprocess
            title_column (str, optional): The name of the column containing title text
            combine_title_text (bool): Whether to combine title and text for preprocessing
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with preprocessed text in a new column named 'processed_text'
        """
        # Register UDF for preprocessing
        preprocess_udf = udf(self.preprocess_text, StringType())
        
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
                col(title_column).cast(StringType()).fillna('') + ' ' + col(text_column).cast(StringType()).fillna('')
            )
            result_df = result_df.withColumn('processed_text', preprocess_udf(col('combined_text')))
        else:
            result_df = result_df.withColumn('processed_text', preprocess_udf(col(text_column)))
        
        return result_df
    
    def tokenize_text(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            list: List of tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # First apply preprocessing to get clean text
        clean_text = self.preprocess_text(text)
        
        # Tokenize the clean text
        tokens = word_tokenize(clean_text)
        
        return tokens
    
    def tokenize_spark_df(self, spark_df, text_column):
        """
        Tokenize text in a Spark DataFrame.
        
        Args:
            spark_df (pyspark.sql.DataFrame): The input Spark DataFrame
            text_column (str): The name of the column containing text to tokenize
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with tokenized text in a new column named 'tokens'
        """
        # Register UDF for tokenization
        tokenize_udf = udf(self.tokenize_text, ArrayType(StringType()))
        
        # Apply tokenization to the specified column
        tokenized_df = spark_df.withColumn('tokens', tokenize_udf(col(text_column)))
        
        return tokenized_df
    
    def extract_quoted_content(self, text):
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
    
    def extract_quoted_content_spark_df(self, spark_df, text_column):
        """
        Extract quoted content from text in a Spark DataFrame.
        
        Args:
            spark_df (pyspark.sql.DataFrame): The input Spark DataFrame
            text_column (str): The name of the column containing text
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with quoted content in a new column named 'quoted_content'
        """
        # Register UDF for extracting quoted content
        extract_quoted_udf = udf(self.extract_quoted_content, ArrayType(StringType()))
        
        # Apply extraction to the specified column
        result_df = spark_df.withColumn('quoted_content', extract_quoted_udf(col(text_column)))
        
        return result_df


def create_spark_preprocessor(spark_session=None):
    """
    Create an EnhancedTextPreprocessor instance and configure it for use with Spark.
    
    Args:
        spark_session (pyspark.sql.SparkSession, optional): An existing SparkSession.
            If None, a new session will be created.
            
    Returns:
        tuple: (SparkSession, EnhancedTextPreprocessor) - The SparkSession and configured EnhancedTextPreprocessor
    """
    # Create SparkSession if not provided
    if spark_session is None:
        spark_session = SparkSession.builder \
            .appName("FakeNewsDetection-TextPreprocessing") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.driver.memory", "8g") \
            .getOrCreate()
    
    # Create preprocessor with default configuration
    preprocessor = EnhancedTextPreprocessor()
    
    return spark_session, preprocessor


def analyze_dataset_characteristics(spark, file_path, sample_size=1000):
    """
    Analyze dataset characteristics to inform preprocessing decisions.
    
    Args:
        spark (pyspark.sql.SparkSession): The Spark session
        file_path (str): Path to the CSV file
        sample_size (int): Number of rows to sample
        
    Returns:
        dict: Dictionary of dataset characteristics
    """
    # Read the CSV file
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
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
        text_length_udf = udf(lambda text: len(text.split()) if text else 0, StringType())
        length_df = sample_df.withColumn("text_length", text_length_udf(col("text")))
        stats["avg_text_length"] = length_df.agg({"text_length": "avg"}).collect()[0][0]
    
    return stats


def main():
    """
    Main function to demonstrate enhanced text preprocessing functionality.
    """
    # Create Spark session and preprocessor
    spark, preprocessor = create_spark_preprocessor()
    
    # Sample text for demonstration
    sample_texts = [
        "This is a sample text with @twitter_handle and #hashtag https://t.co/abcd1234",
        "Another example with special characters: é, ñ, ü and numbers 12345",
        "\"This is quoted content\" that should be extracted separately."
    ]
    
    # Process sample texts
    for i, text in enumerate(sample_texts):
        print(f"\nSample {i+1}:")
        print(f"Original: {text}")
        print(f"Preprocessed: {preprocessor.preprocess_text(text)}")
        print(f"Tokens: {preprocessor.tokenize_text(text)}")
        print(f"Quoted content: {preprocessor.extract_quoted_content(text)}")
    
    # Clean up
    spark.stop()


if __name__ == "__main__":
    main()
