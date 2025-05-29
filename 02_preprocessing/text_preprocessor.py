"""
Text Preprocessor for Fake News Detection

This module provides text preprocessing functionality for the fake news detection pipeline.
It includes functions for text cleaning, normalization, tokenization, and other NLP preprocessing steps.

Author: BDA Team
Date: May 27, 2025
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
from pyspark.sql.functions import udf, col
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


class TextPreprocessor:
    """
    A class for preprocessing text data in the fake news detection pipeline.
    """
    
    def __init__(self, config=None):
        """
        Initialize the TextPreprocessor with optional configuration.
        
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
        
        # Remove URLs
        if self.config['remove_urls']:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
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
    
    def preprocess_pandas_df(self, df, text_column):
        """
        Preprocess text in a pandas DataFrame.
        
        Args:
            df (pandas.DataFrame): The input DataFrame
            text_column (str): The name of the column containing text to preprocess
            
        Returns:
            pandas.DataFrame: DataFrame with preprocessed text in a new column named 'processed_text'
        """
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        return df
    
    def preprocess_spark_df(self, spark_df, text_column):
        """
        Preprocess text in a Spark DataFrame.
        
        Args:
            spark_df (pyspark.sql.DataFrame): The input Spark DataFrame
            text_column (str): The name of the column containing text to preprocess
            
        Returns:
            pyspark.sql.DataFrame: DataFrame with preprocessed text in a new column named 'processed_text'
        """
        # Register UDF for preprocessing
        preprocess_udf = udf(self.preprocess_text, StringType())
        
        # Apply preprocessing to the specified column
        processed_df = spark_df.withColumn('processed_text', preprocess_udf(col(text_column)))
        
        return processed_df
    
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
        
        # Apply preprocessing steps that should happen before tokenization
        if self.config['lowercase']:
            text = text.lower()
            
        if self.config['remove_urls']:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Apply additional filtering steps
        if self.config['remove_stopwords']:
            tokens = [word for word in tokens if word not in self.stop_words]
            
        if self.config['remove_punctuation']:
            tokens = [word for word in tokens if word not in string.punctuation]
        
        if self.config['stemming']:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        if self.config['lemmatization']:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
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


def create_spark_preprocessor(spark_session=None):
    """
    Create a TextPreprocessor instance and configure it for use with Spark.
    
    Args:
        spark_session (pyspark.sql.SparkSession, optional): An existing SparkSession.
            If None, a new session will be created.
            
    Returns:
        tuple: (SparkSession, TextPreprocessor) - The SparkSession and configured TextPreprocessor
    """
    # Create SparkSession if not provided
    if spark_session is None:
        spark_session = SparkSession.builder \
            .appName("FakeNewsDetection-TextPreprocessing") \
            .getOrCreate()
    
    # Create preprocessor with default configuration
    preprocessor = TextPreprocessor()
    
    return spark_session, preprocessor


def main():
    """
    Main function to demonstrate text preprocessing functionality.
    """
    # Create Spark session and preprocessor
    spark, preprocessor = create_spark_preprocessor()
    
    # Sample text for demonstration
    sample_texts = [
        "This is a sample news article from CNN.com about politics in 2023!",
        "BREAKING NEWS: Scientists discover new vaccine for COVID-19 http://example.com",
        "The stock market fell by 5% yesterday, according to financial experts."
    ]
    
    # Create a sample DataFrame
    sample_df = spark.createDataFrame([(i, text) for i, text in enumerate(sample_texts)], 
                                     ["id", "text"])
    
    # Preprocess the text
    processed_df = preprocessor.preprocess_spark_df(sample_df, "text")
    
    # Tokenize the text
    tokenized_df = preprocessor.tokenize_spark_df(sample_df, "text")
    
    # Show results
    print("Original and Preprocessed Text:")
    processed_df.select("id", "text", "processed_text").show(truncate=False)
    
    print("\nTokenized Text:")
    tokenized_df.select("id", "text", "tokens").show(truncate=False)
    
    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    main()

# Last modified: May 29, 2025
