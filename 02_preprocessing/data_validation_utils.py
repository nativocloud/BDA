"""
Data Validation Utilities for Fake News Detection

This module provides utilities for validating and cleaning all columns in the fake news dataset.
It handles null values, blank fields, and malformed data, ensuring high data quality for downstream analysis.

The implementation uses Spark's distributed processing capabilities to ensure scalability.

Last updated: May 29, 2025
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, length, trim, when, lit, regexp_replace, udf, 
    lower, upper, initcap, isnan, count, sum, avg
)
from pyspark.sql.types import StringType, BooleanType, IntegerType, ArrayType
import re
from typing import List, Dict, Any, Optional, Tuple

class DataValidator:
    """
    A class for validating and cleaning data in a Spark DataFrame.
    
    This class provides methods for checking data quality, standardizing formats,
    and handling problematic values across all columns in the dataset.
    """
    
    def __init__(self, min_title_length: int = 5, min_text_length: int = 50):
        """
        Initialize the DataValidator.
        
        Args:
            min_title_length (int): Minimum acceptable length for the title field.
            min_text_length (int): Minimum acceptable length for the text field.
        """
        self.min_title_length = min_title_length
        self.min_text_length = min_text_length
        
        # Common patterns for problematic content
        self.suspicious_patterns = [
            r'<\s*script.*?>.*?<\s*/\s*script\s*>', # Script tags
            r'<\s*style.*?>.*?<\s*/\s*style\s*>',   # Style tags
            r'<\s*iframe.*?>.*?<\s*/\s*iframe\s*>', # iFrame tags
            r'javascript:',                         # JavaScript protocol
            r'data:',                               # Data URI scheme
            r'[\u0080-\u00ff]{10,}',                # Long sequences of non-ASCII chars
            r'(\w)\1{10,}'                          # Repeated characters (e.g., "aaaaaaaaaa")
        ]
    
    def validate_text_fields(self, df: DataFrame, text_columns: List[str]) -> DataFrame:
        """
        Validate text fields in the DataFrame.
        
        This method checks for null values, minimum length, and suspicious patterns
        in text fields, adding validation flag columns.
        
        Args:
            df (DataFrame): The input DataFrame.
            text_columns (List[str]): List of text column names to validate.
            
        Returns:
            DataFrame: The DataFrame with additional validation flag columns.
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
            min_length = self.min_title_length if "title" in column else self.min_text_length
            result_df = result_df.withColumn(
                f"{column}_length_valid",
                when(
                    col(f"{column}_present") & (length(trim(col(column))) >= min_length),
                    lit(True)
                ).otherwise(lit(False))
            )
            
            # Check for suspicious patterns
            has_suspicious_pattern_udf = udf(self._has_suspicious_pattern, BooleanType())
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
    
    def standardize_text_fields(self, df: DataFrame, text_columns: List[str]) -> DataFrame:
        """
        Standardize text fields in the DataFrame.
        
        This method cleans and standardizes text fields by removing extra whitespace,
        normalizing case, and removing problematic characters.
        
        Args:
            df (DataFrame): The input DataFrame.
            text_columns (List[str]): List of text column names to standardize.
            
        Returns:
            DataFrame: The DataFrame with standardized text columns.
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
                    initcap(col(std_column))
                )
            elif "author" in column or "source" in column:
                # For author and source, use title case
                result_df = result_df.withColumn(
                    std_column,
                    initcap(col(std_column))
                )
            else:
                # For other text, leave as is
                pass
        
        return result_df
    
    def validate_categorical_fields(self, df: DataFrame, categorical_columns: List[str]) -> DataFrame:
        """
        Validate categorical fields in the DataFrame.
        
        This method checks for null values and standardizes categorical values,
        adding validation flag columns.
        
        Args:
            df (DataFrame): The input DataFrame.
            categorical_columns (List[str]): List of categorical column names to validate.
            
        Returns:
            DataFrame: The DataFrame with additional validation flag columns.
        """
        result_df = df
        
        for column in categorical_columns:
            # Check for null or empty values
            result_df = result_df.withColumn(
                f"{column}_present", 
                when(
                    (col(column).isNull()) | (trim(col(column)) == ""),
                    lit(False)
                ).otherwise(lit(True))
            )
            
            # Create standardized version of the column
            std_column = f"std_{column}"
            result_df = result_df.withColumn(
                std_column,
                when(
                    col(column).isNull(),
                    None
                ).otherwise(
                    trim(initcap(col(column)))
                )
            )
            
            # Overall validation flag for this column
            result_df = result_df.withColumn(
                f"{column}_valid",
                col(f"{column}_present")
            )
        
        return result_df
    
    def calculate_data_quality_metrics(self, df: DataFrame) -> Dict[str, Any]:
        """
        Calculate data quality metrics for the DataFrame.
        
        This method computes various metrics to assess the overall quality of the dataset,
        including completeness, validity, and consistency.
        
        Args:
            df (DataFrame): The input DataFrame with validation columns.
            
        Returns:
            Dict[str, Any]: A dictionary of data quality metrics.
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
    
    def filter_invalid_records(self, df: DataFrame, required_columns: List[str]) -> DataFrame:
        """
        Filter out invalid records from the DataFrame.
        
        This method removes records that have invalid values in required columns.
        
        Args:
            df (DataFrame): The input DataFrame with validation columns.
            required_columns (List[str]): List of column names that must be valid.
            
        Returns:
            DataFrame: The filtered DataFrame with only valid records.
        """
        result_df = df
        
        for column in required_columns:
            if f"{column}_valid" in df.columns:
                result_df = result_df.filter(col(f"{column}_valid") == True)
        
        return result_df
    
    def process_dataframe(self, df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Process the entire DataFrame for data quality.
        
        This method combines validation, standardization, and filtering
        into a single pipeline.
        
        Args:
            df (DataFrame): The input DataFrame.
            
        Returns:
            Tuple[DataFrame, Dict[str, Any]]: The processed DataFrame and data quality metrics.
        """
        # Define column groups
        text_columns = ["title", "text"]
        categorical_columns = ["author", "source"]
        
        # Validate text fields
        result_df = self.validate_text_fields(df, text_columns)
        
        # Standardize text fields
        result_df = self.standardize_text_fields(result_df, text_columns + categorical_columns)
        
        # Validate categorical fields
        result_df = self.validate_categorical_fields(result_df, categorical_columns)
        
        # Calculate data quality metrics
        metrics = self.calculate_data_quality_metrics(result_df)
        
        # Add overall record validity flag
        result_df = result_df.withColumn(
            "record_valid",
            col("title_valid") & col("text_valid")
        )
        
        return result_df, metrics
    
    def _has_suspicious_pattern(self, text: Optional[str]) -> bool:
        """
        Check if text contains suspicious patterns.
        
        This is a Python UDF that checks for potentially problematic content.
        
        Args:
            text (str): The text to check.
            
        Returns:
            bool: True if suspicious patterns are found, False otherwise.
        """
        if not text:
            return False
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return True
        
        return False


def validate_and_clean_data(df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Validate and clean all columns in a DataFrame.
    
    This function checks for data quality issues, standardizes formats,
    and handles problematic values across all columns.
    
    Args:
        df (DataFrame): The input DataFrame.
        
    Returns:
        Tuple[DataFrame, Dict[str, Any]]: The processed DataFrame and data quality metrics.
    """
    validator = DataValidator()
    return validator.process_dataframe(df)


# Example usage:
# from pyspark.sql import SparkSession
# 
# # Create a Spark session
# spark = SparkSession.builder.appName("DataValidation").getOrCreate()
# 
# # Sample data with various issues
# data = [
#     (1, "Valid Title", "This is a valid text content with sufficient length for analysis.", "John Doe", "Reliable Source"),
#     (2, "", "Text without a title", "Jane Smith", "Another Source"),
#     (3, "Title without text", "", "Unknown", ""),
#     (4, "Title with suspicious content", "<script>alert('XSS')</script>", "Hacker", "Suspicious Source"),
#     (5, "Very short", "Too short", "", "Source"),
#     (6, None, None, None, None)
# ]
# 
# # Create a DataFrame
# df = spark.createDataFrame(data, ["id", "title", "text", "author", "source"])
# 
# # Validate and clean the data
# processed_df, metrics = validate_and_clean_data(df)
# 
# # Show the results
# processed_df.select("id", "title", "std_title", "title_valid", "text", "std_text", "text_valid", "record_valid").show(truncate=False)
# 
# # Print data quality metrics
# print("Data Quality Metrics:")
# print(f"Total rows: {metrics['total_rows']}")
# print(f"Overall quality score: {metrics['overall_quality']:.2f}")
# print("\nCompleteness:")
# for column, score in metrics['completeness'].items():
#     print(f"  {column}: {score:.2f}")
# print("\nValidity:")
# for column, score in metrics['validity'].items():
#     print(f"  {column}: {score:.2f}")
