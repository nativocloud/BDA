"""
Date Utilities for Fake News Detection

This module provides utilities for validating, standardizing, and extracting features from date fields
in the fake news dataset. It handles various date formats, null values, and incorrect formats,
converting them to a standardized format and extracting useful temporal features.

The implementation uses Spark's distributed processing capabilities to ensure scalability.

Last updated: May 29, 2025
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, to_date, year, month, dayofmonth, dayofweek, 
    date_format, when, lit, regexp_replace, udf
)
from pyspark.sql.types import StringType, DateType, IntegerType
from datetime import datetime
import re
from typing import List, Optional, Tuple

class DateProcessor:
    """
    A class for processing date fields in a Spark DataFrame.
    
    This class provides methods for validating, standardizing, and extracting features
    from date fields, handling various formats and edge cases.
    """
    
    def __init__(self, date_column: str = "publish_date", expected_format: str = "MMMM d, yyyy"):
        """
        Initialize the DateProcessor.
        
        Args:
            date_column (str): The name of the date column to process.
            expected_format (str): The expected format of the date column (Spark date format).
        """
        self.date_column = date_column
        self.expected_format = expected_format
        
        # Common date formats to try when parsing dates
        self.date_formats = [
            "MMMM d, yyyy",       # December 25, 2017
            "MMM d, yyyy",        # Dec 25, 2017
            "yyyy-MM-dd",         # 2017-12-25
            "MM/dd/yyyy",         # 12/25/2017
            "dd/MM/yyyy",         # 25/12/2017
            "yyyy/MM/dd",         # 2017/12/25
            "MM-dd-yyyy",         # 12-25-2017
            "dd-MM-yyyy"          # 25-12-2017
        ]
    
    def validate_date_format(self, df: DataFrame) -> DataFrame:
        """
        Validate the date format in the DataFrame.
        
        This method checks if the date column values match the expected format
        and adds a validation flag column.
        
        Args:
            df (DataFrame): The input DataFrame with a date column.
            
        Returns:
            DataFrame: The DataFrame with an additional validation flag column.
        """
        # Create a validation flag column
        return df.withColumn(
            "date_valid",
            when(
                col(self.date_column).isNull(), 
                lit(False)
            ).otherwise(
                to_date(col(self.date_column), self.expected_format).isNotNull()
            )
        )
    
    def standardize_date(self, df: DataFrame) -> DataFrame:
        """
        Standardize the date format in the DataFrame.
        
        This method attempts to parse the date column using various common formats
        and converts it to a standard date type.
        
        Args:
            df (DataFrame): The input DataFrame with a date column.
            
        Returns:
            DataFrame: The DataFrame with a standardized date column.
        """
        # Start with the original DataFrame
        result_df = df
        
        # Try to parse the date using each format
        for date_format in self.date_formats:
            result_df = result_df.withColumn(
                "std_date",
                when(
                    col("std_date").isNull(),
                    to_date(col(self.date_column), date_format)
                ).otherwise(col("std_date"))
            )
        
        # For any remaining null values, use a more complex approach with UDFs
        # This is a fallback for unusual formats
        parse_complex_date_udf = udf(self._parse_complex_date, DateType())
        
        result_df = result_df.withColumn(
            "std_date",
            when(
                col("std_date").isNull() & col(self.date_column).isNotNull(),
                parse_complex_date_udf(col(self.date_column))
            ).otherwise(col("std_date"))
        )
        
        return result_df
    
    def extract_date_features(self, df: DataFrame) -> DataFrame:
        """
        Extract useful date features from the standardized date column.
        
        This method extracts year, month, day, day of week, and YYYYMMDD format
        from the standardized date column.
        
        Args:
            df (DataFrame): The input DataFrame with a standardized date column.
            
        Returns:
            DataFrame: The DataFrame with additional date feature columns.
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
        
        return result_df
    
    def process_date_column(self, df: DataFrame) -> DataFrame:
        """
        Process the date column in the DataFrame.
        
        This method combines validation, standardization, and feature extraction
        into a single pipeline.
        
        Args:
            df (DataFrame): The input DataFrame with a date column.
            
        Returns:
            DataFrame: The DataFrame with processed date columns.
        """
        # Initialize the standardized date column as null
        result_df = df.withColumn("std_date", lit(None).cast(DateType()))
        
        # Standardize the date
        result_df = self.standardize_date(result_df)
        
        # Extract date features
        result_df = self.extract_date_features(result_df)
        
        # Add validation flag
        result_df = result_df.withColumn(
            "date_valid",
            col("std_date").isNotNull()
        )
        
        return result_df
    
    def _parse_complex_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse complex date formats that aren't handled by standard Spark functions.
        
        This is a Python UDF that attempts to parse dates using various approaches.
        
        Args:
            date_str (str): The date string to parse.
            
        Returns:
            datetime: The parsed date as a datetime object, or None if parsing fails.
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


def process_dates(df: DataFrame, date_column: str = "publish_date") -> DataFrame:
    """
    Process date fields in a DataFrame.
    
    This function validates, standardizes, and extracts features from date fields
    in the input DataFrame.
    
    Args:
        df (DataFrame): The input DataFrame with a date column.
        date_column (str): The name of the date column to process.
        
    Returns:
        DataFrame: The DataFrame with processed date columns.
    """
    processor = DateProcessor(date_column=date_column)
    return processor.process_date_column(df)


# Example usage:
# from pyspark.sql import SparkSession
# 
# # Create a Spark session
# spark = SparkSession.builder.appName("DateProcessing").getOrCreate()
# 
# # Sample data with various date formats
# data = [
#     (1, "December 25, 2017"),
#     (2, "Dec 25, 2017"),
#     (3, "2017-12-25"),
#     (4, "12/25/2017"),
#     (5, "25/12/2017"),
#     (6, "2017/12/25"),
#     (7, "12-25-2017"),
#     (8, "25-12-2017"),
#     (9, "25th December 2017"),
#     (10, "December 2017"),
#     (11, None),
#     (12, "Invalid date")
# ]
# 
# # Create a DataFrame
# df = spark.createDataFrame(data, ["id", "publish_date"])
# 
# # Process the date column
# processed_df = process_dates(df)
# 
# # Show the results
# processed_df.select("id", "publish_date", "std_date", "year", "month", "day", "day_of_week", "date_yyyymmdd", "date_valid").show()

# Last modified: May 29, 2025
