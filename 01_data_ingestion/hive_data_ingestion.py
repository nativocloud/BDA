"""
Hive Data Ingestion for Fake News Detection

This module provides functions for loading data from Hive metastore tables
for the fake news detection pipeline. It leverages Spark SQL for efficient
distributed data processing directly from the Hive catalog.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, col, when, count, desc, rand
from typing import Tuple, Dict, Any

class HiveDataIngestion:
    """
    A class for ingesting data from Hive metastore tables for fake news detection.
    
    This class provides methods for loading, combining, and sampling news data
    from Hive tables using Apache Spark for distributed processing.
    """
    
    def __init__(self, spark: SparkSession, real_table: str = "real", fake_table: str = "fake"):
        """
        Initialize the HiveDataIngestion.
        
        Args:
            spark (SparkSession): The Spark session to use.
            real_table (str): The name of the Hive table containing real news.
            fake_table (str): The name of the Hive table containing fake news.
        """
        self.spark = spark
        self.real_table = real_table
        self.fake_table = fake_table
    
    def load_data_from_hive(self) -> Tuple[DataFrame, DataFrame]:
        """
        Load data from Hive metastore tables.
        
        This method loads real and fake news data from Hive tables using Spark SQL.
        
        Returns:
            Tuple[DataFrame, DataFrame]: DataFrames containing real and fake news.
        """
        # Load real news data from Hive table
        real_df = self.spark.table(self.real_table)
        
        # Load fake news data from Hive table
        fake_df = self.spark.table(self.fake_table)
        
        # Print dataset information
        print(f"Real news dataset loaded: {real_df.count()} records")
        print(f"Fake news dataset loaded: {fake_df.count()} records")
        
        # Register DataFrames as temporary views for SQL access
        real_df.createOrReplaceTempView("true_news")
        fake_df.createOrReplaceTempView("fake_news")
        
        # Examine schema of both datasets
        print("\nReal news schema:")
        real_df.printSchema()
        
        print("\nFake news schema:")
        fake_df.printSchema()
        
        return real_df, fake_df
    
    def combine_datasets(self, real_df: DataFrame, fake_df: DataFrame) -> DataFrame:
        """
        Combine real and fake news datasets with labels.
        
        This method adds labels to the datasets (1 for real news, 0 for fake news)
        and combines them into a single dataset.
        
        Args:
            real_df (DataFrame): DataFrame containing real news.
            fake_df (DataFrame): DataFrame containing fake news.
            
        Returns:
            DataFrame: Combined dataset with labels.
        """
        # Add labels to datasets
        real_df_with_label = real_df.withColumn("label", lit(1))  # 1 for real news
        fake_df_with_label = fake_df.withColumn("label", lit(0))  # 0 for fake news
        
        # Combine datasets using unionByName
        combined_df = real_df_with_label.unionByName(fake_df_with_label)
        
        # Register the combined DataFrame as a temporary view
        combined_df.createOrReplaceTempView("combined_news")
        
        # Display combined dataset statistics
        print("\nCombined dataset statistics:")
        self.spark.sql("""
            SELECT 
                label, 
                COUNT(*) as count,
                COUNT(DISTINCT subject) as unique_subjects
            FROM combined_news
            GROUP BY label
            ORDER BY label DESC
        """).show()
        
        return combined_df
    
    def create_balanced_sample(self, combined_df: DataFrame, sample_size: int = 100) -> DataFrame:
        """
        Create a balanced sample dataset with equal numbers of real and fake news articles.
        
        Args:
            combined_df (DataFrame): Combined dataset with labels.
            sample_size (int): Number of articles to sample from each class.
            
        Returns:
            DataFrame: Balanced sample dataset.
        """
        # Sample real news articles (label=1)
        real_sample = combined_df.filter(col("label") == 1) \
                                .orderBy(rand()) \
                                .limit(sample_size)
        
        # Sample fake news articles (label=0)
        fake_sample = combined_df.filter(col("label") == 0) \
                                .orderBy(rand()) \
                                .limit(sample_size)
        
        # Combine the samples
        sample_df = real_sample.unionByName(fake_sample)
        
        # Register the sample DataFrame as a temporary view
        sample_df.createOrReplaceTempView("sample_news")
        
        # Display sample dataset statistics
        print("\nSample dataset statistics:")
        self.spark.sql("""
            SELECT 
                label, 
                COUNT(*) as count,
                COUNT(DISTINCT subject) as unique_subjects
            FROM sample_news
            GROUP BY label
            ORDER BY label DESC
        """).show()
        
        return sample_df
    
    def save_to_parquet(self, df: DataFrame, path: str, partition_by: str = None) -> None:
        """
        Save a DataFrame to Parquet format.
        
        Args:
            df (DataFrame): The DataFrame to save.
            path (str): The path where to save the DataFrame.
            partition_by (str): Column to partition by (optional).
        """
        writer = df.write.mode("overwrite")
        
        if partition_by:
            writer = writer.partitionBy(partition_by)
        
        writer.parquet(path)
        print(f"DataFrame saved to {path}")
    
    def save_to_hive_table(self, df: DataFrame, table_name: str, partition_by: str = None) -> None:
        """
        Save a DataFrame to a Hive table.
        
        Args:
            df (DataFrame): The DataFrame to save.
            table_name (str): The name of the Hive table to create or overwrite.
            partition_by (str): Column to partition by (optional).
        """
        writer = df.write.mode("overwrite").format("parquet")
        
        if partition_by:
            writer = writer.partitionBy(partition_by)
        
        writer.saveAsTable(table_name)
        print(f"DataFrame saved to Hive table: {table_name}")
    
    def process_and_save_data(self, output_dir: str = "dbfs:/FileStore/fake_news_detection/data") -> Dict[str, Any]:
        """
        Process and save data from Hive tables.
        
        This method loads data from Hive tables, combines datasets, creates samples,
        and saves the results in Parquet format and as Hive tables.
        
        Args:
            output_dir (str): Directory to save processed data.
            
        Returns:
            Dict[str, Any]: Dictionary with references to the processed DataFrames.
        """
        # Load data from Hive
        real_df, fake_df = self.load_data_from_hive()
        
        # Combine datasets
        combined_df = self.combine_datasets(real_df, fake_df)
        
        # Check for subject column and drop it to prevent data leakage
        if "subject" in combined_df.columns:
            print("\nWARNING: Dropping 'subject' column to prevent data leakage")
            print("The 'subject' column perfectly discriminates between real and fake news")
            print("Real news: subject='politicsNews', Fake news: subject='News'")
            combined_df = combined_df.drop("subject")
            print("'subject' column dropped successfully")
        
        # Create balanced sample
        sample_df = self.create_balanced_sample(combined_df)
        
        # Save combined dataset to DBFS
        combined_path = f"{output_dir}/combined_data/combined_news.parquet"
        self.save_to_parquet(combined_df, combined_path, partition_by="label")
        
        # Save sample dataset to DBFS
        sample_path = f"{output_dir}/sample_data/sample_news.parquet"
        self.save_to_parquet(sample_df, sample_path)
        
        # Save to Hive tables for easier access
        self.save_to_hive_table(combined_df, "combined_news", partition_by="label")
        self.save_to_hive_table(sample_df, "sample_news")
        
        return {
            "real_df": real_df,
            "fake_df": fake_df,
            "combined_df": combined_df,
            "sample_df": sample_df
        }


# Example usage:
# from pyspark.sql import SparkSession
# 
# # Create a Spark session with appropriate configuration
# spark = SparkSession.builder \
#     .appName("FakeNewsDataIngestion") \
#     .config("spark.sql.files.maxPartitionBytes", "128MB") \
#     .config("spark.sql.shuffle.partitions", "200") \
#     .config("spark.memory.offHeap.enabled", "true") \
#     .config("spark.memory.offHeap.size", "1g") \
#     .enableHiveSupport() \
#     .getOrCreate()
# 
# # Initialize the data ingestion
# ingestion = HiveDataIngestion(spark)
# 
# # Process and save data
# result = ingestion.process_and_save_data()
"""
