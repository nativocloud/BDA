# Data Ingestion for Fake News Detection in Databricks

## Overview

Data ingestion is the first critical step in our fake news detection pipeline. This phase involves loading data from Hive metastore tables, combining datasets, and preparing them for subsequent processing steps. The implementation leverages Apache Spark's distributed processing capabilities to handle large-scale data efficiently within the Databricks environment.

## Hive Metastore Integration

### Why Use Hive Tables?

In Databricks, we use Hive metastore tables as our primary data source for several important reasons:

1. **Centralized Metadata Management**: Schema information is stored centrally, ensuring consistency across sessions and users
2. **Optimized Performance**: Databricks optimizes query execution on Hive tables
3. **Access Control**: Tables can have access controls applied at the table level
4. **Persistence**: Tables persist across cluster restarts and sessions
5. **Catalog Integration**: Tables are visible in the Databricks catalog UI

### Available Tables

Our fake news detection pipeline uses two primary tables in the Hive metastore:

1. **`fake`**: Contains fake news articles with columns: title, text, subject, date
2. **`real`**: Contains real news articles with columns: title, text, subject, date

## Implementation Details

### HiveDataIngestion Class

We've implemented a `HiveDataIngestion` class that handles all aspects of data loading from Hive tables:

```python
class HiveDataIngestion:
    def __init__(self, spark, real_table="real", fake_table="fake"):
        self.spark = spark
        self.real_table = real_table
        self.fake_table = fake_table
    
    def load_data_from_hive(self):
        # Load data from Hive tables
        real_df = self.spark.table(self.real_table)
        fake_df = self.spark.table(self.fake_table)
        return real_df, fake_df
```

### Spark Session Configuration

To access Hive tables and process the complete dataset efficiently, we configure our Spark session with optimized parameters:

```python
spark = SparkSession.builder \
    .appName("FakeNewsDetection") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .enableHiveSupport() \
    .getOrCreate()
```

### Key Configuration Parameters

- **spark.sql.shuffle.partitions = "200"**: Sets the number of partitions used during shuffle operations, balancing parallelism with overhead. For larger datasets, consider increasing to 400-600.
- **spark.executor.memory = "4g"**: Allocates 4GB of memory to each executor, suitable for processing the complete dataset. Increase to 8-16g for very large datasets.
- **spark.driver.memory = "4g"**: Allocates 4GB of memory to the driver process, appropriate for coordinating distributed processing of text data.

## Data Processing Steps

1. **Load Data from Hive**: Read data directly from the `real` and `fake` tables in the Hive metastore
2. **Add Labels**: Add binary labels (1 for real news, 0 for fake news)
3. **Combine Datasets**: Merge real and fake news into a single dataset
4. **Analyze Subject Distribution**: Check for potential data leakage in the subject column
5. **Save Processed Data**: Store data in partitioned Parquet format in DBFS and as Hive tables
6. **Optional Sampling**: Create balanced samples only when explicitly needed for development or testing

## Databricks File System (DBFS) Usage

In Databricks, we use DBFS paths for storing and accessing files:

```python
# Example of saving to DBFS with partitioning and compression
df.write \
    .mode("overwrite") \
    .partitionBy("label") \
    .option("compression", "snappy") \
    .parquet("dbfs:/FileStore/fake_news_detection/data/combined_data/combined_news.parquet")

# Example of reading from DBFS
df = spark.read.parquet("dbfs:/FileStore/fake_news_detection/data/combined_data/combined_news.parquet")
```

### Why Use DBFS?

1. **Persistence**: Files stored in DBFS persist across cluster restarts
2. **Accessibility**: Files can be accessed from any cluster in the workspace
3. **Integration**: DBFS is integrated with Databricks UI for easy browsing
4. **Performance**: DBFS is optimized for Databricks workloads

### Directory Structure

We organize our data in DBFS with the following structure:

```
dbfs:/FileStore/fake_news_detection/
├── data/
│   ├── combined_data/
│   ├── processed_data/
│   ├── feature_data/
│   └── models/
└── logs/
    └── (Performance metrics and visualizations)
```

## Optimization Techniques

We employ several optimization techniques to ensure efficient distributed processing of the complete dataset:

1. **Parquet Format**: Store processed data in columnar Parquet format for efficient I/O
2. **Data Partitioning**: Partition data by label for more efficient filtering and parallel processing
3. **Compression**: Use Snappy compression for a good balance of compression ratio and speed
4. **Caching**: Strategically cache frequently accessed DataFrames with `df.cache()`
5. **Repartitioning**: Optimize partition count based on data size with `df.repartition(n)`

## Usage in Databricks

In Databricks Community Edition, you can access the Hive tables through the Data tab in the sidebar. The tables are located in the `hive_metastore.default` catalog and database.

To query these tables directly in SQL:

```sql
SELECT * FROM hive_metastore.default.real LIMIT 10;
SELECT * FROM hive_metastore.default.fake LIMIT 10;
```

To access the processed data saved as Hive tables:

```sql
SELECT * FROM combined_news WHERE label = 1 LIMIT 10; -- Real news
SELECT * FROM combined_news WHERE label = 0 LIMIT 10; -- Fake news
```

## Next Steps

After data ingestion, the pipeline proceeds to:

1. **Data Preprocessing**: Clean and standardize text data
2. **Feature Engineering**: Extract features from text and metadata
3. **Model Training**: Train machine learning models for fake news detection

The ingested data is made available to these subsequent steps through Hive tables and DBFS Parquet files.
