def save_to_hive_table_safely(df, table_name, partition_by=None, mode="overwrite"):
    """
    Safely saves a DataFrame to a Hive table, handling existing tables and locations.
    
    Args:
        df: DataFrame to save
        table_name: Name of the Hive table
        partition_by: Column(s) to partition by (optional)
        mode: Write mode (default: "overwrite")
    
    Returns:
        bool: True if successful
    """
    print(f"Safely saving DataFrame to Hive table: {table_name}")
    
    # Check if table exists
    tables = spark.sql("SHOW TABLES").select("tableName").rdd.flatMap(lambda x: x).collect()
    table_exists = table_name in tables
    
    if table_exists:
        print(f"Table '{table_name}' already exists. Dropping it...")
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        print(f"Table '{table_name}' dropped successfully.")
    
    # Check if the location exists and remove it if necessary
    # This is needed because dropping the table might not always remove the underlying data
    try:
        location_path = f"dbfs:/user/hive/warehouse/{table_name}"
        print(f"Checking if location exists: {location_path}")
        
        # Use dbutils to check if path exists and delete it if it does
        if dbutils.fs.ls(location_path):
            print(f"Location exists. Removing directory: {location_path}")
            dbutils.fs.rm(location_path, recurse=True)
            print(f"Directory removed successfully.")
    except Exception as e:
        # Path might not exist, which is fine
        print(f"Note: {str(e)}")
    
    # Save the DataFrame to the Hive table
    print(f"Saving DataFrame to table '{table_name}'...")
    
    if partition_by:
        df.write.format("parquet").partitionBy(partition_by).mode(mode).saveAsTable(table_name)
        print(f"DataFrame saved to table '{table_name}' with partitioning on '{partition_by}'.")
    else:
        df.write.format("parquet").mode(mode).saveAsTable(table_name)
        print(f"DataFrame saved to table '{table_name}'.")
    
    # Verify the table was created
    tables = spark.sql("SHOW TABLES").select("tableName").rdd.flatMap(lambda x: x).collect()
    if table_name in tables:
        print(f"Verified: Table '{table_name}' exists.")
        
        # Show table information
        print("\nTable information:")
        spark.sql(f"DESCRIBE TABLE {table_name}").show(truncate=False)
        
        # Show record count
        count = spark.sql(f"SELECT COUNT(*) as count FROM {table_name}").collect()[0]['count']
        print(f"\nRecord count: {count:,}")
        
        return True
    else:
        print(f"Error: Failed to create table '{table_name}'.")
        return False
