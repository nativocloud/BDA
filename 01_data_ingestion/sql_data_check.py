def check_random_records(df, num_records=10):
    """
    Uses SQL to check random records from a DataFrame.
    Optimized for Databricks environment.
    
    Args:
        df: DataFrame to check
        num_records: Number of random records to display (default: 10)
        
    Returns:
        None
    """
    print("\n" + "="*80)
    print(f"📋 CHECKING {num_records} RANDOM RECORDS")
    print("="*80)
    
    # Register DataFrame as a temporary view
    df.createOrReplaceTempView("temp_data")
    
    # SQL query to select random records
    query = f"""
    SELECT *
    FROM temp_data
    ORDER BY rand()
    LIMIT {num_records}
    """
    
    # Execute the query
    result = spark.sql(query)
    
    # Display the results
    print(f"\nRandom sample of {num_records} records:")
    display(result)
    
    # Show schema information
    print("\nSchema information:")
    result.printSchema()
    
    # Show basic statistics for each column
    print("\nBasic statistics for each column:")
    
    # For each column, show some basic stats if applicable
    for column in df.columns:
        col_type = df.schema[column].dataType.typeName()
        
        if col_type in ["string", "binary"]:
            # For string columns, show length statistics
            stats_query = f"""
            SELECT 
                MIN(LENGTH({column})) as min_length,
                MAX(LENGTH({column})) as max_length,
                AVG(LENGTH({column})) as avg_length
            FROM temp_data
            WHERE {column} IS NOT NULL
            """
            stats = spark.sql(stats_query)
            print(f"\nColumn: {column} (Type: {col_type})")
            display(stats)
        elif col_type in ["integer", "long", "double", "float", "decimal"]:
            # For numeric columns, show basic statistics
            stats_query = f"""
            SELECT 
                MIN({column}) as min_value,
                MAX({column}) as max_value,
                AVG({column}) as avg_value
            FROM temp_data
            WHERE {column} IS NOT NULL
            """
            stats = spark.sql(stats_query)
            print(f"\nColumn: {column} (Type: {col_type})")
            display(stats)
    
    print("\n" + "="*80)
    
    return result
