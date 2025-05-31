def basic_dataset_analysis(df):
    """
    Very basic dataset analysis function with minimal Spark operations.
    Only performs essential checks to avoid overwhelming the cluster.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        dict: Dictionary with basic analysis results
    """
    from pyspark.sql.functions import col, count, when
    
    print("\n" + "="*50)
    print("📊 BASIC DATASET ANALYSIS")
    print("="*50)
    
    # Get column names
    columns = df.columns
    print(f"• Columns: {', '.join(columns)}")
    
    # Get total count (single Spark action)
    total_count = df.count()
    print(f"• Total records: {total_count}")
    
    # Check for required columns
    has_text = "text" in columns
    has_label = "label" in columns
    
    # Basic class distribution if label exists
    if has_label:
        print("\n• Class distribution:")
        # Use a single SQL query instead of multiple DataFrame operations
        df.createOrReplaceTempView("temp_data")
        class_dist = spark.sql("""
            SELECT label, COUNT(*) as count
            FROM temp_data
            GROUP BY label
            ORDER BY label
        """)
        class_dist.show()
    
    # Check for null values in important columns
    print("\n• Null value check:")
    null_counts = {}
    
    # Only check a few important columns to minimize operations
    columns_to_check = []
    if has_text:
        columns_to_check.append("text")
    if has_label:
        columns_to_check.append("label")
    if "location" in columns:
        columns_to_check.append("location")
    if "news_source" in columns:
        columns_to_check.append("news_source")
    
    for column_name in columns_to_check:
        null_count = df.filter(col(column_name).isNull()).count()
        null_counts[column_name] = null_count
        print(f"  - Null values in '{column_name}': {null_count}")
    
    # Check for duplicates in text column if it exists
    if has_text:
        print("\n• Duplicate check:")
        unique_count = df.select("text").distinct().count()
        duplicate_count = total_count - unique_count
        print(f"  - Duplicate texts: {duplicate_count}")
    
    print("\n" + "="*50)
    
    # Return minimal results
    return {
        "total_count": total_count,
        "columns": columns,
        "null_counts": null_counts
    }
