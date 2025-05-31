def analyze_dataset_characteristics_optimized(df, save_path=None):
    """
    Optimized version of dataset characteristics analysis that minimizes Spark jobs.
    Analyzes dataset characteristics while reducing the number of actions that trigger Spark jobs.
    
    Args:
        df: DataFrame with text, label, and potentially location/news_source columns
        save_path: Optional path to save visualizations (not used in this version)
        
    Returns:
        dict: Dictionary with analysis results
    """
    from pyspark.sql.functions import col, length, count, when, isnan, isnull, avg, min as spark_min, max as spark_max
    from pyspark.sql.functions import countDistinct, sum as spark_sum, lit
    import builtins  # Import Python's built-in functions to avoid namespace conflicts
    
    print("\n" + "="*80)
    print("📊 OPTIMIZED DATASET CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    # Cache the DataFrame if it's not already cached to improve performance
    if not df.is_cached:
        print("• Caching DataFrame for optimized analysis...")
        df.cache()
    
    # --- Collect basic information in a single action ---
    print("\n• Collecting basic dataset information...")
    
    # Register the DataFrame as a temporary view for SQL operations
    df.createOrReplaceTempView("dataset_analysis")
    
    # Get column names
    columns = df.columns
    
    # Check for required and special columns
    has_text = "text" in columns
    has_label = "label" in columns
    has_location = "location" in columns
    has_news_source = "news_source" in columns
    
    # --- Prepare aggregation query to minimize Spark jobs ---
    # This single query will collect multiple metrics at once
    
    # Start with basic count metrics
    agg_exprs = [
        count("*").alias("total_count")
    ]
    
    # Add text-specific metrics if text column exists
    if has_text:
        agg_exprs.extend([
            countDistinct("text").alias("unique_text_count"),
            avg(length(col("text"))).alias("avg_text_length"),
            spark_min(length(col("text"))).alias("min_text_length"),
            spark_max(length(col("text"))).alias("max_text_length"),
            count(when(length(col("text")) < 10, True)).alias("short_text_count")
        ])
    
    # Add label-specific metrics if label column exists
    if has_label:
        # For binary classification (0/1)
        agg_exprs.extend([
            count(when(col("label") == 0, True)).alias("label_0_count"),
            count(when(col("label") == 1, True)).alias("label_1_count")
        ])
    
    # Add location metrics if location column exists
    if has_location:
        agg_exprs.extend([
            count(when(col("location").isNotNull(), True)).alias("location_not_null_count")
        ])
    
    # Add news_source metrics if news_source column exists
    if has_news_source:
        agg_exprs.extend([
            count(when(col("news_source").isNotNull(), True)).alias("news_source_not_null_count")
        ])
    
    # Add null count metrics for all columns
    for column_name in columns:
        agg_exprs.append(
            count(when(col(column_name).isNull() | isnan(col(column_name)), True)).alias(f"{column_name}_null_count")
        )
    
    # Execute the aggregation query to get all basic metrics in one go
    print("• Executing optimized aggregation query...")
    metrics_df = df.agg(*agg_exprs)
    metrics = metrics_df.collect()[0]
    
    # Extract metrics from the result
    total_count = metrics["total_count"]
    
    # --- 1. Record Count ---
    print("\n1️⃣ RECORD COUNT")
    print(f"• Total records: {total_count:,}")
    
    # --- 2. Column Presence ---
    print("\n2️⃣ COLUMN PRESENCE")
    print(f"• Available columns: {', '.join(columns)}")
    
    # Check for required columns
    required_columns = ["text", "label"]
    missing_columns = [col for col in required_columns if col not in columns]
    if missing_columns:
        print(f"⚠️ Missing required columns: {', '.join(missing_columns)}")
    else:
        print("✅ All required columns present")
    
    # Check for new columns from preprocessing
    new_columns = []
    if has_location:
        new_columns.append("location")
    if has_news_source:
        new_columns.append("news_source")
    
    if new_columns:
        print(f"✅ Detected {len(new_columns)} columns from preprocessing: {', '.join(new_columns)}")
    
    # --- 3. Class Distribution ---
    class_counts = {}
    class_balance = None
    if has_label:
        print("\n3️⃣ CLASS DISTRIBUTION")
        
        # Extract label counts from metrics
        label_0_count = metrics["label_0_count"] if "label_0_count" in metrics else 0
        label_1_count = metrics["label_1_count"] if "label_1_count" in metrics else 0
        
        class_counts = {0: label_0_count, 1: label_1_count}
        
        # Create a DataFrame for display
        class_dist_data = [(0, label_0_count), (1, label_1_count)]
        class_dist = spark.createDataFrame(class_dist_data, ["label", "count"])
        
        print("• Class distribution:")
        display(class_dist)
        
        # Calculate class balance
        if label_0_count > 0 and label_1_count > 0:
            min_count = builtins.min(label_0_count, label_1_count)
            max_count = builtins.max(label_0_count, label_1_count)
            class_balance = min_count / max_count
            print(f"• Class balance ratio (minority/majority): {class_balance:.4f}")
    
    # --- 4. Text Length Statistics ---
    if has_text:
        print("\n4️⃣ TEXT LENGTH STATISTICS")
        
        # Extract text metrics
        unique_text_count = metrics["unique_text_count"]
        avg_text_length = metrics["avg_text_length"]
        min_text_length = metrics["min_text_length"]
        max_text_length = metrics["max_text_length"]
        short_text_count = metrics["short_text_count"]
        
        # Calculate derived metrics
        duplicate_count = total_count - unique_text_count
        duplicate_pct = (duplicate_count / total_count) * 100 if total_count > 0 else 0
        short_texts_pct = (short_text_count / total_count) * 100 if total_count > 0 else 0
        
        # Display basic text statistics
        print("• Text length basic statistics:")
        text_stats_data = [("Average Length", avg_text_length), 
                          ("Minimum Length", min_text_length), 
                          ("Maximum Length", max_text_length)]
        text_stats_df = spark.createDataFrame(text_stats_data, ["Metric", "Value"])
        display(text_stats_df)
        
        # Calculate quantiles with a single action
        print("• Calculating text length quantiles...")
        quantiles = df.select(length(col("text")).alias("text_length")).approxQuantile("text_length", [0.25, 0.5, 0.75], 0.05)
        print(f"• Text length quartiles: 25%={int(quantiles[0])}, 50%={int(quantiles[1])}, 75%={int(quantiles[2])} characters")
        
        print(f"• Very short texts (<10 chars): {short_text_count:,} ({short_texts_pct:.2f}%)")
        
        # Display text length histogram (single action)
        print("\n• Text length distribution histogram:")
        display(df.select(length(col("text")).alias("text_length")).sample(False, 0.1, seed=42))
        
        # Display duplicate information
        print(f"\n• Unique texts: {unique_text_count:,} ({unique_text_count/total_count*100:.2f}%)")
        print(f"• Duplicate texts: {duplicate_count:,} ({duplicate_pct:.2f}%)")
        
        # If there are duplicates, show a sample of the most common duplicated texts
        if duplicate_count > 0:
            print("\n• Sample of duplicated texts (limited to reduce Spark jobs):")
            duplicate_texts_query = """
            SELECT text, COUNT(*) as count
            FROM dataset_analysis
            GROUP BY text
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 5
            """
            duplicate_texts = spark.sql(duplicate_texts_query)
            display(duplicate_texts)
    
    # --- 5. Location Analysis ---
    location_not_null_count = 0
    location_not_null_pct = 0
    if has_location:
        print("\n5️⃣ LOCATION ANALYSIS")
        
        # Extract location metrics
        location_not_null_count = metrics["location_not_null_count"]
        location_null_count = total_count - location_not_null_count
        location_not_null_pct = (location_not_null_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"• Records with location information: {location_not_null_count:,} ({location_not_null_pct:.2f}%)")
        print(f"• Records without location information: {location_null_count:,} ({100-location_not_null_pct:.2f}%)")
        
        # Get top locations with a single SQL query
        if location_not_null_count > 0:
            print("\n• Top 5 most common locations (limited to reduce Spark jobs):")
            top_locations_query = """
            SELECT location, COUNT(*) as count
            FROM dataset_analysis
            WHERE location IS NOT NULL
            GROUP BY location
            ORDER BY COUNT(*) DESC
            LIMIT 5
            """
            top_locations = spark.sql(top_locations_query)
            display(top_locations)
    
    # --- 6. News Source Analysis ---
    source_not_null_count = 0
    source_not_null_pct = 0
    if has_news_source:
        print("\n6️⃣ NEWS SOURCE ANALYSIS")
        
        # Extract news source metrics
        source_not_null_count = metrics["news_source_not_null_count"]
        source_null_count = total_count - source_not_null_count
        source_not_null_pct = (source_not_null_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"• Records with news source information: {source_not_null_count:,} ({source_not_null_pct:.2f}%)")
        print(f"• Records without news source information: {source_null_count:,} ({100-source_not_null_pct:.2f}%)")
        
        # Get top news sources with a single SQL query
        if source_not_null_count > 0:
            print("\n• Top 5 most common news sources (limited to reduce Spark jobs):")
            top_sources_query = """
            SELECT news_source, COUNT(*) as count
            FROM dataset_analysis
            WHERE news_source IS NOT NULL
            GROUP BY news_source
            ORDER BY COUNT(*) DESC
            LIMIT 5
            """
            top_sources = spark.sql(top_sources_query)
            display(top_sources)
            
            # Check for potential data leakage in news sources with a single SQL query
            if has_label:
                print("\n• Checking for potential data leakage in news sources (limited to reduce Spark jobs)...")
                
                # SQL query to find sources that appear predominantly in one class
                leakage_query = """
                SELECT 
                    news_source,
                    SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) as fake_count,
                    SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as true_count,
                    COUNT(*) as total_count,
                    SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) / COUNT(*) as fake_ratio,
                    SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) / COUNT(*) as true_ratio,
                    ABS(SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) / COUNT(*) - 0.5) * 2 as bias_score
                FROM 
                    dataset_analysis
                WHERE 
                    news_source IS NOT NULL
                GROUP BY 
                    news_source
                HAVING 
                    COUNT(*) >= 5
                ORDER BY 
                    bias_score DESC
                LIMIT 5
                """
                
                leakage_df = spark.sql(leakage_query)
                print("\n• Top 5 news sources with potential class bias (higher bias score = higher potential for data leakage):")
                display(leakage_df)
    
    # --- 7. Null Values Check ---
    print("\n7️⃣ NULL VALUES CHECK")
    null_counts = {}
    
    # Extract null counts from metrics
    for column_name in columns:
        null_count = metrics[f"{column_name}_null_count"]
        null_counts[column_name] = null_count
        null_pct = (null_count / total_count) * 100 if total_count > 0 else 0
        print(f"• Null values in '{column_name}': {null_count:,} ({null_pct:.2f}%)")
    
    # Summary of potential issues
    print("\n" + "="*80)
    print("📋 ANALYSIS SUMMARY")
    print("="*80)
    
    issues = []
    
    # Check for class imbalance
    if has_label and class_balance is not None:
        if class_balance < 0.2:
            issues.append(f"Severe class imbalance (ratio: {class_balance:.4f})")
        elif class_balance < 0.5:
            issues.append(f"Moderate class imbalance (ratio: {class_balance:.4f})")
    
    # Check for short texts
    if has_text and short_texts_pct > 1:
        issues.append(f"Dataset contains {short_text_count:,} very short texts (<10 chars)")
    
    # Check for duplicates
    if has_text and duplicate_pct > 1:
        issues.append(f"Dataset contains {duplicate_count:,} duplicate texts ({duplicate_pct:.2f}%)")
    
    # Check for null values
    null_issues = [f"{count:,} null values in '{col}'" for col, count in null_counts.items() if count > 0]
    issues.extend(null_issues)
    
    # Check for location/source extraction effectiveness
    if has_location and location_not_null_pct < 10:
        issues.append(f"Low location extraction rate ({location_not_null_pct:.2f}%)")
    
    if has_news_source and source_not_null_pct < 10:
        issues.append(f"Low news source extraction rate ({source_not_null_pct:.2f}%)")
    
    # Print issues or success message
    if issues:
        print("\n⚠️ POTENTIAL ISSUES IDENTIFIED:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ No significant issues detected in the dataset")
    
    print("\n" + "="*80)
    
    # Return analysis results
    results = {
        "total_count": total_count,
        "columns": columns,
        "missing_required_columns": missing_columns,
        "class_distribution": class_counts if has_label else None,
        "class_balance": class_balance,
        "text_length_stats": {
            "avg": avg_text_length,
            "min": min_text_length,
            "max": max_text_length,
            "quantiles": {
                "25%": quantiles[0] if has_text else None,
                "50%": quantiles[1] if has_text else None,
                "75%": quantiles[2] if has_text else None
            }
        } if has_text else None,
        "short_texts": {
            "count": short_text_count,
            "percentage": short_texts_pct
        } if has_text else None,
        "duplicates": {
            "count": duplicate_count,
            "percentage": duplicate_pct
        } if has_text else None,
        "location_stats": {
            "present_count": location_not_null_count,
            "present_percentage": location_not_null_pct
        } if has_location else None,
        "news_source_stats": {
            "present_count": source_not_null_count,
            "present_percentage": source_not_null_pct
        } if has_news_source else None,
        "null_values": null_counts,
        "issues": issues
    }
    
    print("Analysis complete with optimized Spark job usage.")
    return results
