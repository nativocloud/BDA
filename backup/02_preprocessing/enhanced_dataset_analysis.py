def analyze_dataset_characteristics(df, save_path=None):
    """
    Analyzes dataset characteristics to identify potential issues.
    Provides basic validation checks using native PySpark functionality.
    Optimized for Databricks environment with display() visualizations.
    Enhanced to handle location and news_source columns from preprocessing.
    
    Args:
        df: DataFrame with text, label, and potentially location/news_source columns
        save_path: Optional path to save visualizations (not used in this version)
        
    Returns:
        dict: Dictionary with analysis results
    """
    from pyspark.sql.functions import col, length, count, when, isnan, isnull, avg, min, max, countDistinct
    
    print("\n" + "="*80)
    print("📊 DATASET CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    # 1. Record Count
    print("\n1️⃣ RECORD COUNT")
    total_count = df.count()
    print(f"• Total records: {total_count:,}")
    
    # 2. Column Presence
    print("\n2️⃣ COLUMN PRESENCE")
    columns = df.columns
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
    if "location" in columns:
        new_columns.append("location")
    if "news_source" in columns:
        new_columns.append("news_source")
    
    if new_columns:
        print(f"✅ Detected {len(new_columns)} columns from preprocessing: {', '.join(new_columns)}")
    
    # 3. Class Distribution (if label column exists)
    class_counts = {}
    if "label" in columns:
        print("\n3️⃣ CLASS DISTRIBUTION")
        class_dist = df.groupBy("label").count().orderBy("label")
        
        # Use display() for better visualization in Databricks
        print("• Class distribution:")
        display(class_dist)
        
        # Calculate class balance
        class_counts = {row["label"]: row["count"] for row in class_dist.collect()}
        if len(class_counts) > 1:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            class_balance = min_count / max_count
            print(f"• Class balance ratio (minority/majority): {class_balance:.4f}")
    
    # 4. Text Length Statistics (if text column exists)
    if "text" in columns:
        print("\n4️⃣ TEXT LENGTH STATISTICS")
        
        # Add text length column
        df_with_length = df.withColumn("text_length", length(col("text")))
        
        # Calculate text length statistics
        length_stats = df_with_length.agg(
            avg("text_length").alias("avg_length"),
            min("text_length").alias("min_length"),
            max("text_length").alias("max_length")
        )
        
        # Display statistics
        print("• Text length basic statistics:")
        display(length_stats)
        
        # Calculate quantiles for better understanding of distribution
        quantiles = df_with_length.approxQuantile("text_length", [0.25, 0.5, 0.75], 0.01)
        print(f"• Text length quartiles: 25%={int(quantiles[0])}, 50%={int(quantiles[1])}, 75%={int(quantiles[2])} characters")
        
        # Count short texts
        short_texts = df_with_length.filter(col("text_length") < 10).count()
        short_texts_pct = (short_texts / total_count) * 100
        print(f"• Very short texts (<10 chars): {short_texts:,} ({short_texts_pct:.2f}%)")
        
        # Display text length distribution by label
        print("\n• Text length distribution by label:")
        text_length_by_label = df_with_length.groupBy("label").agg(
            avg("text_length").alias("avg_length"),
            min("text_length").alias("min_length"),
            max("text_length").alias("max_length"),
            count(when(col("text_length") < 10, True)).alias("short_texts_count")
        )
        display(text_length_by_label)
        
        # Display histogram of text lengths using Databricks visualization
        print("\n• Text length distribution histogram:")
        display(df_with_length.select("text_length"))
    
    # 5. Location Analysis (if location column exists from preprocessing)
    if "location" in columns:
        print("\n5️⃣ LOCATION ANALYSIS")
        
        # Count null and non-null values
        location_null_count = df.filter(col("location").isNull()).count()
        location_not_null_count = total_count - location_null_count
        location_not_null_pct = (location_not_null_count / total_count) * 100
        
        print(f"• Records with location information: {location_not_null_count:,} ({location_not_null_pct:.2f}%)")
        print(f"• Records without location information: {location_null_count:,} ({100-location_not_null_pct:.2f}%)")
        
        # Get top locations
        if location_not_null_count > 0:
            print("\n• Top 10 most common locations:")
            top_locations = df.filter(col("location").isNotNull()) \
                              .groupBy("location") \
                              .count() \
                              .orderBy(col("count").desc()) \
                              .limit(10)
            display(top_locations)
            
            # Location distribution by class (if label exists)
            if "label" in columns:
                print("\n• Location distribution by class:")
                location_by_class = df.filter(col("location").isNotNull()) \
                                      .groupBy("label", "location") \
                                      .count() \
                                      .orderBy("label", col("count").desc())
                display(location_by_class)
    
    # 6. News Source Analysis (if news_source column exists from preprocessing)
    if "news_source" in columns:
        print("\n6️⃣ NEWS SOURCE ANALYSIS")
        
        # Count null and non-null values
        source_null_count = df.filter(col("news_source").isNull()).count()
        source_not_null_count = total_count - source_null_count
        source_not_null_pct = (source_not_null_count / total_count) * 100
        
        print(f"• Records with news source information: {source_not_null_count:,} ({source_not_null_pct:.2f}%)")
        print(f"• Records without news source information: {source_null_count:,} ({100-source_not_null_pct:.2f}%)")
        
        # Get top news sources
        if source_not_null_count > 0:
            print("\n• Top 10 most common news sources:")
            top_sources = df.filter(col("news_source").isNotNull()) \
                            .groupBy("news_source") \
                            .count() \
                            .orderBy(col("count").desc()) \
                            .limit(10)
            display(top_sources)
            
            # News source distribution by class (if label exists)
            if "label" in columns:
                print("\n• News source distribution by class:")
                source_by_class = df.filter(col("news_source").isNotNull()) \
                                    .groupBy("label", "news_source") \
                                    .count() \
                                    .orderBy("label", col("count").desc())
                display(source_by_class)
                
                # Check for potential data leakage in news sources
                print("\n• Checking for potential data leakage in news sources...")
                
                # Create temporary view for SQL query
                df.createOrReplaceTempView("news_data")
                
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
                    news_data
                WHERE 
                    news_source IS NOT NULL
                GROUP BY 
                    news_source
                HAVING 
                    COUNT(*) >= 5
                ORDER BY 
                    bias_score DESC
                LIMIT 10
                """
                
                leakage_df = spark.sql(leakage_query)
                print("\n• Top 10 news sources with potential class bias (higher bias score = higher potential for data leakage):")
                display(leakage_df)
    
    # 7. Duplicate Check
    print("\n7️⃣ DUPLICATE CHECK")
    if "text" in columns:
        unique_count = df.select("text").distinct().count()
        duplicate_count = total_count - unique_count
        duplicate_pct = (duplicate_count / total_count) * 100
        print(f"• Unique texts: {unique_count:,} ({unique_count/total_count*100:.2f}%)")
        print(f"• Duplicate texts: {duplicate_count:,} ({duplicate_pct:.2f}%)")
        
        # If there are duplicates, show the most common duplicated texts
        if duplicate_count > 0:
            print("\n• Most common duplicated texts:")
            duplicate_texts = df.groupBy("text").count().filter(col("count") > 1).orderBy(col("count").desc()).limit(10)
            display(duplicate_texts)
    
    # 8. Null Values Check
    print("\n8️⃣ NULL VALUES CHECK")
    null_counts = {}
    null_data = []
    
    for column in columns:
        null_count = df.filter(col(column).isNull() | isnan(col(column))).count()
        null_counts[column] = null_count
        null_pct = (null_count / total_count) * 100
        print(f"• Null values in '{column}': {null_count:,} ({null_pct:.2f}%)")
        null_data.append({"column": column, "null_count": null_count, "null_percentage": null_pct})
    
    # Display null values summary
    if any(count > 0 for count in null_counts.values()):
        print("\n• Null values summary:")
        null_df = spark.createDataFrame(null_data)
        display(null_df)
    
    # Summary of potential issues
    print("\n" + "="*80)
    print("📋 ANALYSIS SUMMARY")
    print("="*80)
    
    issues = []
    
    # Check for class imbalance
    if "label" in columns and len(class_counts) > 1:
        if class_balance < 0.2:
            issues.append(f"Severe class imbalance (ratio: {class_balance:.4f})")
        elif class_balance < 0.5:
            issues.append(f"Moderate class imbalance (ratio: {class_balance:.4f})")
    
    # Check for short texts
    if "text" in columns and short_texts_pct > 1:
        issues.append(f"Dataset contains {short_texts:,} very short texts (<10 chars)")
    
    # Check for duplicates
    if "text" in columns and duplicate_pct > 1:
        issues.append(f"Dataset contains {duplicate_count:,} duplicate texts ({duplicate_pct:.2f}%)")
    
    # Check for null values
    null_issues = [f"{count:,} null values in '{col}'" for col, count in null_counts.items() if count > 0]
    issues.extend(null_issues)
    
    # Check for location/source extraction effectiveness
    if "location" in columns and location_not_null_pct < 10:
        issues.append(f"Low location extraction rate ({location_not_null_pct:.2f}%)")
    
    if "news_source" in columns and source_not_null_pct < 10:
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
        "class_distribution": class_counts if "label" in columns else None,
        "class_balance": class_balance if "label" in columns and len(class_counts) > 1 else None,
        "text_length_stats": {
            "avg": length_stats.collect()[0]["avg_length"],
            "min": length_stats.collect()[0]["min_length"],
            "max": length_stats.collect()[0]["max_length"],
            "quantiles": {
                "25%": quantiles[0],
                "50%": quantiles[1],
                "75%": quantiles[2]
            }
        } if "text" in columns else None,
        "short_texts": {
            "count": short_texts,
            "percentage": short_texts_pct
        } if "text" in columns else None,
        "duplicates": {
            "count": duplicate_count,
            "percentage": duplicate_pct
        } if "text" in columns else None,
        "location_stats": {
            "present_count": location_not_null_count,
            "present_percentage": location_not_null_pct
        } if "location" in columns else None,
        "news_source_stats": {
            "present_count": source_not_null_count,
            "present_percentage": source_not_null_pct
        } if "news_source" in columns else None,
        "null_values": null_counts,
        "issues": issues
    }
    
    return results
