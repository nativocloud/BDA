def analyze_dataset_characteristics(df, save_path=None):
    """
    Analyzes dataset characteristics to identify potential issues.
    Provides visual and user-friendly output with clear explanations and recommendations.
    
    Args:
        df: DataFrame with text and label columns
        save_path: Optional path to save visualizations (for non-Databricks environments)
        
    Returns:
        dict: Dictionary with analysis results and recommendations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pyspark.sql.functions import col, length, count, when, isnan, avg, min, max, countDistinct
    
    print("\n" + "="*80)
    print("📊 DATASET CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    # Check if we're in a Databricks environment
    in_databricks = False
    try:
        from pyspark.dbutils import DBUtils
        in_databricks = True
    except ImportError:
        pass
    
    # Calculate basic statistics using Spark SQL for better performance
    print("\n🔍 ANALYZING DATASET CHARACTERISTICS...")
    
    # Get total count
    total_count = df.count()
    print(f"• Total records: {total_count:,}")
    
    # Class distribution
    class_dist = df.groupBy("label").count().orderBy("label")
    
    # Convert to pandas for visualization
    class_dist_pd = class_dist.toPandas()
    class_dist_pd['percentage'] = class_dist_pd['count'] / total_count * 100
    
    # Calculate class balance
    if len(class_dist_pd) > 1:
        class_balance = class_dist_pd['count'].min() / class_dist_pd['count'].max()
        majority_class = class_dist_pd.loc[class_dist_pd['count'].idxmax(), 'label']
        minority_class = class_dist_pd.loc[class_dist_pd['count'].idxmin(), 'label']
    else:
        class_balance = 1.0
        majority_class = class_dist_pd['label'].iloc[0] if len(class_dist_pd) > 0 else "unknown"
        minority_class = majority_class
    
    # Add text length column
    df_with_length = df.withColumn("text_length", length(col("text")))
    
    # Calculate text length statistics
    length_stats = df_with_length.agg(
        avg("text_length").alias("avg_length"),
        min("text_length").alias("min_length"),
        max("text_length").alias("max_length")
    ).collect()[0]
    
    avg_length = length_stats["avg_length"]
    min_length = length_stats["min_length"]
    max_length = length_stats["max_length"]
    
    # Count short texts
    short_texts = df_with_length.filter(col("text_length") < 10).count()
    short_texts_pct = (short_texts / total_count) * 100
    
    # Count duplicate texts
    duplicate_count = total_count - df.select("text").distinct().count()
    duplicate_pct = (duplicate_count / total_count) * 100
    
    # Count null values
    null_counts = {}
    for column in df.columns:
        null_count = df.filter(col(column).isNull() | isnan(col(column))).count()
        if null_count > 0:
            null_counts[column] = null_count
    
    # Calculate text length distribution for visualization
    # Use approx_quantile for better performance on large datasets
    quantiles = df_with_length.approxQuantile("text_length", 
                                             [0.1, 0.25, 0.5, 0.75, 0.9], 0.01)
    
    # Sample data for visualization (to avoid memory issues)
    sample_size = min(10000, total_count)
    df_sample = df_with_length.sample(fraction=sample_size/total_count, seed=42)
    
    # Convert sample to pandas for visualization
    sample_pd = df_sample.select("text_length", "label").toPandas()
    
    # Create visualizations
    plt.figure(figsize=(16, 12))
    
    # 1. Class distribution
    plt.subplot(2, 2, 1)
    ax = sns.barplot(x='label', y='count', data=class_dist_pd, palette='viridis')
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Class Label (0=Fake, 1=True)')
    plt.ylabel('Count')
    
    # Add count and percentage labels on bars
    for i, row in enumerate(class_dist_pd.itertuples()):
        ax.text(i, row.count/2, f"{row.count:,}\n({row.percentage:.1f}%)", 
                ha='center', va='center', fontweight='bold', color='white')
    
    # 2. Text length distribution by class
    plt.subplot(2, 2, 2)
    sns.boxplot(x='label', y='text_length', data=sample_pd, palette='viridis')
    plt.title('Text Length Distribution by Class', fontsize=14, fontweight='bold')
    plt.xlabel('Class Label (0=Fake, 1=True)')
    plt.ylabel('Text Length (characters)')
    plt.yscale('log')  # Log scale for better visualization
    
    # 3. Text length histogram
    plt.subplot(2, 2, 3)
    sns.histplot(data=sample_pd, x='text_length', bins=30, kde=True)
    plt.title('Text Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    
    # Add vertical lines for quantiles
    colors = ['red', 'orange', 'green', 'orange', 'red']
    labels = ['10%', '25%', 'Median', '75%', '90%']
    
    for q, color, label in zip(quantiles, colors, labels):
        plt.axvline(x=q, color=color, linestyle='--', alpha=0.7)
        plt.text(q, plt.gca().get_ylim()[1]*0.9, f' {label}: {int(q)}', 
                 color=color, fontweight='bold')
    
    # 4. Issues summary
    plt.subplot(2, 2, 4)
    issues = [
        ('Short Texts', short_texts),
        ('Duplicates', duplicate_count)
    ]
    
    # Add null values to issues
    for col, count in null_counts.items():
        issues.append((f'Null in {col}', count))
    
    # Sort issues by count
    issues.sort(key=lambda x: x[1], reverse=True)
    
    # Create bar chart for issues
    if issues:
        issues_df = pd.DataFrame(issues, columns=['Issue', 'Count'])
        issues_df['Percentage'] = issues_df['Count'] / total_count * 100
        
        ax = sns.barplot(x='Count', y='Issue', data=issues_df, palette='Reds_r')
        plt.title('Potential Data Quality Issues', fontsize=14, fontweight='bold')
        plt.xlabel('Count')
        plt.ylabel('Issue Type')
        
        # Add count and percentage labels
        for i, row in enumerate(issues_df.itertuples()):
            ax.text(row.Count + 0.1, i, f"{row.Count:,} ({row.Percentage:.1f}%)", 
                    va='center')
    else:
        plt.text(0.5, 0.5, "No significant issues detected", 
                 ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Display the figure
    plt.show()
    
    # If in Databricks, use display() for interactive visualizations
    if in_databricks:
        try:
            print("\n📊 INTERACTIVE VISUALIZATIONS (DATABRICKS-SPECIFIC):")
            
            # Class distribution
            print("\n1. Class Distribution:")
            display(class_dist)
            
            # Text length statistics
            print("\n2. Text Length Statistics:")
            display(df_with_length.groupBy("label").agg(
                avg("text_length").alias("avg_length"),
                min("text_length").alias("min_length"),
                max("text_length").alias("max_length"),
                count(when(col("text_length") < 10, True)).alias("short_texts")
            ))
            
            # Text length histogram
            print("\n3. Text Length Distribution:")
            display(df_with_length.select("text_length"))
        except:
            print("Note: Some Databricks-specific visualizations couldn't be displayed.")
    
    # Determine dataset quality score
    quality_issues = []
    quality_score = 10  # Start with perfect score
    
    # Check class balance
    if class_balance < 0.2:
        quality_score -= 3
        quality_issues.append(f"Severe class imbalance (ratio: {class_balance:.2f})")
    elif class_balance < 0.5:
        quality_score -= 1
        quality_issues.append(f"Moderate class imbalance (ratio: {class_balance:.2f})")
    
    # Check short texts
    if short_texts_pct > 10:
        quality_score -= 3
        quality_issues.append(f"High percentage of very short texts ({short_texts_pct:.1f}%)")
    elif short_texts_pct > 1:
        quality_score -= 1
        quality_issues.append(f"Some very short texts present ({short_texts_pct:.1f}%)")
    
    # Check duplicates
    if duplicate_pct > 20:
        quality_score -= 3
        quality_issues.append(f"High percentage of duplicate texts ({duplicate_pct:.1f}%)")
    elif duplicate_pct > 5:
        quality_score -= 1
        quality_issues.append(f"Some duplicate texts present ({duplicate_pct:.1f}%)")
    
    # Check null values
    if null_counts:
        total_nulls = sum(null_counts.values())
        null_pct = (total_nulls / (total_count * len(df.columns))) * 100
        
        if null_pct > 10:
            quality_score -= 3
            quality_issues.append(f"High percentage of null values ({null_pct:.1f}%)")
        elif null_pct > 1:
            quality_score -= 1
            quality_issues.append(f"Some null values present ({null_pct:.1f}%)")
    
    # Ensure score is between 0 and 10
    quality_score = max(0, min(10, quality_score))
    
    # Determine quality level
    if quality_score >= 8:
        quality_level = "HIGH"
        quality_color = "🟢"
    elif quality_score >= 5:
        quality_level = "MEDIUM"
        quality_color = "🟠"
    else:
        quality_level = "LOW"
        quality_color = "🔴"
    
    # Print user-friendly summary
    print("\n" + "="*80)
    print(f"{quality_color} DATASET QUALITY ASSESSMENT: {quality_level} (Score: {quality_score}/10)")
    print("="*80)
    
    print("\n📋 SUMMARY:")
    print(f"• Total records: {total_count:,}")
    print(f"• Class distribution: " + ", ".join([f"Class {row.label}: {row.count:,} ({row.percentage:.1f}%)" for _, row in class_dist_pd.iterrows()]))
    print(f"• Class balance ratio: {class_balance:.2f} (minority/majority)")
    print(f"• Text length: avg={avg_length:.1f}, min={min_length}, max={max_length} characters")
    print(f"• Very short texts (<10 chars): {short_texts:,} ({short_texts_pct:.1f}%)")
    print(f"• Duplicate texts: {duplicate_count:,} ({duplicate_pct:.1f}%)")
    
    if null_counts:
        print("• Null values detected in columns: " + 
              ", ".join([f"{col}: {count:,}" for col, count in null_counts.items()]))
    
    if quality_issues:
        print("\n⚠️ QUALITY ISSUES IDENTIFIED:")
        for i, issue in enumerate(quality_issues, 1):
            print(f"  {i}. {issue}")
    
    print("\n🔍 RECOMMENDATIONS:")
    
    if class_balance < 0.5:
        print(f"  • Consider balancing classes using sampling or weighting techniques")
        print(f"    - Majority class ({majority_class}): {class_dist_pd[class_dist_pd['label'] == majority_class]['count'].values[0]:,} records")
        print(f"    - Minority class ({minority_class}): {class_dist_pd[class_dist_pd['label'] == minority_class]['count'].values[0]:,} records")
    
    if short_texts > 0:
        print(f"  • Review and potentially remove very short texts (<10 chars): {short_texts:,} records")
    
    if duplicate_count > 0:
        print(f"  • Consider deduplicating the dataset: {duplicate_count:,} duplicate records")
    
    if null_counts:
        print(f"  • Handle null values in columns: " + 
              ", ".join([f"{col}: {count:,}" for col, count in null_counts.items()]))
    
    print("\n" + "="*80)
    
    # Return analysis results
    results = {
        "total_count": total_count,
        "class_distribution": {row.label: row['count'] for _, row in class_dist_pd.iterrows()},
        "class_balance": class_balance,
        "text_length_stats": {
            "avg": avg_length,
            "min": min_length,
            "max": max_length,
            "quantiles": {
                "10%": quantiles[0],
                "25%": quantiles[1],
                "median": quantiles[2],
                "75%": quantiles[3],
                "90%": quantiles[4]
            }
        },
        "short_texts": {
            "count": short_texts,
            "percentage": short_texts_pct
        },
        "duplicates": {
            "count": duplicate_count,
            "percentage": duplicate_pct
        },
        "null_values": null_counts,
        "quality_score": quality_score,
        "quality_level": quality_level,
        "quality_issues": quality_issues
    }
    
    return results
