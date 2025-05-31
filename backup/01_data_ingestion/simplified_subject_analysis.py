def analyze_subject_distribution(fake_df, true_df):
    """
    Analyzes the distribution of subjects in fake and true news datasets to detect potential data leakage.
    Provides essential checks using native PySpark functionality.
    Optimized for Databricks environment with display() visualizations.
    
    Args:
        fake_df: DataFrame with fake news
        true_df: DataFrame with true news
        
    Returns:
        None
    """
    from pyspark.sql.functions import col, count
    
    print("\n" + "="*80)
    print("📊 SUBJECT DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Check if subject column exists in both DataFrames
    if "subject" not in fake_df.columns or "subject" not in true_df.columns:
        print("\n⚠️ No 'subject' column found in one or both datasets.")
        return
    
    # Get subject distribution in fake news
    print("\n1️⃣ FAKE NEWS SUBJECT DISTRIBUTION")
    fake_subjects = fake_df.groupBy("subject").count().orderBy(col("count").desc())
    
    # Display fake news subject distribution
    print("• Subject distribution in fake news:")
    display(fake_subjects)
    
    # Get subject distribution in true news
    print("\n2️⃣ TRUE NEWS SUBJECT DISTRIBUTION")
    true_subjects = true_df.groupBy("subject").count().orderBy(col("count").desc())
    
    # Display true news subject distribution
    print("• Subject distribution in true news:")
    display(true_subjects)
    
    # Calculate total counts
    fake_total = fake_df.count()
    true_total = true_df.count()
    
    # Get unique subjects in each dataset
    fake_subjects_list = [row["subject"] for row in fake_subjects.collect()]
    true_subjects_list = [row["subject"] for row in true_subjects.collect()]
    
    # Find common and exclusive subjects
    common_subjects = set(fake_subjects_list).intersection(set(true_subjects_list))
    fake_exclusive = set(fake_subjects_list) - set(true_subjects_list)
    true_exclusive = set(true_subjects_list) - set(fake_subjects_list)
    
    # Display subject overlap analysis
    print("\n3️⃣ SUBJECT OVERLAP ANALYSIS")
    print(f"• Total unique subjects in fake news: {len(fake_subjects_list)}")
    print(f"• Total unique subjects in true news: {len(true_subjects_list)}")
    print(f"• Subjects common to both datasets: {len(common_subjects)}")
    print(f"• Subjects exclusive to fake news: {len(fake_exclusive)}")
    print(f"• Subjects exclusive to true news: {len(true_exclusive)}")
    
    # Create a comparison view for common subjects
    if common_subjects:
        print("\n• Distribution of common subjects:")
        
        # Create temporary views for SQL query
        fake_df.createOrReplaceTempView("fake_news")
        true_df.createOrReplaceTempView("true_news")
        
        # SQL query to compare subject distributions
        comparison_query = """
        SELECT 
            f.subject,
            f.count as fake_count,
            t.count as true_count,
            f.count / {0} * 100 as fake_percentage,
            t.count / {1} * 100 as true_percentage
        FROM 
            (SELECT subject, COUNT(*) as count FROM fake_news GROUP BY subject) f
        JOIN 
            (SELECT subject, COUNT(*) as count FROM true_news GROUP BY subject) t
        ON 
            f.subject = t.subject
        ORDER BY 
            (f.count / {0} * 100) - (t.count / {1} * 100) DESC
        """.format(fake_total, true_total)
        
        comparison_df = spark.sql(comparison_query)
        display(comparison_df)
    
    # Data leakage assessment
    print("\n4️⃣ DATA LEAKAGE ASSESSMENT")
    
    # Check for perfect separation
    perfect_separation = len(common_subjects) == 0 and len(fake_exclusive) > 0 and len(true_exclusive) > 0
    
    if perfect_separation:
        print("\n🔴 HIGH RISK OF DATA LEAKAGE DETECTED")
        print("• The 'subject' column perfectly separates fake from true news")
        print("• No overlap between subjects in fake and true news")
        print("• This column should be removed before model training")
    elif len(common_subjects) < min(len(fake_subjects_list), len(true_subjects_list)) * 0.2:
        print("\n🟠 MODERATE RISK OF DATA LEAKAGE DETECTED")
        print("• The 'subject' column strongly separates fake from true news")
        print(f"• Only {len(common_subjects)} subjects appear in both datasets")
        print("• Consider removing this column before model training")
    else:
        print("\n🟢 LOW RISK OF DATA LEAKAGE DETECTED")
        print("• The 'subject' column shows significant overlap between fake and true news")
        print(f"• {len(common_subjects)} subjects appear in both datasets")
        print("• This column may be safe to use, but monitor its impact on model performance")
    
    # Recommendation
    print("\n⚠️ RECOMMENDATION:")
    if perfect_separation or len(common_subjects) < min(len(fake_subjects_list), len(true_subjects_list)) * 0.2:
        print("  df = df.drop('subject')  # Remove the subject column to prevent data leakage")
    else:
        print("  # Monitor the impact of the 'subject' column on model performance")
        print("  # Consider feature importance analysis to assess its contribution")
    
    print("\n" + "="*80)
