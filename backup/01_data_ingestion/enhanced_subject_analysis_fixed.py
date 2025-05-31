def analyze_subject_distribution(fake_df, true_df, max_subjects=10, save_path=None):
    """
    Analyzes the distribution of subjects in fake and true news datasets to detect potential data leakage.
    Provides visual and user-friendly output with clear explanations and recommendations.
    
    Args:
        fake_df: DataFrame with fake news
        true_df: DataFrame with true news
        max_subjects: Maximum number of top subjects to display in charts
        save_path: Optional path to save visualizations (for Databricks environments)
        
    Returns:
        dict: Dictionary with analysis results and recommendations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pyspark.sql.functions import col
    
    print("\n" + "="*80)
    print("📊 SUBJECT DISTRIBUTION ANALYSIS FOR DATA LEAKAGE DETECTION")
    print("="*80)
    
    # Check if subject column exists in both DataFrames
    if "subject" not in fake_df.columns or "subject" not in true_df.columns:
        print("\n⚠️ No 'subject' column found in one or both datasets.")
        return {"has_subject": False}
    
    # Get subject distribution in fake news
    fake_subjects = fake_df.groupBy("subject").count().orderBy("count", ascending=False)
    fake_subjects_pd = fake_subjects.toPandas()
    
    # Get subject distribution in true news
    true_subjects = true_df.groupBy("subject").count().orderBy("count", ascending=False)
    true_subjects_pd = true_subjects.toPandas()
    
    # Calculate total counts
    fake_total = fake_subjects_pd['count'].sum()
    true_total = true_subjects_pd['count'].sum()
    
    # Calculate percentages
    fake_subjects_pd['percentage'] = fake_subjects_pd['count'] / fake_total * 100
    true_subjects_pd['percentage'] = true_subjects_pd['count'] / true_total * 100
    
    # Get top subjects
    top_fake_subjects = fake_subjects_pd.head(max_subjects)
    top_true_subjects = true_subjects_pd.head(max_subjects)
    
    # Clean up subject names for display
    def clean_subject_name(subject, max_length=20):
        if subject is None:
            return "null"
        subject_str = str(subject)
        if len(subject_str) > max_length:
            return subject_str[:max_length] + '...'
        return subject_str
    
    # Apply cleaning to subject names
    top_fake_subjects_display = top_fake_subjects.copy()
    top_fake_subjects_display['subject_display'] = top_fake_subjects_display['subject'].apply(clean_subject_name)
    
    top_true_subjects_display = top_true_subjects.copy()
    top_true_subjects_display['subject_display'] = top_true_subjects_display['subject'].apply(clean_subject_name)
    
    # Find common subjects between top fake and true news
    common_subjects = set(top_fake_subjects['subject']).intersection(set(top_true_subjects['subject']))
    
    # Calculate overlap metrics
    fake_top_coverage = top_fake_subjects['count'].sum() / fake_total * 100
    true_top_coverage = top_true_subjects['count'].sum() / true_total * 100
    
    # Create a combined dataframe for comparison
    combined_data = []
    
    # Add fake news subjects
    for _, row in top_fake_subjects_display.iterrows():
        combined_data.append({
            'subject': row['subject_display'],
            'count': row['count'],
            'percentage': row['percentage'],
            'type': 'Fake News'
        })
    
    # Add true news subjects
    for _, row in top_true_subjects_display.iterrows():
        combined_data.append({
            'subject': row['subject_display'],
            'count': row['count'],
            'percentage': row['percentage'],
            'type': 'True News'
        })
    
    combined_df = pd.DataFrame(combined_data)
    
    # Assess data leakage risk
    # Calculate separation score (how well subjects separate fake from true news)
    separation_score = 0
    evidence = []
    
    # Check if top subjects are exclusive to one class
    fake_exclusive = set(top_fake_subjects['subject']) - set(true_subjects_pd['subject'])
    true_exclusive = set(top_true_subjects['subject']) - set(fake_subjects_pd['subject'])
    
    # Calculate percentage of data covered by exclusive subjects
    fake_exclusive_coverage = fake_subjects_pd[fake_subjects_pd['subject'].isin(fake_exclusive)]['count'].sum() / fake_total * 100
    true_exclusive_coverage = true_subjects_pd[true_subjects_pd['subject'].isin(true_exclusive)]['count'].sum() / true_total * 100
    
    # If top subjects have little overlap and cover most of the data, high separation score
    if len(common_subjects) < 2 and (fake_exclusive_coverage > 70 or true_exclusive_coverage > 70):
        separation_score = 10  # Maximum risk
        evidence.append(f"Top subjects have almost no overlap between fake and true news")
        evidence.append(f"{len(fake_exclusive)} subjects are exclusive to fake news, covering {fake_exclusive_coverage:.1f}% of fake news")
        evidence.append(f"{len(true_exclusive)} subjects are exclusive to true news, covering {true_exclusive_coverage:.1f}% of true news")
    elif len(common_subjects) < max_subjects // 2:
        separation_score = 8
        evidence.append(f"Limited overlap in top subjects between fake and true news")
        evidence.append(f"Only {len(common_subjects)} subjects appear in both top fake and true news lists")
    else:
        # Check distribution within common subjects
        for subject in common_subjects:
            fake_pct = fake_subjects_pd[fake_subjects_pd['subject'] == subject]['percentage'].values[0]
            true_pct = true_subjects_pd[true_subjects_pd['subject'] == subject]['percentage'].values[0]
            
            # If the same subject has very different percentages, increase separation score
            if abs(fake_pct - true_pct) > 20:
                separation_score += 1
                evidence.append(f"Subject '{clean_subject_name(subject)}' has significantly different distribution: {fake_pct:.1f}% in fake vs {true_pct:.1f}% in true news")
    
    # Cap the score at 10
    separation_score = min(separation_score, 10)
    
    # Create visualizations with adjusted layout
    plt.figure(figsize=(20, 16))  # Increased figure size
    
    # Adjust subplot parameters
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. Top subjects in fake news
    plt.subplot(2, 2, 1)
    sns.barplot(x='percentage', y='subject_display', data=top_fake_subjects_display.head(10), palette='Reds_r')
    plt.title('Top Subjects in Fake News', fontsize=16, fontweight='bold')
    plt.xlabel('Percentage (%)', fontsize=12)
    plt.ylabel('Subject', fontsize=12)
    
    # 2. Top subjects in true news
    plt.subplot(2, 2, 2)
    sns.barplot(x='percentage', y='subject_display', data=top_true_subjects_display.head(10), palette='Blues_r')
    plt.title('Top Subjects in True News', fontsize=16, fontweight='bold')
    plt.xlabel('Percentage (%)', fontsize=12)
    plt.ylabel('Subject', fontsize=12)
    
    # 3. Comparison of top subjects - use a separate figure to avoid layout issues
    plt.subplot(2, 1, 2)
    
    # Create a simpler comparison plot that's less likely to have layout issues
    comparison_data = []
    
    # Get subjects that appear in either top list
    all_top_subjects = set(top_fake_subjects_display['subject_display']).union(set(top_true_subjects_display['subject_display']))
    top_subjects_list = list(all_top_subjects)[:10]  # Limit to 10 subjects
    
    # For each subject, get percentages in fake and true news
    for subject_display in top_subjects_list:
        # Find original subject
        fake_match = top_fake_subjects_display[top_fake_subjects_display['subject_display'] == subject_display]
        true_match = top_true_subjects_display[top_true_subjects_display['subject_display'] == subject_display]
        
        fake_pct = fake_match['percentage'].values[0] if not fake_match.empty else 0
        true_pct = true_match['percentage'].values[0] if not true_match.empty else 0
        
        comparison_data.append({
            'subject': subject_display,
            'Fake News': fake_pct,
            'True News': true_pct
        })
    
    # Convert to DataFrame and sort
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(by='Fake News', ascending=False)
    
    # Plot as grouped bar chart
    comparison_melted = pd.melt(comparison_df, id_vars=['subject'], var_name='News Type', value_name='Percentage')
    sns.barplot(x='Percentage', y='subject', hue='News Type', data=comparison_melted, palette=['red', 'blue'])
    
    plt.title('Subject Distribution Comparison (Fake vs True News)', fontsize=16, fontweight='bold')
    plt.xlabel('Percentage (%)', fontsize=12)
    plt.ylabel('Subject', fontsize=12)
    plt.legend(title='News Type')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # Display the figure
    plt.tight_layout(pad=3.0)  # Increased padding
    plt.show()
    
    # Determine risk level based on separation score
    if separation_score >= 8:
        risk_level = "HIGH"
        risk_color = "🔴"
        recommendation = "The 'subject' column MUST be removed before model training to prevent data leakage."
    elif separation_score >= 5:
        risk_level = "MEDIUM"
        risk_color = "🟠"
        recommendation = "Consider removing the 'subject' column or use careful cross-validation to assess its impact."
    else:
        risk_level = "LOW"
        risk_color = "🟢"
        recommendation = "The 'subject' column appears to have limited predictive power for separating classes."
    
    # Print user-friendly summary
    print("\n" + "="*80)
    print(f"{risk_color} DATA LEAKAGE RISK ASSESSMENT: {risk_level} (Score: {separation_score}/10)")
    print("="*80)
    
    print("\n📋 SUMMARY:")
    print(f"• Analyzed {len(fake_subjects_pd)} unique subjects in fake news and {len(true_subjects_pd)} in true news")
    print(f"• Top {max_subjects} subjects cover {fake_top_coverage:.1f}% of fake news and {true_top_coverage:.1f}% of true news")
    print(f"• Found {len(common_subjects)} common subjects among top {max_subjects} in both categories")
    print(f"• {len(fake_exclusive)} subjects appear exclusively in fake news (covering {fake_exclusive_coverage:.1f}%)")
    print(f"• {len(true_exclusive)} subjects appear exclusively in true news (covering {true_exclusive_coverage:.1f}%)")
    
    print("\n🔍 EVIDENCE OF DATA LEAKAGE:")
    for i, item in enumerate(evidence, 1):
        print(f"  {i}. {item}")
    
    print("\n⚠️ RECOMMENDATION:")
    print(f"  {recommendation}")
    
    if risk_level == "HIGH":
        print("\n🛑 ACTION REQUIRED:")
        print("  df = df.drop('subject')  # Remove the subject column to prevent data leakage")
    
    print("\n" + "="*80)
    
    # Return analysis results
    results = {
        "has_subject": True,
        "risk_level": risk_level,
        "risk_score": separation_score,
        "recommendation": recommendation,
        "evidence": evidence,
        "fake_subjects_count": len(fake_subjects_pd),
        "true_subjects_count": len(true_subjects_pd),
        "common_subjects": list(common_subjects),
        "fake_exclusive": list(fake_exclusive),
        "true_exclusive": list(true_exclusive),
        "fake_top_coverage": fake_top_coverage,
        "true_top_coverage": true_top_coverage
    }
    
    return results
