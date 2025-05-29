"""
Script to extract and analyze topic and source metadata from news articles.
"""

import pandas as pd
import numpy as np
import os
import time
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Start timer
start_time = time.time()

# Define paths
data_dir = "/home/ubuntu/fake_news_detection/data"
results_dir = "/home/ubuntu/fake_news_detection/logs"

# Create directories if they don't exist
os.makedirs(results_dir, exist_ok=True)

print("Loading data...")
# Load the sampled data
df = pd.read_csv(f"{data_dir}/news_sample.csv")

# Function to extract source and location from text
def extract_metadata(text):
    """Extract source and location information from text."""
    if pd.isna(text):
        return None, None
    
    # Common news sources
    sources = [
        'Reuters', 'AP', 'Associated Press', 'CNN', 'Fox News', 'MSNBC', 'BBC', 
        'New York Times', 'Washington Post', 'USA Today', 'NPR', 'CBS', 'NBC', 
        'ABC News', 'The Guardian', 'Bloomberg', 'Wall Street Journal', 'WSJ',
        'Huffington Post', 'Breitbart', 'BuzzFeed', 'Daily Mail', 'The Hill'
    ]
    
    # Try to find source at the end of the text with pattern "- Source"
    source_match = re.search(r'-\s*([^-\n]+?)$', text)
    if source_match:
        potential_source = source_match.group(1).strip()
        # Check if the extracted text contains a known source
        for known_source in sources:
            if known_source.lower() in potential_source.lower():
                return known_source, None
    
    # Try to find source in the text
    for source in sources:
        if source.lower() in text.lower():
            return source, None
    
    # If no source found
    return None, None

# Apply extraction to the dataset
print("Extracting metadata...")
metadata = df['text'].apply(extract_metadata)
df['source'] = metadata.apply(lambda x: x[0])
df['location'] = metadata.apply(lambda x: x[1])

# Count sources
print("Analyzing sources...")
source_counts = df['source'].value_counts()
print(f"Found {len(source_counts)} unique sources")
print(source_counts.head(10))

# Analyze source distribution by label
source_by_label = pd.crosstab(df['source'], df['label'])
source_by_label.columns = ['Fake', 'Real']
source_by_label['Total'] = source_by_label['Fake'] + source_by_label['Real']
source_by_label['Fake_Ratio'] = source_by_label['Fake'] / source_by_label['Total']
source_by_label['Real_Ratio'] = source_by_label['Real'] / source_by_label['Total']

# Sort by total count
source_by_label = source_by_label.sort_values('Total', ascending=False)
print("\nSource distribution by label:")
print(source_by_label.head(10))

# Extract topics using NLP techniques
print("\nExtracting topics...")
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    
    # Use CountVectorizer for topic modeling
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=1000)
    doc_term_matrix = count_vectorizer.fit_transform(df['text'].fillna(''))
    
    # LDA for topic modeling
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(doc_term_matrix)
    
    # Get feature names
    feature_names = count_vectorizer.get_feature_names_out()
    
    # Extract top words for each topic
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics[f"Topic {topic_idx}"] = top_words
        print(f"Topic {topic_idx}: {', '.join(top_words)}")
    
    # Get dominant topic for each document
    doc_topics = lda.transform(doc_term_matrix)
    df['dominant_topic'] = np.argmax(doc_topics, axis=1)
    
    # Analyze topic distribution by label
    topic_by_label = pd.crosstab(df['dominant_topic'], df['label'])
    topic_by_label.columns = ['Fake', 'Real']
    topic_by_label['Total'] = topic_by_label['Fake'] + topic_by_label['Real']
    topic_by_label['Fake_Ratio'] = topic_by_label['Fake'] / topic_by_label['Total']
    topic_by_label['Real_Ratio'] = topic_by_label['Real'] / topic_by_label['Total']
    
    print("\nTopic distribution by label:")
    print(topic_by_label)
    
    # Create topic names based on top words
    topic_names = {i: f"Topic {i}: {', '.join(topics[f'Topic {i}'][:3])}" for i in range(len(topics))}
    
    # Replace topic numbers with names
    df['topic_name'] = df['dominant_topic'].map(topic_names)
    
except Exception as e:
    print(f"Error in topic extraction: {e}")
    df['dominant_topic'] = None
    df['topic_name'] = None
    topics = {}
    topic_by_label = pd.DataFrame()

# Visualize source distribution
print("\nCreating visualizations...")
plt.figure(figsize=(12, 8))
top_sources = source_counts.head(10).index
source_data = source_by_label.loc[source_by_label.index.isin(top_sources)]

# Create stacked bar chart
source_data[['Fake', 'Real']].plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Distribution of Fake and Real News by Source')
plt.xlabel('Source')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{results_dir}/source_distribution.png")

# Visualize topic distribution
if not topic_by_label.empty:
    plt.figure(figsize=(12, 8))
    topic_by_label[['Fake', 'Real']].plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Distribution of Fake and Real News by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/topic_distribution.png")

# Create heatmap of source vs topic
if 'dominant_topic' in df.columns and df['dominant_topic'].notna().any():
    plt.figure(figsize=(14, 10))
    # Filter to include only top sources
    filtered_df = df[df['source'].isin(top_sources)]
    if not filtered_df.empty:
        heatmap_data = pd.crosstab(filtered_df['source'], filtered_df['dominant_topic'])
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Source vs Topic Distribution')
        plt.xlabel('Topic')
        plt.ylabel('Source')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/source_topic_heatmap.png")

# Create dashboard visualization
plt.figure(figsize=(15, 12))

# Source distribution
plt.subplot(2, 2, 1)
source_counts.head(5).plot(kind='bar')
plt.title('Top 5 Sources')
plt.xlabel('Source')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Topic distribution
if 'dominant_topic' in df.columns and df['dominant_topic'].notna().any():
    plt.subplot(2, 2, 2)
    df['dominant_topic'].value_counts().sort_index().plot(kind='bar')
    plt.title('Documents per Topic')
    plt.xlabel('Topic')
    plt.ylabel('Count')

# Fake vs Real by source
plt.subplot(2, 2, 3)
if not source_data.empty:
    source_data['Fake_Ratio'].plot(kind='bar', color='red', alpha=0.7)
    plt.title('Fake News Ratio by Source')
    plt.xlabel('Source')
    plt.ylabel('Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)

# Fake vs Real by topic
if not topic_by_label.empty:
    plt.subplot(2, 2, 4)
    topic_by_label['Fake_Ratio'].plot(kind='bar', color='red', alpha=0.7)
    plt.title('Fake News Ratio by Topic')
    plt.xlabel('Topic')
    plt.ylabel('Ratio')
    plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(f"{results_dir}/metadata_dashboard.png")

# Save enhanced dataset
df.to_csv(f"{data_dir}/news_sample_enhanced.csv", index=False)

# Save metadata analysis results
metadata_results = {
    "sources": {
        "unique_count": len(source_counts),
        "top_sources": source_counts.head(10).to_dict(),
        "source_by_label": source_by_label.head(10).to_dict()
    },
    "topics": {
        "topic_words": topics,
        "topic_by_label": topic_by_label.to_dict() if not topic_by_label.empty else {}
    },
    "execution_time": time.time() - start_time
}

with open(f"{results_dir}/metadata_analysis.json", "w") as f:
    json.dump(metadata_results, f, indent=2)

print(f"\nMetadata extraction and analysis completed in {time.time() - start_time:.2f} seconds")
print(f"Enhanced dataset saved to {data_dir}/news_sample_enhanced.csv")
print(f"Results saved to {results_dir}")
