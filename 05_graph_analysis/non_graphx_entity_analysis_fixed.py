"""
Script to implement non-GraphX NER-based entity extraction and feature engineering for fake news detection.
This script uses traditional NLP techniques to extract and analyze entities and create features.
"""

import pandas as pd
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import spacy
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Start timer
start_time = time.time()

# Define paths
data_dir = "/home/ubuntu/fake_news_detection/data"
results_dir = "/home/ubuntu/fake_news_detection/logs"
config_dir = "/home/ubuntu/fake_news_detection/config"

# Create directories if they don't exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)

# Configuration parameters
config = {
    "min_entity_freq": 2,  # Minimum frequency for entity to be included in analysis
    "top_n_entities": 20,  # Number of top entities to display in visualizations
    "n_topics": 5,         # Number of topics to extract
    "n_features": 100      # Number of features to use for topic modeling
}

# Save configuration
with open(f"{config_dir}/non_graphx_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Loading data...")
# Try to load the NER-enhanced dataset
try:
    # First try to read the NER-enhanced dataset
    df = pd.read_csv(f"{data_dir}/news_sample_ner_enhanced.csv")
    print(f"Loaded NER-enhanced dataset with {len(df)} records")
except FileNotFoundError:
    try:
        # Fall back to metadata-enhanced dataset
        df = pd.read_csv(f"{data_dir}/news_sample_enhanced.csv")
        print(f"NER-enhanced dataset not found, loaded metadata-enhanced dataset with {len(df)} records")
    except FileNotFoundError:
        # Fall back to original sample
        df = pd.read_csv(f"{data_dir}/news_sample.csv")
        print(f"Enhanced datasets not found, loaded original sample with {len(df)} records")
        # Add empty entity columns
        df['people'] = df.apply(lambda x: [], axis=1)
        df['places'] = df.apply(lambda x: [], axis=1)
        df['organizations'] = df.apply(lambda x: [], axis=1)
        df['events'] = df.apply(lambda x: [], axis=1)

# Convert string representations of lists to actual lists if needed
for col_name in ['people', 'places', 'organizations', 'event_types']:
    if col_name in df.columns:
        if df[col_name].dtype == 'object' and isinstance(df[col_name].iloc[0], str):
            try:
                df[col_name] = df[col_name].apply(lambda x: eval(x) if isinstance(x, str) else x)
            except:
                print(f"Warning: Could not convert {col_name} to list, using empty lists")
                df[col_name] = df[col_name].apply(lambda x: [])

# Ensure text column is available
if 'text' not in df.columns and 'content' in df.columns:
    df['text'] = df['content']

# Download required NLTK data
print("Setting up NLP tools...")
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except:
    print("NLTK download failed, but continuing...")
    stop_words = set()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("Loaded spaCy model")
except:
    print("Installing spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model after installation")
    except:
        print("Failed to load spaCy model, using pattern-based extraction only")
        nlp = None

# Function to extract entities if not already present
def extract_entities(text):
    """Extract named entities using spaCy."""
    if pd.isna(text) or nlp is None:
        return [], [], [], []
    
    people = []
    places = []
    organizations = []
    dates = []
    
    # Process text with spaCy
    doc = nlp(text[:100000])  # Limit text length to avoid memory issues
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            places.append(ent.text)
        elif ent.label_ == "ORG":
            organizations.append(ent.text)
        elif ent.label_ in ["DATE", "TIME"]:
            dates.append(ent.text)
    
    return people, places, organizations, dates

# Apply entity extraction if needed
if 'people' not in df.columns or len(df['people'].iloc[0]) == 0:
    print("Extracting entities...")
    # Process in batches to avoid memory issues
    batch_size = 50
    all_people = []
    all_places = []
    all_orgs = []
    all_dates = []
    
    for i in range(0, len(df), batch_size):
        batch = df['text'].iloc[i:i+batch_size]
        batch_results = [extract_entities(text) for text in batch]
        
        batch_people = [res[0] for res in batch_results]
        batch_places = [res[1] for res in batch_results]
        batch_orgs = [res[2] for res in batch_results]
        batch_dates = [res[3] for res in batch_results]
        
        all_people.extend(batch_people)
        all_places.extend(batch_places)
        all_orgs.extend(batch_orgs)
        all_dates.extend(batch_dates)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
    
    df['people'] = all_people
    df['places'] = all_places
    df['organizations'] = all_orgs
    df['dates'] = all_dates

# Count entities per document
df['people_count'] = df['people'].apply(len)
df['places_count'] = df['places'].apply(len)
df['org_count'] = df['organizations'].apply(len)
if 'dates' in df.columns:
    df['date_count'] = df['dates'].apply(len)
if 'event_types' in df.columns:
    df['event_count'] = df['event_types'].apply(len)

# Analyze entity distributions
print("Analyzing entity distributions...")

# Function to count and analyze entities
def analyze_entities(entity_lists, min_freq=2):
    """Count and analyze entities across documents."""
    all_entities = []
    for entities in entity_lists:
        all_entities.extend(entities)
    
    # Count entities using Counter
    entity_counts = Counter(all_entities)
    
    # Filter by minimum frequency
    filtered_counts = Counter({k: v for k, v in entity_counts.items() if v >= min_freq})
    
    return filtered_counts

# Analyze entities by type
people_counts = analyze_entities(df['people'], config["min_entity_freq"])
place_counts = analyze_entities(df['places'], config["min_entity_freq"])
org_counts = analyze_entities(df['organizations'], config["min_entity_freq"])
if 'event_types' in df.columns:
    event_counts = Counter([event for events in df['event_types'] for event in events if isinstance(events, list)])
else:
    event_counts = Counter()

print(f"Found {len(people_counts)} unique people mentioned at least {config['min_entity_freq']} times")
print(f"Found {len(place_counts)} unique places mentioned at least {config['min_entity_freq']} times")
print(f"Found {len(org_counts)} unique organizations mentioned at least {config['min_entity_freq']} times")
print(f"Found {len(event_counts)} unique event types")

# Analyze entity distribution by label (fake vs real)
def analyze_entity_by_label(df, entity_col):
    """Analyze entity distribution by label."""
    fake_entities = []
    real_entities = []
    
    for i, row in df.iterrows():
        if row['label'] == 0:  # Fake
            fake_entities.extend(row[entity_col])
        else:  # Real
            real_entities.extend(row[entity_col])
    
    fake_counts = Counter(fake_entities)
    real_counts = Counter(real_entities)
    
    return fake_counts, real_counts

# Analyze entities by label
people_fake, people_real = analyze_entity_by_label(df, 'people')
places_fake, places_real = analyze_entity_by_label(df, 'places')
orgs_fake, orgs_real = analyze_entity_by_label(df, 'organizations')

# Analyze events by label if available
if 'event_types' in df.columns:
    events_fake = []
    events_real = []
    for i, row in df.iterrows():
        if isinstance(row['event_types'], list):
            if row['label'] == 0:  # Fake
                events_fake.extend(row['event_types'])
            else:  # Real
                events_real.extend(row['event_types'])

    events_fake_counts = Counter(events_fake)
    events_real_counts = Counter(events_real)
else:
    events_fake_counts = Counter()
    events_real_counts = Counter()

# Create visualizations
print("Creating visualizations...")

# Function to create entity comparison chart
def create_entity_comparison(fake_counts, real_counts, entity_type, top_n=10):
    """Create comparison chart for entity distribution in fake vs real news."""
    # Get top entities by total count
    combined = Counter()
    for k, v in fake_counts.items():
        combined[k] += v
    for k, v in real_counts.items():
        combined[k] += v
    
    top_entities = [entity for entity, count in combined.most_common(top_n)]
    
    # Create dataframe for plotting
    plot_data = []
    for entity in top_entities:
        fake_count = fake_counts.get(entity, 0)
        real_count = real_counts.get(entity, 0)
        total = fake_count + real_count
        if total > 0:
            fake_ratio = fake_count / total
            real_ratio = real_count / total
            plot_data.append({
                'Entity': entity,
                'Fake': fake_count,
                'Real': real_count,
                'Fake_Ratio': fake_ratio,
                'Real_Ratio': real_ratio
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create stacked bar chart
    plt.figure(figsize=(12, 8))
    plot_df[['Fake', 'Real']].plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title(f'Distribution of {entity_type} in Fake and Real News')
    plt.xlabel(entity_type)
    plt.ylabel('Count')
    plt.xticks(range(len(plot_df)), plot_df['Entity'], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/non_graphx_{entity_type.lower()}_distribution.png")
    
    return plot_df

# Create entity comparison charts
if people_counts:
    people_comparison = create_entity_comparison(people_fake, people_real, 'People', config["top_n_entities"])
if place_counts:
    places_comparison = create_entity_comparison(places_fake, places_real, 'Places', config["top_n_entities"])
if org_counts:
    orgs_comparison = create_entity_comparison(orgs_fake, orgs_real, 'Organizations', config["top_n_entities"])
if event_counts:
    events_comparison = create_entity_comparison(events_fake_counts, events_real_counts, 'Events', config["top_n_entities"])

# Create entity count comparison by article type
plt.figure(figsize=(10, 6))
entity_counts = df.groupby('label').agg({
    'people_count': 'mean',
    'places_count': 'mean',
    'org_count': 'mean'
}).reset_index()
if 'event_count' in df.columns:
    entity_counts['event_count'] = df.groupby('label')['event_count'].mean().values
entity_counts['label'] = entity_counts['label'].map({0: 'Fake', 1: 'Real'})
entity_counts = entity_counts.melt(id_vars=['label'], var_name='Entity Type', value_name='Average Count')
entity_counts['Entity Type'] = entity_counts['Entity Type'].str.replace('_count', '')

sns.barplot(x='Entity Type', y='Average Count', hue='label', data=entity_counts)
plt.title('Average Entity Counts in Fake vs Real News')
plt.tight_layout()
plt.savefig(f"{results_dir}/non_graphx_avg_entity_counts.png")

# Create dashboard visualization
plt.figure(figsize=(15, 12))

# Entity count comparison
plt.subplot(2, 2, 1)
sns.barplot(x='Entity Type', y='Average Count', hue='label', data=entity_counts)
plt.title('Average Entity Counts')
plt.legend(title='')

# Top people comparison
if people_counts:
    plt.subplot(2, 2, 2)
    top_5_people = people_comparison.sort_values('Fake', ascending=False).head(5)
    top_5_people[['Fake', 'Real']].plot(kind='bar', stacked=True)
    plt.title('Top 5 People Mentioned')
    plt.xticks(range(len(top_5_people)), top_5_people['Entity'], rotation=45, ha='right')
    plt.legend(title='')

# Top places comparison
if place_counts:
    plt.subplot(2, 2, 3)
    top_5_places = places_comparison.sort_values('Fake', ascending=False).head(5)
    top_5_places[['Fake', 'Real']].plot(kind='bar', stacked=True)
    plt.title('Top 5 Places Mentioned')
    plt.xticks(range(len(top_5_places)), top_5_places['Entity'], rotation=45, ha='right')
    plt.legend(title='')

# Top events comparison
if event_counts:
    plt.subplot(2, 2, 4)
    top_5_events = events_comparison.sort_values('Fake', ascending=False).head(5)
    top_5_events[['Fake', 'Real']].plot(kind='bar', stacked=True)
    plt.title('Top 5 Events Mentioned')
    plt.xticks(range(len(top_5_events)), top_5_events['Entity'], rotation=45, ha='right')
    plt.legend(title='')

plt.tight_layout()
plt.savefig(f"{results_dir}/non_graphx_entity_dashboard.png")

# Topic modeling with TF-IDF and SVD
print("Performing topic modeling...")
# Prepare text data
texts = df['text'].fillna('').tolist()

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(
    max_features=config["n_features"],
    stop_words='english',
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Perform topic modeling with SVD (similar to LSA)
svd = TruncatedSVD(n_components=config["n_topics"])
topic_matrix = svd.fit_transform(tfidf_matrix)

# Get top terms for each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
topics = []
for i, comp in enumerate(svd.components_):
    terms_idx = comp.argsort()[:-11:-1]  # Get indices of top 10 terms
    terms = [feature_names[idx] for idx in terms_idx]
    topics.append(terms)

print("Topics extracted:")
for i, topic_terms in enumerate(topics):
    print(f"Topic {i}: {', '.join(topic_terms)}")

# Assign dominant topic to each document
df['dominant_topic'] = topic_matrix.argmax(axis=1)

# Analyze topic distribution by label
topic_distribution = pd.crosstab(df['dominant_topic'], df['label'], rownames=['dominant_topic'], colnames=['label'])
topic_distribution.columns = ['Fake', 'Real']
topic_distribution['Total'] = topic_distribution['Fake'] + topic_distribution['Real']
topic_distribution['Fake_Ratio'] = topic_distribution['Fake'] / topic_distribution['Total']
topic_distribution['Real_Ratio'] = topic_distribution['Real'] / topic_distribution['Total']

print("Topic distribution by label:")
print(topic_distribution)

# Visualize topic distribution
plt.figure(figsize=(12, 6))
topic_distribution[['Fake', 'Real']].plot(kind='bar', stacked=True)
plt.title('Topic Distribution in Fake vs Real News')
plt.xlabel('Topic')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{results_dir}/non_graphx_topic_distribution.png")

# Create features for machine learning
print("Creating non-GraphX features for machine learning...")

# Entity-based features
def create_entity_features(row):
    features = {}
    
    # Entity count features
    features['person_count'] = len(row['people']) if 'people' in row and row['people'] else 0
    features['place_count'] = len(row['places']) if 'places' in row and row['places'] else 0
    features['org_count'] = len(row['organizations']) if 'organizations' in row and row['organizations'] else 0
    if 'event_types' in row and row['event_types']:
        features['event_count'] = len(row['event_types'])
    else:
        features['event_count'] = 0
    
    # Entity presence features (for top entities)
    top_people = [entity for entity, _ in people_counts.most_common(10)]
    for person in top_people:
        features[f'has_person_{person.replace(" ", "_")}'] = 1 if person in row['people'] else 0
        
    top_places = [entity for entity, _ in place_counts.most_common(10)]
    for place in top_places:
        features[f'has_place_{place.replace(" ", "_")}'] = 1 if place in row['places'] else 0
        
    top_orgs = [entity for entity, _ in org_counts.most_common(10)]
    for org in top_orgs:
        features[f'has_org_{org.replace(" ", "_")}'] = 1 if org in row['organizations'] else 0
    
    # Topic features
    for i in range(config["n_topics"]):
        features[f'topic_{i}_score'] = topic_matrix[row.name, i]
    
    # Text statistics features
    if 'text' in row and not pd.isna(row['text']):
        text = row['text']
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        # Use simple splitting instead of sent_tokenize to avoid NLTK punkt_tab dependency
        features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        features['sentence_count'] = max(1, features['sentence_count'])  # Ensure at least 1 sentence
        features['avg_word_length'] = sum(len(word) for word in text.split()) / max(1, len(text.split()))
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    else:
        features['text_length'] = 0
        features['word_count'] = 0
        features['sentence_count'] = 0
        features['avg_word_length'] = 0
        features['avg_sentence_length'] = 0
    
    return features

# Apply feature creation
non_graphx_features = []
for i, row in df.iterrows():
    features = create_entity_features(row)
    non_graphx_features.append(features)

# Convert to DataFrame
non_graphx_features_df = pd.DataFrame(non_graphx_features)

# Add to original DataFrame
for col in non_graphx_features_df.columns:
    df[f'non_graphx_{col}'] = non_graphx_features_df[col]

# Save enhanced dataset with non-GraphX features
df.to_csv(f"{data_dir}/news_sample_non_graphx_enhanced.csv", index=False)

# Save non-GraphX analysis results
non_graphx_results = {
    "entities": {
        "people_count": len(people_counts),
        "places_count": len(place_counts),
        "organizations_count": len(org_counts),
        "events_count": len(event_counts)
    },
    "top_people": dict(people_counts.most_common(config["top_n_entities"])),
    "top_places": dict(place_counts.most_common(config["top_n_entities"])),
    "top_organizations": dict(org_counts.most_common(config["top_n_entities"])),
    "topics": {i: topics[i] for i in range(len(topics))},
    "topic_distribution": topic_distribution.to_dict(),
    "execution_time": time.time() - start_time
}

with open(f"{results_dir}/non_graphx_analysis.json", "w") as f:
    json.dump(non_graphx_results, f, indent=2)

print(f"\nNon-GraphX entity analysis completed in {time.time() - start_time:.2f} seconds")
print(f"Enhanced dataset saved to {data_dir}/news_sample_non_graphx_enhanced.csv")
print(f"Results saved to {results_dir}")

# Last modified: May 29, 2025
