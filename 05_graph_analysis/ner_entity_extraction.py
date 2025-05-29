"""
Script to implement NER-based entity extraction for enhanced metadata analysis.
Extracts people, places, organizations, events, dates, sources, and dateline locations.
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
import spacy
import nltk
from nltk.tokenize import sent_tokenize

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
    "event_keywords": [
        "hurricane", "storm", "earthquake", "flood", "war", "attack", "bombing",
        "election", "vote", "campaign", "protest", "demonstration", "riot",
        "pandemic", "outbreak", "crisis", "scandal", "investigation", "summit",
        "conference", "meeting", "agreement", "deal", "treaty", "legislation"
    ]
}

# Save configuration
with open(f"{config_dir}/ner_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Loading data...")
# Load the enhanced data from previous step
try:
    df = pd.read_csv(f"{data_dir}/news_sample_enhanced.csv")
    print(f"Loaded enhanced dataset with {len(df)} records")
except FileNotFoundError:
    # Fall back to original sample if enhanced not available
    df = pd.read_csv(f"{data_dir}/news_sample.csv")
    print(f"Enhanced dataset not found, loaded original sample with {len(df)} records")

# Download required NLTK data
print("Setting up NLP tools...")
try:
    nltk.download('punkt')
except:
    print("NLTK download failed, but continuing...")

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

# Function to extract dateline
def extract_dateline(text):
    """Extract dateline (location at beginning of article) if present."""
    if pd.isna(text):
        return None
    
    # Common dateline patterns
    patterns = [
        r'^([A-Z]+[A-Z\s]+),\s',  # WASHINGTON, ...
        r'^([A-Z]+[A-Z\s]+)\s\(',  # WASHINGTON (Reuters) ...
        r'^([A-Z]+[A-Z\s]+)\s-\s',  # WASHINGTON - ...
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return None

# Function to extract source location
def extract_source_location(text):
    """Extract location where the source claims to be reporting from."""
    if pd.isna(text):
        return None
    
    # Look for patterns like "reporting from [LOCATION]"
    patterns = [
        r'reporting from ([A-Za-z\s]+)',
        r'reports from ([A-Za-z\s]+)',
        r'in ([A-Za-z\s]+) for',
        r'from ([A-Za-z\s]+) reports',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return None

# Function to detect events
def detect_events(text, event_keywords):
    """Detect events mentioned in the text based on keywords."""
    if pd.isna(text):
        return []
    
    events = []
    text_lower = text.lower()
    
    for keyword in event_keywords:
        if keyword in text_lower:
            # Get context around the keyword
            pattern = r'[^.!?]*\b' + keyword + r'\b[^.!?]*[.!?]'
            matches = re.findall(pattern, text_lower)
            for match in matches:
                events.append((keyword, match.strip()))
    
    return events

# Function to perform NER using spaCy
def extract_entities(text):
    """Extract named entities using spaCy."""
    if pd.isna(text) or nlp is None:
        return [], [], [], []
    
    people = []
    places = []
    organizations = []
    dates = []
    
    # Process text with spaCy
    doc = nlp(text[:1000000])  # Limit text length to avoid memory issues
    
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

print("Extracting entities...")
# Apply extraction to the dataset
df['dateline'] = df['text'].apply(extract_dateline)
df['source_location'] = df['text'].apply(extract_source_location)

# Extract events
df['events'] = df['text'].apply(lambda x: detect_events(x, config["event_keywords"]))
df['event_count'] = df['events'].apply(len)
df['event_types'] = df['events'].apply(lambda events: [event[0] for event in events])

# Extract named entities
if nlp is not None:
    print("Performing NER extraction...")
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
    df['date_count'] = df['dates'].apply(len)
else:
    print("Skipping NER extraction due to missing spaCy model")
    df['people'] = [[] for _ in range(len(df))]
    df['places'] = [[] for _ in range(len(df))]
    df['organizations'] = [[] for _ in range(len(df))]
    df['dates'] = [[] for _ in range(len(df))]
    df['people_count'] = 0
    df['places_count'] = 0
    df['org_count'] = 0
    df['date_count'] = 0

# Analyze entity distributions
print("Analyzing entity distributions...")

# Function to count and analyze entities
def analyze_entities(entity_lists, min_freq=2):
    """Count and analyze entities across documents."""
    all_entities = []
    for entities in entity_lists:
        all_entities.extend(entities)
    
    # Count entities
    entity_counts = Counter(all_entities)
    
    # Filter by minimum frequency
    filtered_counts = {k: v for k, v in entity_counts.items() if v >= min_freq}
    
    return filtered_counts

# Analyze entities by type
people_counts = analyze_entities(df['people'], config["min_entity_freq"])
place_counts = analyze_entities(df['places'], config["min_entity_freq"])
org_counts = analyze_entities(df['organizations'], config["min_entity_freq"])
event_counts = Counter([event[0] for events in df['event_types'] for event in events])

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

# Analyze events by label
events_fake = []
events_real = []
for i, row in df.iterrows():
    if row['label'] == 0:  # Fake
        events_fake.extend(row['event_types'])
    else:  # Real
        events_real.extend(row['event_types'])

events_fake_counts = Counter(events_fake)
events_real_counts = Counter(events_real)

# Analyze datelines and source locations
dateline_counts = df['dateline'].value_counts()
source_location_counts = df['source_location'].value_counts()

print(f"Found {len(dateline_counts)} unique datelines")
print(f"Found {len(source_location_counts)} unique source locations")

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
    plt.savefig(f"{results_dir}/{entity_type.lower()}_distribution.png")
    
    return plot_df

# Create entity comparison charts
if people_counts:
    people_comparison = create_entity_comparison(people_fake, people_real, 'People', config["top_n_entities"])
if place_counts:
    places_comparison = create_entity_comparison(places_fake, places_real, 'Places', config["top_n_entities"])
if org_counts:
    orgs_comparison = create_entity_comparison(orgs_fake, orgs_real, 'Organizations', config["top_n_entities"])
events_comparison = create_entity_comparison(events_fake_counts, events_real_counts, 'Events', config["top_n_entities"])

# Create entity count comparison by article type
plt.figure(figsize=(10, 6))
entity_counts = df.groupby('label').agg({
    'people_count': 'mean',
    'places_count': 'mean',
    'org_count': 'mean',
    'event_count': 'mean'
}).reset_index()
entity_counts['label'] = entity_counts['label'].map({0: 'Fake', 1: 'Real'})
entity_counts = entity_counts.melt(id_vars=['label'], var_name='Entity Type', value_name='Average Count')
entity_counts['Entity Type'] = entity_counts['Entity Type'].str.replace('_count', '')

sns.barplot(x='Entity Type', y='Average Count', hue='label', data=entity_counts)
plt.title('Average Entity Counts in Fake vs Real News')
plt.tight_layout()
plt.savefig(f"{results_dir}/avg_entity_counts.png")

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
plt.subplot(2, 2, 4)
top_5_events = events_comparison.sort_values('Fake', ascending=False).head(5)
top_5_events[['Fake', 'Real']].plot(kind='bar', stacked=True)
plt.title('Top 5 Events Mentioned')
plt.xticks(range(len(top_5_events)), top_5_events['Entity'], rotation=45, ha='right')
plt.legend(title='')

plt.tight_layout()
plt.savefig(f"{results_dir}/entity_dashboard.png")

# Save enhanced dataset with NER
df.to_csv(f"{data_dir}/news_sample_ner_enhanced.csv", index=False)

# Save entity analysis results
entity_results = {
    "people": {
        "unique_count": len(people_counts),
        "top_people": {k: v for k, v in sorted(people_counts.items(), key=lambda x: x[1], reverse=True)[:config["top_n_entities"]]}
    },
    "places": {
        "unique_count": len(place_counts),
        "top_places": {k: v for k, v in sorted(place_counts.items(), key=lambda x: x[1], reverse=True)[:config["top_n_entities"]]}
    },
    "organizations": {
        "unique_count": len(org_counts),
        "top_organizations": {k: v for k, v in sorted(org_counts.items(), key=lambda x: x[1], reverse=True)[:config["top_n_entities"]]}
    },
    "events": {
        "unique_count": len(event_counts),
        "top_events": {k: v for k, v in sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:config["top_n_entities"]]}
    },
    "datelines": {
        "unique_count": len(dateline_counts),
        "top_datelines": dateline_counts.head(config["top_n_entities"]).to_dict()
    },
    "source_locations": {
        "unique_count": len(source_location_counts),
        "top_source_locations": source_location_counts.head(config["top_n_entities"]).to_dict()
    },
    "execution_time": time.time() - start_time
}

with open(f"{results_dir}/entity_analysis.json", "w") as f:
    json.dump(entity_results, f, indent=2)

print(f"\nNER entity extraction and analysis completed in {time.time() - start_time:.2f} seconds")
print(f"Enhanced dataset saved to {data_dir}/news_sample_ner_enhanced.csv")
print(f"Results saved to {results_dir}")

# Last modified: May 29, 2025
