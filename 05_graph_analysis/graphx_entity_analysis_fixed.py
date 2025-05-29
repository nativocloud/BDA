"""
Script to implement GraphX-based entity analysis for fake news detection.
This script creates a graph structure from extracted entities and analyzes relationships.
"""

import pandas as pd
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, array, lit, collect_list, count, when
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType
from graphframes import GraphFrame

# Start timer
start_time = time.time()

# Define paths
data_dir = "/home/ubuntu/fake_news_detection/data"
results_dir = "/home/ubuntu/fake_news_detection/logs"
config_dir = "/home/ubuntu/fake_news_detection/config"
checkpoint_dir = "/home/ubuntu/fake_news_detection/checkpoint"

# Create directories if they don't exist
os.makedirs(results_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Configuration parameters
config = {
    "min_entity_freq": 2,  # Minimum frequency for entity to be included in graph
    "top_n_entities": 20,  # Number of top entities to display in visualizations
    "min_edge_weight": 2   # Minimum co-occurrence weight for edge to be included
}

# Save configuration
with open(f"{config_dir}/graphx_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Initializing Spark session...")
# Initialize Spark session
spark = SparkSession.builder \
    .appName("GraphX Entity Analysis") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .master("local[*]") \
    .getOrCreate()

# Set checkpoint directory for connected components algorithm
spark.sparkContext.setCheckpointDir(checkpoint_dir)
print(f"Set checkpoint directory to {checkpoint_dir}")

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
            df[col_name] = df[col_name].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Create Spark DataFrame
print("Creating Spark DataFrame...")
# Define schema for the DataFrame
schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("label", IntegerType(), False),
    StructField("people", ArrayType(StringType()), True),
    StructField("places", ArrayType(StringType()), True),
    StructField("organizations", ArrayType(StringType()), True),
    StructField("event_types", ArrayType(StringType()), True)
])

# Add ID column if not present
if 'id' not in df.columns:
    df['id'] = range(len(df))

# Convert pandas DataFrame to Spark DataFrame
# Ensure lists are properly handled
spark_df = spark.createDataFrame([
    (
        int(row['id']), 
        int(row['label']), 
        row['people'] if 'people' in df.columns and isinstance(row['people'], list) else [],
        row['places'] if 'places' in df.columns and isinstance(row['places'], list) else [],
        row['organizations'] if 'organizations' in df.columns and isinstance(row['organizations'], list) else [],
        row['event_types'] if 'event_types' in df.columns and isinstance(row['event_types'], list) else []
    )
    for _, row in df.iterrows()
], schema=schema)

# Create entity nodes
print("Creating entity nodes...")
# Explode people entities
people_df = spark_df.select(
    explode(col("people")).alias("entity"),
    lit("person").alias("entity_type"),
    col("label")
)

# Explode place entities
places_df = spark_df.select(
    explode(col("places")).alias("entity"),
    lit("place").alias("entity_type"),
    col("label")
)

# Explode organization entities
org_df = spark_df.select(
    explode(col("organizations")).alias("entity"),
    lit("organization").alias("entity_type"),
    col("label")
)

# Explode event entities
event_df = spark_df.select(
    explode(col("event_types")).alias("entity"),
    lit("event").alias("entity_type"),
    col("label")
)

# Union all entity dataframes
all_entities_df = people_df.union(places_df).union(org_df).union(event_df)

# Count entity occurrences and filter by minimum frequency
entity_counts = all_entities_df.groupBy("entity", "entity_type") \
    .agg(count("*").alias("count")) \
    .filter(col("count") >= config["min_entity_freq"])

# Count entity occurrences by label
entity_label_counts = all_entities_df.groupBy("entity", "entity_type", "label") \
    .agg(count("*").alias("label_count"))

# Join to get fake and real counts
entity_stats = entity_counts.join(
    entity_label_counts.filter(col("label") == 0).select(
        col("entity"),
        col("label_count").alias("fake_count")
    ),
    "entity",
    "left"
).join(
    entity_label_counts.filter(col("label") == 1).select(
        col("entity"),
        col("label_count").alias("real_count")
    ),
    "entity",
    "left"
)

# Fill null values with 0
entity_stats = entity_stats.fillna({"fake_count": 0, "real_count": 0})

# Calculate fake and real ratios
entity_stats = entity_stats.withColumn(
    "fake_ratio", 
    col("fake_count") / (col("fake_count") + col("real_count"))
).withColumn(
    "real_ratio", 
    col("real_count") / (col("fake_count") + col("real_count"))
)

# Create vertices for GraphFrames
vertices = entity_stats.select(
    col("entity").alias("id"),
    col("entity_type"),
    col("count"),
    col("fake_count"),
    col("real_count"),
    col("fake_ratio"),
    col("real_ratio")
)

print(f"Created {vertices.count()} entity nodes")

# Create edges between co-occurring entities
print("Creating entity relationship edges...")
# Function to create all pairs of entities in a document
def create_entity_pairs(row):
    all_entities = []
    if 'people' in row and row['people']:
        all_entities.extend([(entity, 'person') for entity in row['people']])
    if 'places' in row and row['places']:
        all_entities.extend([(entity, 'place') for entity in row['places']])
    if 'organizations' in row and row['organizations']:
        all_entities.extend([(entity, 'organization') for entity in row['organizations']])
    if 'event_types' in row and row['event_types']:
        all_entities.extend([(entity, 'event') for entity in row['event_types']])
    
    pairs = []
    for i in range(len(all_entities)):
        for j in range(i+1, len(all_entities)):
            # Create edge in both directions for undirected graph
            pairs.append((all_entities[i][0], all_entities[j][0], all_entities[i][1], all_entities[j][1], row['label']))
            pairs.append((all_entities[j][0], all_entities[i][0], all_entities[j][1], all_entities[i][1], row['label']))
    
    return pairs

# Create all entity pairs
all_pairs = []
for _, row in df.iterrows():
    all_pairs.extend(create_entity_pairs(row))

# Create DataFrame for edges
edge_schema = StructType([
    StructField("src", StringType(), False),
    StructField("dst", StringType(), False),
    StructField("src_type", StringType(), False),
    StructField("dst_type", StringType(), False),
    StructField("label", IntegerType(), False)
])

edges_df = spark.createDataFrame(all_pairs, schema=edge_schema)

# Count co-occurrences
edge_counts = edges_df.groupBy("src", "dst", "src_type", "dst_type") \
    .agg(
        count("*").alias("weight"),
        count(when(col("label") == 0, 1)).alias("fake_weight"),
        count(when(col("label") == 1, 1)).alias("real_weight")
    )

# Filter edges by minimum weight
filtered_edges = edge_counts.filter(col("weight") >= config["min_edge_weight"])

# Create edges for GraphFrames
edges = filtered_edges.select(
    col("src"),
    col("dst"),
    col("weight"),
    col("fake_weight"),
    col("real_weight"),
    col("src_type"),
    col("dst_type")
)

print(f"Created {edges.count()} relationship edges")

# Create GraphFrame
print("Creating GraphFrame...")
g = GraphFrame(vertices, edges)

# Run PageRank algorithm
print("Running PageRank algorithm...")
results = g.pageRank(resetProbability=0.15, tol=0.01)
pr_vertices = results.vertices.select("id", "entity_type", "pagerank", "count", "fake_count", "real_count")

# Run connected components algorithm
print("Finding connected components...")
try:
    connected_components = g.connectedComponents()
    cc_vertices = connected_components.select("id", "component")

    # Join PageRank and connected components results
    enriched_vertices = pr_vertices.join(cc_vertices, "id")

    # Get top entities by PageRank
    top_entities = enriched_vertices.orderBy(col("pagerank").desc()).limit(config["top_n_entities"])
    top_entities_pd = top_entities.toPandas()

    print("Top entities by PageRank:")
    for i, row in top_entities_pd.iterrows():
        print(f"{i+1}. {row['id']} ({row['entity_type']}): {row['pagerank']:.4f}")

    # Get top connected components
    top_components = connected_components.groupBy("component") \
        .agg(count("*").alias("size")) \
        .orderBy(col("size").desc()) \
        .limit(10)

    top_components_pd = top_components.toPandas()
    print("\nTop connected components:")
    for i, row in top_components_pd.iterrows():
        print(f"{i+1}. Component {row['component']}: {row['size']} entities")
except Exception as e:
    print(f"Error running connected components: {e}")
    print("Continuing without connected components analysis...")
    # Create a simplified version without connected components
    enriched_vertices = pr_vertices
    top_entities = enriched_vertices.orderBy(col("pagerank").desc()).limit(config["top_n_entities"])
    top_entities_pd = top_entities.toPandas()
    
    print("Top entities by PageRank:")
    for i, row in top_entities_pd.iterrows():
        print(f"{i+1}. {row['id']} ({row['entity_type']}): {row['pagerank']:.4f}")

# Create visualizations
print("Creating visualizations...")

# Convert to pandas for visualization
vertices_pd = vertices.toPandas()
edges_pd = edges.toPandas()

# Entity type distribution
plt.figure(figsize=(10, 6))
entity_type_counts = vertices_pd.groupby('entity_type').size()
entity_type_counts.plot(kind='bar')
plt.title('Entity Type Distribution')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f"{results_dir}/graphx_entity_type_distribution.png")

# Entity fake vs real distribution
plt.figure(figsize=(12, 8))
# Get top entities by count
top_by_count = vertices_pd.nlargest(10, 'count')
# Create stacked bar chart
top_by_count.plot(kind='bar', x='id', y=['fake_count', 'real_count'], stacked=True)
plt.title('Top 10 Entities - Fake vs Real Distribution')
plt.xlabel('Entity')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{results_dir}/graphx_top_entities_distribution.png")

# PageRank distribution
plt.figure(figsize=(12, 8))
plt.scatter(top_entities_pd['count'], top_entities_pd['pagerank'], 
           c=top_entities_pd['entity_type'].astype('category').cat.codes, 
           alpha=0.7, s=100)
for i, row in top_entities_pd.iterrows():
    plt.annotate(row['id'], (row['count'], row['pagerank']), 
                fontsize=9, ha='center', va='bottom')
plt.title('Entity PageRank vs Frequency')
plt.xlabel('Entity Frequency')
plt.ylabel('PageRank Score')
plt.colorbar(label='Entity Type')
plt.tight_layout()
plt.savefig(f"{results_dir}/graphx_pagerank_distribution.png")

# Create network visualization
print("Creating network visualization...")
try:
    import networkx as nx
    
    # Create graph from top entities
    G = nx.Graph()
    
    # Get top entities by PageRank
    top_entities_list = [row['id'] for _, row in top_entities_pd.iterrows()]
    
    # Add nodes with attributes
    for _, row in vertices_pd[vertices_pd['id'].isin(top_entities_list)].iterrows():
        G.add_node(row['id'], 
                  entity_type=row['entity_type'], 
                  count=row['count'],
                  fake_ratio=row['fake_ratio'] if 'fake_ratio' in row else 0)
    
    # Add edges between top entities
    for _, row in edges_pd[(edges_pd['src'].isin(top_entities_list)) & (edges_pd['dst'].isin(top_entities_list))].iterrows():
        G.add_edge(row['src'], row['dst'], weight=row['weight'])
    
    # Create layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Create plot
    plt.figure(figsize=(15, 15))
    
    # Node colors by entity type
    entity_types = nx.get_node_attributes(G, 'entity_type')
    type_colors = {'person': 'red', 'place': 'blue', 'organization': 'green', 'event': 'purple'}
    node_colors = [type_colors.get(entity_types.get(node, 'other'), 'gray') for node in G.nodes()]
    
    # Node sizes by count
    counts = nx.get_node_attributes(G, 'count')
    node_sizes = [50 + 10 * counts.get(node, 1) for node in G.nodes()]
    
    # Edge weights
    edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 label=etype, markersize=10) 
                      for etype, color in type_colors.items()]
    plt.legend(handles=legend_elements, title='Entity Type')
    
    plt.title(f'Entity Relationship Network (Top {len(top_entities_list)} Entities)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/graphx_network_visualization.png", dpi=300)
    
except ImportError:
    print("NetworkX not available, skipping network visualization")
except Exception as e:
    print(f"Error creating network visualization: {e}")

# Create features for machine learning
print("Creating GraphX-based features for machine learning...")

# Get entity PageRank scores
pagerank_scores = results.vertices.select("id", "pagerank").toPandas()
pagerank_dict = dict(zip(pagerank_scores['id'], pagerank_scores['pagerank']))

# Create entity-based features
def create_graphx_features(row):
    features = {}
    
    # Entity count features
    features['person_count'] = len(row['people']) if 'people' in row and row['people'] else 0
    features['place_count'] = len(row['places']) if 'places' in row and row['places'] else 0
    features['org_count'] = len(row['organizations']) if 'organizations' in row and row['organizations'] else 0
    features['event_count'] = len(row['event_types']) if 'event_types' in row and row['event_types'] else 0
    
    # Average PageRank features
    if 'people' in row and row['people']:
        pr_scores = [pagerank_dict.get(entity, 0) for entity in row['people']]
        features['avg_person_pagerank'] = sum(pr_scores) / len(pr_scores) if pr_scores else 0
        features['max_person_pagerank'] = max(pr_scores) if pr_scores else 0
    else:
        features['avg_person_pagerank'] = 0
        features['max_person_pagerank'] = 0
        
    if 'places' in row and row['places']:
        pr_scores = [pagerank_dict.get(entity, 0) for entity in row['places']]
        features['avg_place_pagerank'] = sum(pr_scores) / len(pr_scores) if pr_scores else 0
        features['max_place_pagerank'] = max(pr_scores) if pr_scores else 0
    else:
        features['avg_place_pagerank'] = 0
        features['max_place_pagerank'] = 0
        
    if 'organizations' in row and row['organizations']:
        pr_scores = [pagerank_dict.get(entity, 0) for entity in row['organizations']]
        features['avg_org_pagerank'] = sum(pr_scores) / len(pr_scores) if pr_scores else 0
        features['max_org_pagerank'] = max(pr_scores) if pr_scores else 0
    else:
        features['avg_org_pagerank'] = 0
        features['max_org_pagerank'] = 0
        
    if 'event_types' in row and row['event_types']:
        pr_scores = [pagerank_dict.get(entity, 0) for entity in row['event_types']]
        features['avg_event_pagerank'] = sum(pr_scores) / len(pr_scores) if pr_scores else 0
        features['max_event_pagerank'] = max(pr_scores) if pr_scores else 0
    else:
        features['avg_event_pagerank'] = 0
        features['max_event_pagerank'] = 0
    
    return features

# Apply feature creation
graphx_features = []
for _, row in df.iterrows():
    features = create_graphx_features(row)
    graphx_features.append(features)

# Convert to DataFrame
graphx_features_df = pd.DataFrame(graphx_features)

# Add to original DataFrame
for col in graphx_features_df.columns:
    df[f'graphx_{col}'] = graphx_features_df[col]

# Save enhanced dataset with GraphX features
df.to_csv(f"{data_dir}/news_sample_graphx_enhanced.csv", index=False)

# Save GraphX analysis results
graphx_results = {
    "vertices": {
        "count": int(vertices.count()),
        "by_type": vertices.groupBy("entity_type").count().toPandas().set_index("entity_type")["count"].to_dict()
    },
    "edges": {
        "count": int(edges.count()),
        "avg_weight": float(edges.agg({"weight": "avg"}).collect()[0][0])
    },
    "top_entities": top_entities_pd.to_dict(orient="records"),
    "execution_time": time.time() - start_time
}

with open(f"{results_dir}/graphx_analysis.json", "w") as f:
    json.dump(graphx_results, f, indent=2)

print(f"\nGraphX entity analysis completed in {time.time() - start_time:.2f} seconds")
print(f"Enhanced dataset saved to {data_dir}/news_sample_graphx_enhanced.csv")
print(f"Results saved to {results_dir}")

# Stop Spark session
spark.stop()

# Last modified: May 29, 2025
