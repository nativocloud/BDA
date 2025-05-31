# %% [markdown]
# # Fake News Detection: Graph Analysis
# 
# This notebook contains all the necessary code for graph-based entity analysis in the fake news detection project. The code is organized into independent functions, without dependencies on external modules or classes, to facilitate execution in Databricks Community Edition.

# %% [markdown]
# ## Setup and Imports

# %%
# Import necessary libraries
import os
import time
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, array, lit, collect_list, count, when, udf, struct, 
    array_contains, concat_ws, split, size, expr
)
from pyspark.sql.types import (
    ArrayType, StringType, StructType, StructField, IntegerType, 
    FloatType, BooleanType, MapType
)

# Try to import GraphFrames - this may fail in environments without GraphX support
try:
    from graphframes import GraphFrame
    graphframes_available = True
    print("GraphFrames is available")
except ImportError:
    graphframes_available = False
    print("GraphFrames is not available - will use alternative implementation")

# %%
# Initialize Spark session optimized for Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsDetection_GraphAnalysis") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

# If GraphFrames is available, configure it
if graphframes_available:
    try:
        spark.conf.set("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12")
    except:
        print("Warning: Could not configure GraphFrames package. If running in Databricks, this may be normal.")

# Display Spark configuration
print(f"Spark version: {spark.version}")
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")

# %%
# Start timer for performance tracking
start_time = time.time()

# %% [markdown]
# ## Reusable Functions

# %% [markdown]
# ### Data Loading Functions

# %%
def load_preprocessed_data(path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet"):
    """
    Load preprocessed data from Parquet file.
    
    Args:
        path (str): Path to the preprocessed data Parquet file
        
    Returns:
        DataFrame: Spark DataFrame with preprocessed data
    """
    print(f"Loading preprocessed data from {path}...")
    
    try:
        # Load data from Parquet file
        df = spark.read.parquet(path)
        
        # Display basic information
        print(f"Successfully loaded {df.count()} records.")
        df.printSchema()
        
        # Cache the DataFrame for better performance
        df.cache()
        print("Preprocessed DataFrame cached.")
        
        return df
    
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        print("Please ensure the preprocessing notebook ran successfully and saved data to the correct path.")
        return None

# %%
def load_feature_data(path="dbfs:/FileStore/fake_news_detection/feature_data/features.parquet"):
    """
    Load feature data from Parquet file.
    
    Args:
        path (str): Path to the feature data Parquet file
        
    Returns:
        DataFrame: Spark DataFrame with feature data
    """
    print(f"Loading feature data from {path}...")
    
    try:
        # Load data from Parquet file
        df = spark.read.parquet(path)
        
        # Display basic information
        print(f"Successfully loaded {df.count()} records.")
        df.printSchema()
        
        # Cache the DataFrame for better performance
        df.cache()
        print("Feature DataFrame cached.")
        
        return df
    
    except Exception as e:
        print(f"Error loading feature data: {e}")
        print("Please ensure the feature engineering notebook ran successfully and saved data to the correct path.")
        return None

# %% [markdown]
# ### Named Entity Recognition Functions

# %%
def extract_entities_with_spacy(text):
    """
    Extract named entities from text using SpaCy.
    
    Args:
        text (str): Text to extract entities from
        
    Returns:
        dict: Dictionary with entity lists by type
    """
    # This is a simplified version - in production, use a proper SpaCy pipeline
    # Here we use regex patterns to simulate entity extraction
    
    if not text or not isinstance(text, str):
        return {"people": [], "places": [], "organizations": [], "events": []}
    
    # Simple regex patterns for demonstration
    people_pattern = r'(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.) [A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+ [A-Z][a-z]+'
    places_pattern = r'(in|at|from) ([A-Z][a-z]+(,)? [A-Z][a-z]+|[A-Z][a-z]+)'
    org_pattern = r'([A-Z][a-z]* (Corporation|Inc\.|Company|Association|Organization|Agency|Department))|([A-Z][A-Z]+)'
    event_pattern = r'(conference|meeting|summit|election|war|attack|ceremony|festival|celebration)'
    
    # Extract entities
    people = list(set(re.findall(people_pattern, text)))
    places = list(set([m.group(2) for m in re.finditer(places_pattern, text)]))
    organizations = list(set(re.findall(org_pattern, text)))
    events = list(set(re.findall(event_pattern, text, re.IGNORECASE)))
    
    # Clean up entities
    people = [p.strip() for p in people]
    places = [p.strip() for p in places]
    organizations = [o.strip() for o in organizations]
    events = [e.strip() for e in events]
    
    return {
        "people": people,
        "places": places,
        "organizations": organizations,
        "events": events
    }

# %%
def extract_entities_from_dataframe(df, text_column="text", batch_size=1000):
    """
    Extract named entities from a DataFrame using SpaCy.
    
    Args:
        df (DataFrame): Spark DataFrame with text data
        text_column (str): Name of the column containing text
        batch_size (int): Number of rows to process in each batch
        
    Returns:
        DataFrame: DataFrame with extracted entities
    """
    print("Extracting named entities from text...")
    
    # Register UDF for entity extraction
    extract_entities_udf = udf(extract_entities_with_spacy, MapType(StringType(), ArrayType(StringType())))
    
    # Apply UDF to extract entities
    df_with_entities = df.withColumn("entities", extract_entities_udf(col(text_column)))
    
    # Explode the entities map into separate columns
    df_with_entities = df_with_entities \
        .withColumn("people", col("entities")["people"]) \
        .withColumn("places", col("entities")["places"]) \
        .withColumn("organizations", col("entities")["organizations"]) \
        .withColumn("events", col("entities")["events"]) \
        .drop("entities")
    
    # Show sample of extracted entities
    df_with_entities.select("id", "label", "people", "places", "organizations", "events") \
        .show(5, truncate=50, vertical=True)
    
    return df_with_entities

# %% [markdown]
# ### Graph Creation Functions

# %%
def create_entity_nodes(df_with_entities, min_entity_freq=2):
    """
    Create entity nodes for graph analysis.
    
    Args:
        df_with_entities (DataFrame): DataFrame with extracted entities
        min_entity_freq (int): Minimum frequency for entity to be included
        
    Returns:
        DataFrame: DataFrame with entity nodes
    """
    print("Creating entity nodes...")
    
    # Explode people entities
    people_df = df_with_entities.select(
        explode(col("people")).alias("entity"),
        lit("person").alias("entity_type"),
        col("label")
    )
    
    # Explode place entities
    places_df = df_with_entities.select(
        explode(col("places")).alias("entity"),
        lit("place").alias("entity_type"),
        col("label")
    )
    
    # Explode organization entities
    org_df = df_with_entities.select(
        explode(col("organizations")).alias("entity"),
        lit("organization").alias("entity_type"),
        col("label")
    )
    
    # Explode event entities
    event_df = df_with_entities.select(
        explode(col("events")).alias("entity"),
        lit("event").alias("entity_type"),
        col("label")
    )
    
    # Union all entity dataframes
    all_entities_df = people_df.union(places_df).union(org_df).union(event_df)
    
    # Count entity occurrences and filter by minimum frequency
    entity_counts = all_entities_df.groupBy("entity", "entity_type") \
        .agg(count("*").alias("count")) \
        .filter(col("count") >= min_entity_freq)
    
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
    
    return vertices

# %%
def create_entity_edges(df_with_entities, min_edge_weight=2):
    """
    Create edges between co-occurring entities.
    
    Args:
        df_with_entities (DataFrame): DataFrame with extracted entities
        min_edge_weight (int): Minimum co-occurrence weight for edge to be included
        
    Returns:
        DataFrame: DataFrame with entity edges
    """
    print("Creating entity relationship edges...")
    
    # Create a UDF to generate all entity pairs in a document
    @udf(ArrayType(StructType([
        StructField("src", StringType()),
        StructField("dst", StringType()),
        StructField("src_type", StringType()),
        StructField("dst_type", StringType())
    ])))
    def generate_entity_pairs(people, places, organizations, events):
        # Collect all entities with their types
        all_entities = []
        if people:
            all_entities.extend([(entity, "person") for entity in people])
        if places:
            all_entities.extend([(entity, "place") for entity in places])
        if organizations:
            all_entities.extend([(entity, "organization") for entity in organizations])
        if events:
            all_entities.extend([(entity, "event") for entity in events])
        
        # Generate all pairs
        pairs = []
        for i in range(len(all_entities)):
            for j in range(i+1, len(all_entities)):
                # Create edge in both directions for undirected graph
                pairs.append({
                    "src": all_entities[i][0],
                    "dst": all_entities[j][0],
                    "src_type": all_entities[i][1],
                    "dst_type": all_entities[j][1]
                })
                pairs.append({
                    "src": all_entities[j][0],
                    "dst": all_entities[i][0],
                    "src_type": all_entities[j][1],
                    "dst_type": all_entities[i][1]
                })
        
        return pairs
    
    # Generate all entity pairs
    pairs_df = df_with_entities.select(
        col("people"),
        col("places"),
        col("organizations"),
        col("events"),
        col("label"),
        generate_entity_pairs(
            col("people"),
            col("places"),
            col("organizations"),
            col("events")
        ).alias("pairs")
    )
    
    # Explode the pairs
    edges_df = pairs_df.select(
        explode(col("pairs")).alias("pair"),
        col("label")
    ).select(
        col("pair.src").alias("src"),
        col("pair.dst").alias("dst"),
        col("pair.src_type").alias("src_type"),
        col("pair.dst_type").alias("dst_type"),
        col("label")
    )
    
    # Count co-occurrences
    edge_counts = edges_df.groupBy("src", "dst", "src_type", "dst_type") \
        .agg(
            count("*").alias("weight"),
            count(when(col("label") == 0, 1)).alias("fake_weight"),
            count(when(col("label") == 1, 1)).alias("real_weight")
        )
    
    # Filter edges by minimum weight
    filtered_edges = edge_counts.filter(col("weight") >= min_edge_weight)
    
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
    
    return edges

# %% [markdown]
# ### GraphX Analysis Functions

# %%
def create_graphframe(vertices, edges):
    """
    Create a GraphFrame from vertices and edges.
    
    Args:
        vertices (DataFrame): DataFrame with entity nodes
        edges (DataFrame): DataFrame with entity edges
        
    Returns:
        GraphFrame: GraphFrame object or None if GraphFrames not available
    """
    if not graphframes_available:
        print("GraphFrames is not available. Cannot create GraphFrame.")
        return None
    
    print("Creating GraphFrame...")
    try:
        g = GraphFrame(vertices, edges)
        print(f"GraphFrame created with {g.vertices.count()} vertices and {g.edges.count()} edges")
        return g
    except Exception as e:
        print(f"Error creating GraphFrame: {e}")
        return None

# %%
def run_pagerank(g, reset_probability=0.15, tolerance=0.01):
    """
    Run PageRank algorithm on a GraphFrame.
    
    Args:
        g (GraphFrame): GraphFrame object
        reset_probability (float): Reset probability for PageRank
        tolerance (float): Convergence tolerance
        
    Returns:
        tuple: (vertices_with_pagerank, edges_with_pagerank) DataFrames with PageRank scores
    """
    if g is None:
        print("GraphFrame is None. Cannot run PageRank.")
        return None, None
    
    print("Running PageRank algorithm...")
    try:
        results = g.pageRank(resetProbability=reset_probability, tol=tolerance)
        pr_vertices = results.vertices.select(
            "id", "entity_type", "pagerank", "count", "fake_count", "real_count", "fake_ratio", "real_ratio"
        )
        pr_edges = results.edges.select(
            "src", "dst", "weight", "fake_weight", "real_weight", "src_type", "dst_type", "weight"
        )
        
        print("PageRank algorithm completed successfully")
        return pr_vertices, pr_edges
    except Exception as e:
        print(f"Error running PageRank: {e}")
        return None, None

# %%
def run_connected_components(g):
    """
    Run Connected Components algorithm on a GraphFrame.
    
    Args:
        g (GraphFrame): GraphFrame object
        
    Returns:
        DataFrame: DataFrame with component assignments
    """
    if g is None:
        print("GraphFrame is None. Cannot run Connected Components.")
        return None
    
    print("Running Connected Components algorithm...")
    try:
        result = g.connectedComponents()
        print("Connected Components algorithm completed successfully")
        return result
    except Exception as e:
        print(f"Error running Connected Components: {e}")
        return None

# %%
def run_triangle_count(g):
    """
    Run Triangle Count algorithm on a GraphFrame.
    
    Args:
        g (GraphFrame): GraphFrame object
        
    Returns:
        DataFrame: DataFrame with triangle counts
    """
    if g is None:
        print("GraphFrame is None. Cannot run Triangle Count.")
        return None
    
    print("Running Triangle Count algorithm...")
    try:
        result = g.triangleCount()
        print("Triangle Count algorithm completed successfully")
        return result
    except Exception as e:
        print(f"Error running Triangle Count: {e}")
        return None

# %% [markdown]
# ### Non-GraphX Analysis Functions

# %%
def create_networkx_graph(vertices_df, edges_df):
    """
    Create a NetworkX graph from vertices and edges DataFrames.
    
    Args:
        vertices_df (DataFrame): DataFrame with entity nodes
        edges_df (DataFrame): DataFrame with entity edges
        
    Returns:
        nx.Graph: NetworkX graph object
    """
    print("Creating NetworkX graph...")
    
    # Convert vertices to pandas
    vertices_pd = vertices_df.toPandas()
    
    # Convert edges to pandas
    edges_pd = edges_df.toPandas()
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for _, row in vertices_pd.iterrows():
        G.add_node(
            row['id'],
            entity_type=row['entity_type'],
            count=row['count'],
            fake_count=row['fake_count'],
            real_count=row['real_count'],
            fake_ratio=row['fake_ratio'],
            real_ratio=row['real_ratio']
        )
    
    # Add edges with attributes
    for _, row in edges_pd.iterrows():
        G.add_edge(
            row['src'],
            row['dst'],
            weight=row['weight'],
            fake_weight=row['fake_weight'],
            real_weight=row['real_weight'],
            src_type=row['src_type'],
            dst_type=row['dst_type']
        )
    
    print(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

# %%
def calculate_networkx_metrics(G):
    """
    Calculate various network metrics using NetworkX.
    
    Args:
        G (nx.Graph): NetworkX graph object
        
    Returns:
        dict: Dictionary with network metrics
    """
    print("Calculating network metrics...")
    
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Degree metrics
    degrees = [d for n, d in G.degree()]
    metrics['avg_degree'] = sum(degrees) / len(degrees) if degrees else 0
    metrics['max_degree'] = max(degrees) if degrees else 0
    
    # Centrality metrics (for a sample of nodes to avoid performance issues)
    sample_size = min(100, len(G.nodes()))
    sample_nodes = list(G.nodes())[:sample_size]
    sample_graph = G.subgraph(sample_nodes)
    
    try:
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(sample_graph, k=min(10, sample_size-1))
        metrics['max_betweenness'] = max(betweenness.values()) if betweenness else 0
        metrics['avg_betweenness'] = sum(betweenness.values()) / len(betweenness) if betweenness else 0
        
        # Closeness centrality
        closeness = nx.closeness_centrality(sample_graph)
        metrics['max_closeness'] = max(closeness.values()) if closeness else 0
        metrics['avg_closeness'] = sum(closeness.values()) / len(closeness) if closeness else 0
        
        # Eigenvector centrality
        eigenvector = nx.eigenvector_centrality(sample_graph, max_iter=100)
        metrics['max_eigenvector'] = max(eigenvector.values()) if eigenvector else 0
        metrics['avg_eigenvector'] = sum(eigenvector.values()) / len(eigenvector) if eigenvector else 0
    except:
        print("Warning: Some centrality metrics could not be calculated")
    
    # Component analysis
    components = list(nx.connected_components(G))
    metrics['num_components'] = len(components)
    component_sizes = [len(c) for c in components]
    metrics['largest_component_size'] = max(component_sizes) if component_sizes else 0
    metrics['avg_component_size'] = sum(component_sizes) / len(component_sizes) if component_sizes else 0
    
    # Triangle count
    triangles = sum(nx.triangles(G).values()) / 3  # Divide by 3 as each triangle is counted 3 times
    metrics['triangle_count'] = triangles
    
    print("Network metrics calculated successfully")
    return metrics

# %%
def calculate_pagerank_networkx(G, alpha=0.85, max_iter=100):
    """
    Calculate PageRank using NetworkX.
    
    Args:
        G (nx.Graph): NetworkX graph object
        alpha (float): Damping parameter
        max_iter (int): Maximum number of iterations
        
    Returns:
        dict: Dictionary with PageRank scores
    """
    print("Calculating PageRank using NetworkX...")
    
    try:
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=max_iter)
        print("PageRank calculation completed successfully")
        return pagerank
    except Exception as e:
        print(f"Error calculating PageRank: {e}")
        return {}

# %% [markdown]
# ### Visualization Functions

# %%
def visualize_entity_distribution(vertices_df, top_n=20):
    """
    Visualize the distribution of entities by type and label.
    
    Args:
        vertices_df (DataFrame): DataFrame with entity nodes
        top_n (int): Number of top entities to display
        
    Returns:
        None
    """
    # Convert to pandas for visualization
    vertices_pd = vertices_df.toPandas()
    
    # Sort by count
    vertices_pd = vertices_pd.sort_values('count', ascending=False)
    
    # Get top N entities
    top_entities = vertices_pd.head(top_n)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Entity count by type
    entity_type_counts = vertices_pd.groupby('entity_type')['count'].sum().reset_index()
    sns.barplot(x='entity_type', y='count', data=entity_type_counts, ax=axes[0, 0])
    axes[0, 0].set_title('Entity Count by Type')
    axes[0, 0].set_xlabel('Entity Type')
    axes[0, 0].set_ylabel('Count')
    
    # Plot 2: Top entities by count
    sns.barplot(x='count', y='id', data=top_entities, ax=axes[0, 1])
    axes[0, 1].set_title(f'Top {top_n} Entities by Count')
    axes[0, 1].set_xlabel('Count')
    axes[0, 1].set_ylabel('Entity')
    
    # Plot 3: Fake vs Real ratio for top entities
    top_entities_melted = pd.melt(
        top_entities, 
        id_vars=['id', 'entity_type'], 
        value_vars=['fake_count', 'real_count'],
        var_name='label', 
        value_name='count'
    )
    sns.barplot(x='count', y='id', hue='label', data=top_entities_melted, ax=axes[1, 0])
    axes[1, 0].set_title(f'Fake vs Real Count for Top {top_n} Entities')
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_ylabel('Entity')
    
    # Plot 4: Fake ratio for top entities
    sns.barplot(x='fake_ratio', y='id', data=top_entities, ax=axes[1, 1])
    axes[1, 1].set_title(f'Fake Ratio for Top {top_n} Entities')
    axes[1, 1].set_xlabel('Fake Ratio')
    axes[1, 1].set_ylabel('Entity')
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.show()

# %%
def visualize_network(G, top_n=100, layout='spring'):
    """
    Visualize the entity network.
    
    Args:
        G (nx.Graph): NetworkX graph object
        top_n (int): Number of top nodes to display
        layout (str): Layout algorithm to use
        
    Returns:
        None
    """
    # Get top N nodes by degree
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n]
    
    # Create subgraph with top nodes
    subgraph = G.subgraph(top_nodes)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(subgraph, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(subgraph)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(subgraph)
    else:
        pos = nx.spring_layout(subgraph, seed=42)
    
    # Create figure
    plt.figure(figsize=(20, 16))
    
    # Get node attributes
    node_types = nx.get_node_attributes(subgraph, 'entity_type')
    node_counts = nx.get_node_attributes(subgraph, 'count')
    node_fake_ratios = nx.get_node_attributes(subgraph, 'fake_ratio')
    
    # Define colors for entity types
    type_colors = {
        'person': 'blue',
        'place': 'green',
        'organization': 'red',
        'event': 'purple'
    }
    
    # Define node colors based on entity type
    node_colors = [type_colors.get(node_types.get(n, 'unknown'), 'gray') for n in subgraph.nodes()]
    
    # Define node sizes based on count
    node_sizes = [50 + 10 * node_counts.get(n, 1) for n in subgraph.nodes()]
    
    # Define edge widths based on weight
    edge_widths = [0.1 + 0.1 * subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, 
        pos, 
        node_color=node_colors, 
        node_size=node_sizes, 
        alpha=0.7
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        subgraph, 
        pos, 
        width=edge_widths, 
        alpha=0.3, 
        edge_color='gray'
    )
    
    # Draw labels for top 20 nodes
    top_20_nodes = sorted(degrees, key=degrees.get, reverse=True)[:20]
    labels = {n: n for n in top_20_nodes if n in subgraph}
    nx.draw_networkx_labels(
        subgraph, 
        pos, 
        labels=labels, 
        font_size=10, 
        font_weight='bold'
    )
    
    # Add legend for entity types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=entity_type)
        for entity_type, color in type_colors.items()
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f'Entity Network (Top {top_n} Nodes by Degree)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# %%
def visualize_pagerank_distribution(vertices_with_pagerank, top_n=20):
    """
    Visualize the distribution of PageRank scores.
    
    Args:
        vertices_with_pagerank (DataFrame): DataFrame with PageRank scores
        top_n (int): Number of top entities to display
        
    Returns:
        None
    """
    if vertices_with_pagerank is None:
        print("No PageRank data available for visualization.")
        return
    
    # Convert to pandas for visualization
    vertices_pd = vertices_with_pagerank.toPandas()
    
    # Sort by PageRank
    vertices_pd = vertices_pd.sort_values('pagerank', ascending=False)
    
    # Get top N entities by PageRank
    top_entities = vertices_pd.head(top_n)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: PageRank distribution
    sns.histplot(vertices_pd['pagerank'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('PageRank Distribution')
    axes[0, 0].set_xlabel('PageRank')
    axes[0, 0].set_ylabel('Count')
    
    # Plot 2: Top entities by PageRank
    sns.barplot(x='pagerank', y='id', data=top_entities, ax=axes[0, 1])
    axes[0, 1].set_title(f'Top {top_n} Entities by PageRank')
    axes[0, 1].set_xlabel('PageRank')
    axes[0, 1].set_ylabel('Entity')
    
    # Plot 3: PageRank by entity type
    sns.boxplot(x='entity_type', y='pagerank', data=vertices_pd, ax=axes[1, 0])
    axes[1, 0].set_title('PageRank by Entity Type')
    axes[1, 0].set_xlabel('Entity Type')
    axes[1, 0].set_ylabel('PageRank')
    
    # Plot 4: PageRank vs Fake Ratio
    sns.scatterplot(x='fake_ratio', y='pagerank', hue='entity_type', data=vertices_pd, ax=axes[1, 1])
    axes[1, 1].set_title('PageRank vs Fake Ratio')
    axes[1, 1].set_xlabel('Fake Ratio')
    axes[1, 1].set_ylabel('PageRank')
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Data Storage Functions

# %%
def save_to_parquet(df, path, partition_by=None):
    """
    Save a DataFrame in Parquet format.
    
    Args:
        df (DataFrame): DataFrame to save
        path (str): Path where to save the DataFrame
        partition_by (str): Column to partition by (optional)
    """
    print(f"Saving DataFrame to {path}...")
    
    writer = df.write.mode("overwrite")
    
    if partition_by:
        writer = writer.partitionBy(partition_by)
    
    writer.parquet(path)
    print(f"DataFrame saved to {path}")

# %%
def save_to_hive_table(df, table_name, partition_by=None):
    """
    Save a DataFrame to a Hive table.
    
    Args:
        df (DataFrame): DataFrame to save
        table_name (str): Name of the Hive table to create or replace
        partition_by (str): Column to partition by (optional)
    """
    print(f"Saving DataFrame to Hive table {table_name}...")
    
    writer = df.write.mode("overwrite").format("parquet")
    
    if partition_by:
        writer = writer.partitionBy(partition_by)
    
    writer.saveAsTable(table_name)
    print(f"DataFrame saved to Hive table: {table_name}")

# %%
def save_network_metrics(metrics, path):
    """
    Save network metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary with network metrics
        path (str): Path where to save the metrics
    """
    print(f"Saving network metrics to {path}...")
    
    try:
        # Convert to JSON
        metrics_json = json.dumps(metrics, indent=2)
        
        # Write to file
        with open(path, 'w') as f:
            f.write(metrics_json)
        
        print(f"Network metrics saved to {path}")
    except Exception as e:
        print(f"Error saving network metrics: {e}")

# %% [markdown]
# ## Complete Graph Analysis Pipeline

# %%
def analyze_entity_graph(
    input_path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet",
    output_dir="dbfs:/FileStore/fake_news_detection/graph_data",
    min_entity_freq=2,
    min_edge_weight=2,
    top_n=20,
    create_tables=True
):
    """
    Complete pipeline for graph-based entity analysis.
    
    Args:
        input_path (str): Path to preprocessed data
        output_dir (str): Directory to save results
        min_entity_freq (int): Minimum frequency for entity to be included
        min_edge_weight (int): Minimum co-occurrence weight for edge to be included
        top_n (int): Number of top entities to display in visualizations
        create_tables (bool): Whether to create Hive tables
        
    Returns:
        dict: Dictionary with references to analysis results
    """
    print("Starting graph-based entity analysis pipeline...")
    start_time = time.time()
    
    # Create output directories
    try:
        dbutils.fs.mkdirs(output_dir.replace("dbfs:", ""))
    except:
        print("Warning: Could not create directories. This is expected in local environments.")
        os.makedirs(output_dir.replace("dbfs:/", "/tmp/"), exist_ok=True)
    
    # 1. Load preprocessed data
    df = load_preprocessed_data(input_path)
    if df is None:
        print("Error: Could not load preprocessed data. Pipeline aborted.")
        return None
    
    # 2. Extract named entities
    df_with_entities = extract_entities_from_dataframe(df, "text")
    
    # 3. Create entity nodes
    vertices = create_entity_nodes(df_with_entities, min_entity_freq)
    
    # 4. Create entity edges
    edges = create_entity_edges(df_with_entities, min_edge_weight)
    
    # 5. Visualize entity distribution
    visualize_entity_distribution(vertices, top_n)
    
    # 6. Create graph and run analysis
    results = {}
    
    if graphframes_available:
        # 6.1. GraphX analysis
        g = create_graphframe(vertices, edges)
        
        if g is not None:
            # Run PageRank
            vertices_with_pagerank, edges_with_pagerank = run_pagerank(g)
            results['vertices_with_pagerank'] = vertices_with_pagerank
            results['edges_with_pagerank'] = edges_with_pagerank
            
            # Visualize PageRank distribution
            visualize_pagerank_distribution(vertices_with_pagerank, top_n)
            
            # Run Connected Components
            components = run_connected_components(g)
            results['components'] = components
            
            # Run Triangle Count
            triangles = run_triangle_count(g)
            results['triangles'] = triangles
            
            # Save results
            if vertices_with_pagerank is not None:
                save_to_parquet(vertices_with_pagerank, f"{output_dir}/vertices_with_pagerank.parquet")
                if create_tables:
                    save_to_hive_table(vertices_with_pagerank, "entity_vertices_with_pagerank")
            
            if edges_with_pagerank is not None:
                save_to_parquet(edges_with_pagerank, f"{output_dir}/edges_with_pagerank.parquet")
                if create_tables:
                    save_to_hive_table(edges_with_pagerank, "entity_edges_with_pagerank")
    
    # 6.2. NetworkX analysis (as fallback or complement)
    G = create_networkx_graph(vertices, edges)
    results['networkx_graph'] = G
    
    # Calculate network metrics
    metrics = calculate_networkx_metrics(G)
    results['network_metrics'] = metrics
    
    # Calculate PageRank using NetworkX
    pagerank = calculate_pagerank_networkx(G)
    results['networkx_pagerank'] = pagerank
    
    # Visualize network
    visualize_network(G, top_n)
    
    # Save network metrics
    try:
        save_network_metrics(metrics, f"{output_dir.replace('dbfs:/', '/tmp/')}/network_metrics.json")
    except:
        print("Warning: Could not save network metrics to file. This is expected in Databricks.")
    
    # 7. Save basic entity data
    save_to_parquet(vertices, f"{output_dir}/entity_vertices.parquet")
    save_to_parquet(edges, f"{output_dir}/entity_edges.parquet")
    
    if create_tables:
        save_to_hive_table(vertices, "entity_vertices")
        save_to_hive_table(edges, "entity_edges")
    
    print(f"\nGraph-based entity analysis pipeline completed in {time.time() - start_time:.2f} seconds!")
    
    return results

# %% [markdown]
# ## Step-by-Step Tutorial

# %% [markdown]
# ### 1. Load Preprocessed Data

# %%
# Load preprocessed data
preprocessed_df = load_preprocessed_data()

# Display sample data
if preprocessed_df:
    print("Preprocessed data sample:")
    preprocessed_df.show(5, truncate=80)

# %% [markdown]
# ### 2. Extract Named Entities

# %%
# Extract named entities from text
if preprocessed_df:
    df_with_entities = extract_entities_from_dataframe(preprocessed_df, "text")
    
    # Display sample of extracted entities
    print("Sample of extracted entities:")
    df_with_entities.select("id", "label", "people", "places", "organizations", "events").show(5, truncate=50)

# %% [markdown]
# ### 3. Create Entity Nodes and Edges

# %%
# Create entity nodes
if 'df_with_entities' in locals():
    vertices = create_entity_nodes(df_with_entities, min_entity_freq=2)
    
    # Display sample of entity nodes
    print("Sample of entity nodes:")
    vertices.show(10, truncate=50)
    
    # Create entity edges
    edges = create_entity_edges(df_with_entities, min_edge_weight=2)
    
    # Display sample of entity edges
    print("Sample of entity edges:")
    edges.show(10, truncate=50)

# %% [markdown]
# ### 4. Visualize Entity Distribution

# %%
# Visualize entity distribution
if 'vertices' in locals():
    visualize_entity_distribution(vertices, top_n=20)

# %% [markdown]
# ### 5. GraphX Analysis (if available)

# %%
# Create GraphFrame and run GraphX analysis
if 'vertices' in locals() and 'edges' in locals() and graphframes_available:
    # Create GraphFrame
    g = create_graphframe(vertices, edges)
    
    if g is not None:
        # Run PageRank
        vertices_with_pagerank, edges_with_pagerank = run_pagerank(g)
        
        # Display sample of vertices with PageRank
        if vertices_with_pagerank is not None:
            print("Sample of vertices with PageRank:")
            vertices_with_pagerank.orderBy("pagerank", ascending=False).show(10, truncate=50)
        
        # Visualize PageRank distribution
        visualize_pagerank_distribution(vertices_with_pagerank, top_n=20)
        
        # Run Connected Components
        components = run_connected_components(g)
        
        # Display sample of components
        if components is not None:
            print("Sample of connected components:")
            components.show(10, truncate=50)
        
        # Run Triangle Count
        triangles = run_triangle_count(g)
        
        # Display sample of triangle counts
        if triangles is not None:
            print("Sample of triangle counts:")
            triangles.show(10, truncate=50)

# %% [markdown]
# ### 6. NetworkX Analysis (alternative to GraphX)

# %%
# Create NetworkX graph and run analysis
if 'vertices' in locals() and 'edges' in locals():
    # Create NetworkX graph
    G = create_networkx_graph(vertices, edges)
    
    # Calculate network metrics
    metrics = calculate_networkx_metrics(G)
    
    # Display network metrics
    print("Network metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Calculate PageRank using NetworkX
    pagerank = calculate_pagerank_networkx(G)
    
    # Display top entities by PageRank
    print("Top entities by PageRank (NetworkX):")
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    for entity, score in top_pagerank:
        print(f"{entity}: {score:.6f}")
    
    # Visualize network
    visualize_network(G, top_n=100)

# %% [markdown]
# ### 7. Save Results

# %%
# Save results to Parquet files and Hive tables
if 'vertices' in locals() and 'edges' in locals():
    # Save entity nodes and edges
    save_to_parquet(vertices, "dbfs:/FileStore/fake_news_detection/graph_data/entity_vertices.parquet")
    save_to_parquet(edges, "dbfs:/FileStore/fake_news_detection/graph_data/entity_edges.parquet")
    
    # Save PageRank results if available
    if 'vertices_with_pagerank' in locals() and vertices_with_pagerank is not None:
        save_to_parquet(vertices_with_pagerank, "dbfs:/FileStore/fake_news_detection/graph_data/vertices_with_pagerank.parquet")
    
    if 'edges_with_pagerank' in locals() and edges_with_pagerank is not None:
        save_to_parquet(edges_with_pagerank, "dbfs:/FileStore/fake_news_detection/graph_data/edges_with_pagerank.parquet")
    
    # Save network metrics if available
    if 'metrics' in locals():
        try:
            save_network_metrics(metrics, "/tmp/network_metrics.json")
            print("Network metrics saved to /tmp/network_metrics.json")
        except Exception as e:
            print(f"Error saving network metrics: {e}")

# %% [markdown]
# ### 8. Complete Pipeline

# %%
# Run the complete graph analysis pipeline
results = analyze_entity_graph(
    input_path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet",
    output_dir="dbfs:/FileStore/fake_news_detection/graph_data",
    min_entity_freq=2,
    min_edge_weight=2,
    top_n=20,
    create_tables=True
)

# %% [markdown]
# ## Important Notes
# 
# 1. **Graph Analysis Importance**: Graph-based analysis is crucial for fake news detection as it helps identify relationships between entities (people, places, organizations, events) that may indicate suspicious patterns.
# 
# 2. **GraphX vs NetworkX**: This notebook provides two implementation paths:
#    - GraphX (via GraphFrames) for distributed processing in Spark
#    - NetworkX as a fallback for environments without GraphX support
# 
# 3. **Entity Extraction**: We use a simplified entity extraction approach for demonstration. In production, use a proper NLP pipeline with SpaCy or similar tools.
# 
# 4. **Graph Algorithms**: We implement several graph algorithms:
#    - PageRank to identify influential entities
#    - Connected Components to find entity clusters
#    - Triangle Count to measure network cohesion
# 
# 5. **Visualization**: The notebook includes several visualization functions to help understand the entity network structure and distribution.
# 
# 6. **Performance Considerations**: Graph algorithms can be computationally expensive. For large datasets, consider:
#    - Increasing minimum entity frequency and edge weight thresholds
#    - Using sampling techniques
#    - Leveraging distributed processing with GraphX
# 
# 7. **Databricks Integration**: The code is optimized for Databricks Community Edition with appropriate configurations for memory and processing.
