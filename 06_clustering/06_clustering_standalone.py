# %% [markdown]
# # Fake News Detection: Clustering Analysis
# 
# This notebook contains all the necessary code for clustering analysis in the fake news detection project. The code is organized into independent functions, without dependencies on external modules or classes, to facilitate execution in Databricks Community Edition.

# %% [markdown]
# ## Setup and Imports

# %%
# Import necessary libraries
import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.cm as cm

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType

# %%
# Initialize Spark session optimized for Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsDetection_ClusteringAnalysis") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

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
def convert_spark_to_pandas(spark_df, columns=None, limit=None):
    """
    Convert Spark DataFrame to Pandas DataFrame.
    
    Args:
        spark_df (DataFrame): Spark DataFrame to convert
        columns (list): List of columns to include (None for all)
        limit (int): Maximum number of rows to convert (None for all)
        
    Returns:
        DataFrame: Pandas DataFrame
    """
    print("Converting Spark DataFrame to Pandas DataFrame...")
    
    if spark_df is None:
        print("Error: Input DataFrame is None")
        return None
    
    try:
        # Select specified columns or all columns
        if columns:
            df = spark_df.select(columns)
        else:
            df = spark_df
        
        # Limit rows if specified
        if limit:
            df = df.limit(limit)
        
        # Convert to Pandas
        pandas_df = df.toPandas()
        
        print(f"Converted {len(pandas_df)} rows to Pandas DataFrame")
        return pandas_df
    
    except Exception as e:
        print(f"Error converting to Pandas DataFrame: {e}")
        return None

# %% [markdown]
# ### Text Preprocessing Functions

# %%
def preprocess_text(df, text_column="text", title_column=None):
    """
    Preprocess text data for clustering.
    
    Args:
        df (DataFrame): Pandas DataFrame with text data
        text_column (str): Name of the text column
        title_column (str): Name of the title column (optional)
        
    Returns:
        DataFrame: DataFrame with preprocessed text
    """
    print("Preprocessing text data...")
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Fill NaN values
    processed_df[text_column] = processed_df[text_column].fillna('')
    
    # Process title if available
    if title_column and title_column in processed_df.columns:
        processed_df[title_column] = processed_df[title_column].fillna('')
        # Combine title and text for better context
        processed_df['content'] = processed_df[title_column] + " " + processed_df[text_column]
    else:
        processed_df['content'] = processed_df[text_column]
    
    # Convert to lowercase
    processed_df['content'] = processed_df['content'].str.lower()
    
    print(f"Dataset shape: {processed_df.shape}")
    if 'label' in processed_df.columns:
        print(f"Class distribution:\n{processed_df['label'].value_counts()}")
    
    return processed_df

# %% [markdown]
# ### Feature Extraction Functions

# %%
def extract_tfidf_features(df, text_column="content", max_features=1000):
    """
    Extract TF-IDF features from text.
    
    Args:
        df (DataFrame): Pandas DataFrame with text data
        text_column (str): Name of the text column
        max_features (int): Maximum number of features to extract
        
    Returns:
        tuple: (X_tfidf, vectorizer) - TF-IDF matrix and vectorizer
    """
    print(f"Extracting TF-IDF features (max_features={max_features})...")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    
    # Fit and transform text data
    X_tfidf = vectorizer.fit_transform(df[text_column])
    
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
    
    # Get feature names for later use
    feature_names = vectorizer.get_feature_names_out()
    print(f"Number of features: {len(feature_names)}")
    
    return X_tfidf, vectorizer

# %% [markdown]
# ### Clustering Functions

# %%
def find_optimal_k(X, k_range=range(2, 11)):
    """
    Find optimal number of clusters using silhouette score.
    
    Args:
        X: Feature matrix
        k_range: Range of k values to try
        
    Returns:
        tuple: (optimal_k, silhouette_scores) - Optimal k and all scores
    """
    print("Finding optimal number of clusters using silhouette score...")
    
    silhouette_scores = []
    
    for k in k_range:
        print(f"Trying k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"K={k}, Silhouette Score={silhouette_avg:.4f}")
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    return optimal_k, silhouette_scores

# %%
def perform_kmeans_clustering(X, k):
    """
    Perform K-means clustering.
    
    Args:
        X: Feature matrix
        k: Number of clusters
        
    Returns:
        tuple: (kmeans, cluster_labels) - KMeans model and cluster labels
    """
    print(f"Performing K-means clustering with k={k}...")
    
    # Run K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    print(f"K-means clustering completed with {k} clusters")
    
    return kmeans, cluster_labels

# %%
def perform_dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering.
    
    Args:
        X: Feature matrix
        eps: Maximum distance between samples
        min_samples: Minimum number of samples in a neighborhood
        
    Returns:
        tuple: (dbscan, cluster_labels) - DBSCAN model and cluster labels
    """
    print(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    
    # Count number of clusters (excluding noise points with label -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"DBSCAN clustering completed with {n_clusters} clusters and {n_noise} noise points")
    
    return dbscan, cluster_labels

# %% [markdown]
# ### Topic Modeling Functions

# %%
def perform_lda_topic_modeling(X, n_topics):
    """
    Perform topic modeling with Latent Dirichlet Allocation.
    
    Args:
        X: Feature matrix
        n_topics: Number of topics
        
    Returns:
        tuple: (lda, doc_topic_dist) - LDA model and document-topic distributions
    """
    print(f"Performing topic modeling with LDA (n_topics={n_topics})...")
    
    # Run LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    # Get document-topic distributions
    doc_topic_dist = lda.transform(X)
    
    print(f"LDA topic modeling completed with {n_topics} topics")
    
    return lda, doc_topic_dist

# %%
def extract_top_terms(model, feature_names, n_top_words=10):
    """
    Extract top terms for each cluster or topic.
    
    Args:
        model: Clustering or topic model
        feature_names: List of feature names
        n_top_words: Number of top words to extract
        
    Returns:
        dict: Dictionary with top terms for each cluster/topic
    """
    print(f"Extracting top {n_top_words} terms for each cluster/topic...")
    
    top_terms = {}
    
    # Check if model is KMeans
    if hasattr(model, 'cluster_centers_'):
        # For KMeans
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        for cluster in range(len(order_centroids)):
            top_terms[cluster] = [feature_names[ind] for ind in order_centroids[cluster, :n_top_words]]
            print(f"Cluster {cluster}: {', '.join(top_terms[cluster])}")
    
    # Check if model is LDA
    elif hasattr(model, 'components_'):
        # For LDA
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[:-n_top_words-1:-1]
            top_terms[topic_idx] = [feature_names[i] for i in top_indices]
            print(f"Topic {topic_idx}: {', '.join(top_terms[topic_idx])}")
    
    else:
        print("Error: Unsupported model type")
        return None
    
    return top_terms

# %% [markdown]
# ### Dimensionality Reduction Functions

# %%
def perform_pca(X, n_components=2):
    """
    Perform Principal Component Analysis for dimensionality reduction.
    
    Args:
        X: Feature matrix
        n_components: Number of components
        
    Returns:
        array: Reduced feature matrix
    """
    print(f"Performing PCA with {n_components} components...")
    
    # Convert sparse matrix to dense if needed
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    # Run PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    
    print(f"PCA completed, explained variance ratio: {pca.explained_variance_ratio_}")
    
    return X_pca

# %%
def perform_tsne(X, n_components=2, perplexity=30):
    """
    Perform t-SNE for dimensionality reduction.
    
    Args:
        X: Feature matrix
        n_components: Number of components
        perplexity: Perplexity parameter for t-SNE
        
    Returns:
        array: Reduced feature matrix
    """
    print(f"Performing t-SNE with {n_components} components (perplexity={perplexity})...")
    
    # Convert sparse matrix to dense if needed
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    # Run t-SNE
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_dense)
    
    print("t-SNE completed")
    
    return X_tsne

# %% [markdown]
# ### Analysis Functions

# %%
def analyze_clusters(df, cluster_column):
    """
    Analyze clusters and their relationship with labels.
    
    Args:
        df: DataFrame with cluster assignments
        cluster_column: Name of the cluster column
        
    Returns:
        DataFrame: DataFrame with cluster statistics
    """
    print(f"Analyzing clusters from column '{cluster_column}'...")
    
    # Check if label column exists
    if 'label' not in df.columns:
        print("Warning: 'label' column not found, cannot analyze relationship with labels")
        cluster_stats = df.groupby(cluster_column).size().reset_index(name='count')
        return cluster_stats
    
    # Group by cluster and calculate statistics
    cluster_stats = df.groupby(cluster_column).agg({
        'label': ['count', 'mean'],  # mean of label gives proportion of real news (label=1)
    })
    
    # Flatten column names
    cluster_stats.columns = ['count', 'real_proportion']
    
    # Calculate fake proportion
    cluster_stats['fake_proportion'] = 1 - cluster_stats['real_proportion']
    
    # Reset index for easier handling
    cluster_stats = cluster_stats.reset_index()
    
    print(cluster_stats)
    
    return cluster_stats

# %%
def compare_clustering_methods(df, method1_column, method2_column):
    """
    Compare two clustering methods.
    
    Args:
        df: DataFrame with cluster assignments
        method1_column: Name of the first method's column
        method2_column: Name of the second method's column
        
    Returns:
        DataFrame: Cross-tabulation of the two methods
    """
    print(f"Comparing clustering methods: '{method1_column}' vs '{method2_column}'...")
    
    # Create cross-tabulation
    comparison = pd.crosstab(df[method1_column], df[method2_column])
    
    print(f"{method1_column} vs {method2_column}:")
    print(comparison)
    
    return comparison

# %% [markdown]
# ### Visualization Functions

# %%
def visualize_silhouette_scores(k_values, silhouette_scores, output_path=None):
    """
    Visualize silhouette scores for different k values.
    
    Args:
        k_values: Range of k values
        silhouette_scores: List of silhouette scores
        output_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Silhouette score plot saved to {output_path}")
    
    plt.show()

# %%
def visualize_clusters_2d(X_2d, labels, title, colormap='viridis', output_path=None):
    """
    Visualize clusters in 2D.
    
    Args:
        X_2d: 2D feature matrix
        labels: Cluster labels
        title: Plot title
        colormap: Colormap to use
        output_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=colormap, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Cluster visualization saved to {output_path}")
    
    plt.show()

# %%
def visualize_cluster_composition(cluster_stats, output_path=None):
    """
    Visualize cluster composition (real vs fake news).
    
    Args:
        cluster_stats: DataFrame with cluster statistics
        output_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 6))
    
    # Check if required columns exist
    if 'real_proportion' in cluster_stats.columns and 'fake_proportion' in cluster_stats.columns:
        # Set cluster column as index if it's not already
        if 'cluster' in cluster_stats.columns:
            cluster_stats = cluster_stats.set_index('cluster')
        
        # Plot stacked bar chart
        cluster_stats[['real_proportion', 'fake_proportion']].plot(
            kind='bar', stacked=True, colormap='coolwarm'
        )
        plt.title('Cluster Composition (Real vs Fake News)')
        plt.xlabel('Cluster')
        plt.ylabel('Proportion')
        plt.xticks(rotation=0)
        plt.legend(['Real News', 'Fake News'])
        
        if output_path:
            plt.savefig(output_path)
            print(f"Cluster composition plot saved to {output_path}")
        
        plt.show()
    else:
        print("Error: Required columns not found in cluster_stats")

# %%
def visualize_top_terms(top_terms, output_path=None):
    """
    Visualize top terms for each cluster or topic.
    
    Args:
        top_terms: Dictionary with top terms
        output_path: Path to save the figure (optional)
    """
    n_clusters = len(top_terms)
    plt.figure(figsize=(15, 10))
    
    for i in range(n_clusters):
        plt.subplot(int(np.ceil(n_clusters/2)), 2, i+1)
        y_pos = np.arange(len(top_terms[i]))
        plt.barh(y_pos, range(len(top_terms[i]), 0, -1))
        plt.yticks(y_pos, top_terms[i])
        plt.title(f'Cluster/Topic {i}')
        plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Top terms visualization saved to {output_path}")
    
    plt.show()

# %%
def visualize_clustering_dashboard(df, X_2d, cluster_column='cluster', output_path=None):
    """
    Create a dashboard-style visualization of clustering results.
    
    Args:
        df: DataFrame with cluster assignments
        X_2d: 2D feature matrix (e.g., from t-SNE)
        cluster_column: Name of the cluster column
        output_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(15, 12))
    
    # Cluster distribution
    plt.subplot(2, 2, 1)
    cluster_counts = df[cluster_column].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.title('Documents per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    
    # Check if topic column exists
    if 'dominant_topic' in df.columns:
        # Topic distribution
        plt.subplot(2, 2, 2)
        topic_counts = df['dominant_topic'].value_counts().sort_index()
        plt.bar(topic_counts.index, topic_counts.values)
        plt.title('Documents per Topic')
        plt.xlabel('Topic')
        plt.ylabel('Count')
    
    # t-SNE visualization with clusters
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df[cluster_column], cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Clusters (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Check if label column exists
    if 'label' in df.columns:
        # t-SNE visualization with original labels
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['label'], cmap='coolwarm', alpha=0.7, s=30)
        plt.colorbar(scatter, label='Label (0=Fake, 1=Real)')
        plt.title('Original Labels (t-SNE)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Clustering dashboard saved to {output_path}")
    
    plt.show()

# %% [markdown]
# ### Data Storage Functions

# %%
def save_model(model, path):
    """
    Save a model to disk.
    
    Args:
        model: Model to save
        path: Path where to save the model
    """
    print(f"Saving model to {path}...")
    
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")

# %%
def save_results(df, path):
    """
    Save clustering results to disk.
    
    Args:
        df: DataFrame with clustering results
        path: Path where to save the results
    """
    print(f"Saving clustering results to {path}...")
    
    try:
        df.to_csv(path, index=False)
        print(f"Results saved to {path}")
    except Exception as e:
        print(f"Error saving results: {e}")

# %%
def save_to_parquet(df, path):
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: Spark DataFrame to save
        path: Path where to save the DataFrame
    """
    print(f"Saving DataFrame to {path}...")
    
    try:
        df.write.mode("overwrite").parquet(path)
        print(f"DataFrame saved to {path}")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")

# %% [markdown]
# ## Complete Clustering Pipeline

# %%
def perform_clustering_analysis(
    input_path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet",
    output_dir="dbfs:/FileStore/fake_news_detection/clustering_data",
    max_features=1000,
    k_range=range(2, 11),
    sample_size=None
):
    """
    Complete pipeline for clustering analysis.
    
    Args:
        input_path (str): Path to preprocessed data
        output_dir (str): Directory to save results
        max_features (int): Maximum number of features for TF-IDF
        k_range (range): Range of k values to try
        sample_size (int): Number of samples to use (None for all)
        
    Returns:
        dict: Dictionary with references to analysis results
    """
    print("Starting clustering analysis pipeline...")
    start_time = time.time()
    
    # Create output directories
    try:
        dbutils.fs.mkdirs(output_dir.replace("dbfs:", ""))
    except:
        print("Warning: Could not create directories. This is expected in local environments.")
        os.makedirs(output_dir.replace("dbfs:/", "/tmp/"), exist_ok=True)
    
    # 1. Load preprocessed data
    spark_df = load_preprocessed_data(input_path)
    if spark_df is None:
        print("Error: Could not load preprocessed data. Pipeline aborted.")
        return None
    
    # 2. Convert to Pandas DataFrame
    df = convert_spark_to_pandas(spark_df, limit=sample_size)
    if df is None:
        print("Error: Could not convert to Pandas DataFrame. Pipeline aborted.")
        return None
    
    # 3. Preprocess text
    df = preprocess_text(df, text_column="text", title_column="title")
    
    # 4. Extract TF-IDF features
    X_tfidf, vectorizer = extract_tfidf_features(df, text_column="content", max_features=max_features)
    
    # 5. Find optimal number of clusters
    optimal_k, silhouette_scores = find_optimal_k(X_tfidf, k_range)
    
    # 6. Visualize silhouette scores
    visualize_silhouette_scores(
        k_range, silhouette_scores, 
        output_path=f"{output_dir.replace('dbfs:/', '/tmp/')}/silhouette_scores.png"
    )
    
    # 7. Perform K-means clustering
    kmeans, cluster_labels = perform_kmeans_clustering(X_tfidf, optimal_k)
    df['kmeans_cluster'] = cluster_labels
    
    # 8. Extract top terms for each cluster
    feature_names = vectorizer.get_feature_names_out()
    kmeans_top_terms = extract_top_terms(kmeans, feature_names)
    
    # 9. Visualize top terms
    visualize_top_terms(
        kmeans_top_terms, 
        output_path=f"{output_dir.replace('dbfs:/', '/tmp/')}/cluster_top_terms.png"
    )
    
    # 10. Perform dimensionality reduction for visualization
    X_pca = perform_pca(X_tfidf)
    X_tsne = perform_tsne(X_tfidf)
    
    # 11. Visualize clusters
    visualize_clusters_2d(
        X_pca, cluster_labels, "K-means Clusters (PCA)", 
        output_path=f"{output_dir.replace('dbfs:/', '/tmp/')}/kmeans_clusters_pca.png"
    )
    visualize_clusters_2d(
        X_tsne, cluster_labels, "K-means Clusters (t-SNE)", 
        output_path=f"{output_dir.replace('dbfs:/', '/tmp/')}/kmeans_clusters_tsne.png"
    )
    
    # 12. Analyze clusters
    cluster_stats = analyze_clusters(df, 'kmeans_cluster')
    
    # 13. Visualize cluster composition
    visualize_cluster_composition(
        cluster_stats, 
        output_path=f"{output_dir.replace('dbfs:/', '/tmp/')}/cluster_composition.png"
    )
    
    # 14. Perform topic modeling with LDA
    lda, doc_topic_dist = perform_lda_topic_modeling(X_tfidf, optimal_k)
    df['dominant_topic'] = np.argmax(doc_topic_dist, axis=1)
    
    # 15. Extract top terms for each topic
    lda_top_terms = extract_top_terms(lda, feature_names)
    
    # 16. Visualize top terms for topics
    visualize_top_terms(
        lda_top_terms, 
        output_path=f"{output_dir.replace('dbfs:/', '/tmp/')}/lda_topics.png"
    )
    
    # 17. Analyze topics
    topic_stats = analyze_clusters(df, 'dominant_topic')
    
    # 18. Visualize topic composition
    visualize_cluster_composition(
        topic_stats, 
        output_path=f"{output_dir.replace('dbfs:/', '/tmp/')}/topic_composition.png"
    )
    
    # 19. Compare K-means clusters with LDA topics
    comparison = compare_clustering_methods(df, 'kmeans_cluster', 'dominant_topic')
    
    # 20. Create dashboard visualization
    visualize_clustering_dashboard(
        df, X_tsne, 'kmeans_cluster', 
        output_path=f"{output_dir.replace('dbfs:/', '/tmp/')}/clustering_dashboard.png"
    )
    
    # 21. Save models and results
    try:
        save_model(
            kmeans, 
            f"{output_dir.replace('dbfs:/', '/tmp/')}/kmeans_model.pkl"
        )
        save_model(
            lda, 
            f"{output_dir.replace('dbfs:/', '/tmp/')}/lda_model.pkl"
        )
        save_model(
            vectorizer, 
            f"{output_dir.replace('dbfs:/', '/tmp/')}/tfidf_vectorizer.pkl"
        )
        save_results(
            df, 
            f"{output_dir.replace('dbfs:/', '/tmp/')}/clustering_results.csv"
        )
    except:
        print("Warning: Could not save models and results to local files. This is expected in Databricks.")
    
    # 22. Convert results back to Spark DataFrame for storage
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        results_df = spark.createDataFrame(df)
        save_to_parquet(results_df, f"{output_dir}/clustering_results.parquet")
    except Exception as e:
        print(f"Warning: Could not save results to Parquet: {e}")
    
    print(f"\nClustering analysis pipeline completed in {time.time() - start_time:.2f} seconds!")
    
    return {
        "kmeans_model": kmeans,
        "lda_model": lda,
        "vectorizer": vectorizer,
        "optimal_k": optimal_k,
        "silhouette_scores": silhouette_scores,
        "cluster_stats": cluster_stats,
        "topic_stats": topic_stats,
        "comparison": comparison,
        "results_df": df
    }

# %% [markdown]
# ## Step-by-Step Tutorial

# %% [markdown]
# ### 1. Load and Preprocess Data

# %%
# Load preprocessed data
spark_df = load_preprocessed_data()

# Convert to Pandas DataFrame (limit to 5000 rows for demonstration)
if spark_df:
    df = convert_spark_to_pandas(spark_df, limit=5000)
    
    # Preprocess text
    if df is not None:
        df = preprocess_text(df, text_column="text", title_column="title")
        
        # Display sample data
        print("\nSample data after preprocessing:")
        print(df[['content', 'label']].head(3))

# %% [markdown]
# ### 2. Extract Features

# %%
# Extract TF-IDF features
if 'df' in locals() and df is not None:
    X_tfidf, vectorizer = extract_tfidf_features(df, text_column="content", max_features=1000)
    
    # Display feature names
    feature_names = vectorizer.get_feature_names_out()
    print("\nSample feature names:")
    print(feature_names[:20])

# %% [markdown]
# ### 3. Find Optimal Number of Clusters

# %%
# Find optimal number of clusters
if 'X_tfidf' in locals():
    # Use a smaller range for demonstration
    k_range = range(2, 7)
    optimal_k, silhouette_scores = find_optimal_k(X_tfidf, k_range)
    
    # Visualize silhouette scores
    visualize_silhouette_scores(k_range, silhouette_scores)

# %% [markdown]
# ### 4. Perform K-means Clustering

# %%
# Perform K-means clustering
if 'X_tfidf' in locals() and 'optimal_k' in locals():
    kmeans, cluster_labels = perform_kmeans_clustering(X_tfidf, optimal_k)
    
    # Add cluster labels to DataFrame
    df['kmeans_cluster'] = cluster_labels
    
    # Display cluster distribution
    print("\nCluster distribution:")
    print(df['kmeans_cluster'].value_counts().sort_index())

# %% [markdown]
# ### 5. Extract Top Terms for Each Cluster

# %%
# Extract top terms for each cluster
if 'kmeans' in locals() and 'feature_names' in locals():
    kmeans_top_terms = extract_top_terms(kmeans, feature_names)
    
    # Visualize top terms
    visualize_top_terms(kmeans_top_terms)

# %% [markdown]
# ### 6. Visualize Clusters

# %%
# Perform dimensionality reduction for visualization
if 'X_tfidf' in locals():
    # PCA for visualization
    X_pca = perform_pca(X_tfidf)
    
    # t-SNE for better separation
    X_tsne = perform_tsne(X_tfidf)
    
    # Visualize clusters with PCA
    if 'cluster_labels' in locals():
        visualize_clusters_2d(X_pca, cluster_labels, "K-means Clusters (PCA)")
        
        # Visualize clusters with t-SNE
        visualize_clusters_2d(X_tsne, cluster_labels, "K-means Clusters (t-SNE)")

# %% [markdown]
# ### 7. Analyze Clusters

# %%
# Analyze clusters
if 'df' in locals() and 'kmeans_cluster' in df.columns:
    cluster_stats = analyze_clusters(df, 'kmeans_cluster')
    
    # Visualize cluster composition
    visualize_cluster_composition(cluster_stats)

# %% [markdown]
# ### 8. Perform Topic Modeling with LDA

# %%
# Perform topic modeling with LDA
if 'X_tfidf' in locals() and 'optimal_k' in locals():
    lda, doc_topic_dist = perform_lda_topic_modeling(X_tfidf, optimal_k)
    
    # Add dominant topic to DataFrame
    df['dominant_topic'] = np.argmax(doc_topic_dist, axis=1)
    
    # Display topic distribution
    print("\nTopic distribution:")
    print(df['dominant_topic'].value_counts().sort_index())

# %% [markdown]
# ### 9. Extract Top Terms for Each Topic

# %%
# Extract top terms for each topic
if 'lda' in locals() and 'feature_names' in locals():
    lda_top_terms = extract_top_terms(lda, feature_names)
    
    # Visualize top terms for topics
    visualize_top_terms(lda_top_terms)

# %% [markdown]
# ### 10. Analyze Topics

# %%
# Analyze topics
if 'df' in locals() and 'dominant_topic' in df.columns:
    topic_stats = analyze_clusters(df, 'dominant_topic')
    
    # Visualize topic composition
    visualize_cluster_composition(topic_stats)

# %% [markdown]
# ### 11. Compare K-means Clusters with LDA Topics

# %%
# Compare K-means clusters with LDA topics
if 'df' in locals() and 'kmeans_cluster' in df.columns and 'dominant_topic' in df.columns:
    comparison = compare_clustering_methods(df, 'kmeans_cluster', 'dominant_topic')
    
    # Visualize comparison
    plt.figure(figsize=(10, 8))
    sns.heatmap(comparison, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('K-means Clusters vs LDA Topics')
    plt.xlabel('LDA Topic')
    plt.ylabel('K-means Cluster')
    plt.show()

# %% [markdown]
# ### 12. Create Dashboard Visualization

# %%
# Create dashboard visualization
if 'df' in locals() and 'X_tsne' in locals() and 'kmeans_cluster' in df.columns:
    visualize_clustering_dashboard(df, X_tsne, 'kmeans_cluster')

# %% [markdown]
# ### 13. Save Models and Results

# %%
# Save models and results
if all(var in locals() for var in ['kmeans', 'lda', 'vectorizer', 'df']):
    # Create output directory
    output_dir = "/tmp/clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    save_model(kmeans, f"{output_dir}/kmeans_model.pkl")
    save_model(lda, f"{output_dir}/lda_model.pkl")
    save_model(vectorizer, f"{output_dir}/tfidf_vectorizer.pkl")
    
    # Save results
    save_results(df, f"{output_dir}/clustering_results.csv")
    
    print(f"Models and results saved to {output_dir}")

# %% [markdown]
# ### 14. Complete Pipeline

# %%
# Run the complete clustering analysis pipeline
results = perform_clustering_analysis(
    input_path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet",
    output_dir="dbfs:/FileStore/fake_news_detection/clustering_data",
    max_features=1000,
    k_range=range(2, 11),
    sample_size=5000  # Limit to 5000 samples for demonstration
)

# %% [markdown]
# ## Important Notes
# 
# 1. **Clustering Purpose**: Clustering helps identify natural groupings in the fake news dataset, revealing patterns that may not be apparent through supervised learning alone.
# 
# 2. **Feature Extraction**: TF-IDF vectorization is used to convert text into numerical features, capturing the importance of words in documents relative to the corpus.
# 
# 3. **Optimal Clusters**: The silhouette score is used to determine the optimal number of clusters, balancing cohesion within clusters and separation between clusters.
# 
# 4. **K-means vs LDA**: The notebook implements both K-means clustering (for direct grouping) and LDA topic modeling (for thematic analysis), providing complementary perspectives.
# 
# 5. **Visualization**: Multiple visualization techniques (PCA, t-SNE) are used to project high-dimensional data into 2D space for interpretation.
# 
# 6. **Cluster Analysis**: Each cluster is analyzed for its composition of fake vs. real news and its characteristic terms, helping identify thematic patterns.
# 
# 7. **Performance Considerations**: For large datasets, consider:
#    - Reducing the maximum number of features
#    - Using a sample of the data
#    - Limiting the range of k values to explore
# 
# 8. **Databricks Integration**: The code is optimized for Databricks Community Edition with appropriate configurations for memory and processing.
