"""
Script to implement and run clustering analysis on the fake news dataset.
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.cm as cm

# Start timer
start_time = time.time()

# Define paths
data_dir = "/home/ubuntu/fake_news_detection/data"
models_dir = "/home/ubuntu/fake_news_detection/models"
results_dir = "/home/ubuntu/fake_news_detection/logs"

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("Loading data...")
# Load the sampled data
df = pd.read_csv(f"{data_dir}/news_sample.csv")

# Basic preprocessing
print("Preprocessing text...")
# Fill NaN values
df['text'] = df['text'].fillna('')
if 'title' in df.columns:
    df['title'] = df['title'].fillna('')
    # Combine title and text for better context
    df['content'] = df['title'] + " " + df['text']
else:
    df['content'] = df['text']

# Convert to lowercase
df['content'] = df['content'].str.lower()

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Feature extraction with TF-IDF
print("Extracting features...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['content'])
print(f"TF-IDF matrix shape: {X_tfidf.shape}")

# Save feature names for later use
feature_names = vectorizer.get_feature_names_out()

# Perform K-means clustering
print("Performing K-means clustering...")
# Determine optimal number of clusters using silhouette score
silhouette_scores = []
k_values = range(2, 11)  # Try 2 to 10 clusters

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_tfidf)
    silhouette_avg = silhouette_score(X_tfidf, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"K={k}, Silhouette Score={silhouette_avg:.4f}")

# Find optimal k
optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

# Run K-means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(X_tfidf)

# Visualize silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, 'o-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.savefig(f"{results_dir}/kmeans_silhouette_scores.png")

# Perform dimensionality reduction for visualization
print("Performing dimensionality reduction for visualization...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

# Visualize clusters with PCA
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['kmeans_cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('K-means Clusters (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig(f"{results_dir}/kmeans_clusters_pca.png")

# Visualize clusters with t-SNE for better separation
print("Performing t-SNE for better visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['kmeans_cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('K-means Clusters (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig(f"{results_dir}/kmeans_clusters_tsne.png")

# Analyze clusters
print("Analyzing clusters...")
cluster_stats = df.groupby('kmeans_cluster').agg({
    'label': ['count', 'mean'],  # mean of label gives proportion of real news (label=1)
})
cluster_stats.columns = ['count', 'real_proportion']
cluster_stats['fake_proportion'] = 1 - cluster_stats['real_proportion']
print(cluster_stats)

# Visualize cluster composition
plt.figure(figsize=(12, 6))
cluster_stats[['real_proportion', 'fake_proportion']].plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Cluster Composition (Real vs Fake News)')
plt.xlabel('Cluster')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(['Real News', 'Fake News'])
plt.savefig(f"{results_dir}/cluster_composition.png")

# Extract top terms for each cluster
print("Extracting top terms for each cluster...")
kmeans_order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
top_terms = {}

for cluster in range(optimal_k):
    top_terms[cluster] = [feature_names[ind] for ind in kmeans_order_centroids[cluster, :10]]
    print(f"Cluster {cluster}: {', '.join(top_terms[cluster])}")

# Visualize top terms
plt.figure(figsize=(15, 10))
for i in range(optimal_k):
    plt.subplot(int(np.ceil(optimal_k/2)), 2, i+1)
    y_pos = np.arange(len(top_terms[i]))
    plt.barh(y_pos, range(len(top_terms[i]), 0, -1))
    plt.yticks(y_pos, top_terms[i])
    plt.title(f'Cluster {i}')
    plt.tight_layout()
plt.savefig(f"{results_dir}/cluster_top_terms.png")

# Perform topic modeling with LDA
print("Performing topic modeling with LDA...")
n_topics = optimal_k  # Use same number as optimal clusters for comparison
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X_tfidf)

# Extract top terms for each topic
lda_top_terms = {}
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[:-11:-1]  # Get indices of top 10 terms
    lda_top_terms[topic_idx] = [feature_names[i] for i in top_indices]
    print(f"Topic {topic_idx}: {', '.join(lda_top_terms[topic_idx])}")

# Visualize LDA topics
plt.figure(figsize=(15, 10))
for i in range(n_topics):
    plt.subplot(int(np.ceil(n_topics/2)), 2, i+1)
    y_pos = np.arange(len(lda_top_terms[i]))
    plt.barh(y_pos, range(len(lda_top_terms[i]), 0, -1))
    plt.yticks(y_pos, lda_top_terms[i])
    plt.title(f'Topic {i}')
    plt.tight_layout()
plt.savefig(f"{results_dir}/lda_topics.png")

# Get document-topic distributions
doc_topic_dist = lda.transform(X_tfidf)
df['dominant_topic'] = np.argmax(doc_topic_dist, axis=1)

# Compare LDA topics with original labels
topic_stats = df.groupby('dominant_topic').agg({
    'label': ['count', 'mean'],  # mean of label gives proportion of real news (label=1)
})
topic_stats.columns = ['count', 'real_proportion']
topic_stats['fake_proportion'] = 1 - topic_stats['real_proportion']
print(topic_stats)

# Visualize topic composition
plt.figure(figsize=(12, 6))
topic_stats[['real_proportion', 'fake_proportion']].plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Topic Composition (Real vs Fake News)')
plt.xlabel('Topic')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(['Real News', 'Fake News'])
plt.savefig(f"{results_dir}/topic_composition.png")

# Compare K-means clusters with LDA topics
comparison = pd.crosstab(df['kmeans_cluster'], df['dominant_topic'])
print("K-means clusters vs LDA topics:")
print(comparison)

plt.figure(figsize=(10, 8))
sns.heatmap(comparison, annot=True, cmap='YlGnBu', fmt='d')
plt.title('K-means Clusters vs LDA Topics')
plt.xlabel('LDA Topic')
plt.ylabel('K-means Cluster')
plt.savefig(f"{results_dir}/kmeans_vs_lda.png")

# Create a dashboard-style visualization
plt.figure(figsize=(15, 12))

# Cluster distribution
plt.subplot(2, 2, 1)
cluster_counts = df['kmeans_cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.title('Documents per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')

# Topic distribution
plt.subplot(2, 2, 2)
topic_counts = df['dominant_topic'].value_counts().sort_index()
plt.bar(topic_counts.index, topic_counts.values)
plt.title('Documents per Topic')
plt.xlabel('Topic')
plt.ylabel('Count')

# t-SNE visualization with clusters
plt.subplot(2, 2, 3)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['kmeans_cluster'], cmap='viridis', alpha=0.7, s=30)
plt.colorbar(scatter, label='Cluster')
plt.title('K-means Clusters (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# t-SNE visualization with original labels
plt.subplot(2, 2, 4)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['label'], cmap='coolwarm', alpha=0.7, s=30)
plt.colorbar(scatter, label='Label (0=Fake, 1=Real)')
plt.title('Original Labels (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.tight_layout()
plt.savefig(f"{results_dir}/clustering_dashboard.png")

# Save clustering results
clustering_results = {
    "kmeans": {
        "optimal_k": int(optimal_k),
        "silhouette_scores": [float(score) for score in silhouette_scores],
        "cluster_stats": cluster_stats.to_dict(),
        "top_terms": top_terms
    },
    "lda": {
        "n_topics": int(n_topics),
        "topic_stats": topic_stats.to_dict(),
        "top_terms": lda_top_terms
    },
    "execution_time": time.time() - start_time
}

with open(f"{results_dir}/clustering_results.json", "w") as f:
    json.dump(clustering_results, f, indent=2)

print(f"Clustering analysis completed in {time.time() - start_time:.2f} seconds")
print(f"Results saved to {results_dir}")
