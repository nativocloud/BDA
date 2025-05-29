"""
Script to demonstrate a minimal streaming pipeline for fake news detection.
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Start timer
start_time = time.time()

# Define paths
data_dir = "/home/ubuntu/fake_news_detection/data"
models_dir = "/home/ubuntu/fake_news_detection/models"
results_dir = "/home/ubuntu/fake_news_detection/logs"
stream_dir = "/home/ubuntu/fake_news_detection/data/streaming"
output_dir = "/home/ubuntu/fake_news_detection/data/streaming_output"

# Create directories if they don't exist
os.makedirs(stream_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

print("Loading model and vectorizer...")
# Load the trained model and vectorizer
with open(f"{models_dir}/rf_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(f"{models_dir}/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Loading streaming sample data...")
# Load the streaming sample data
stream_sample = pd.read_csv(f"{data_dir}/stream_sample.csv")

# Function to process a batch of data
def process_batch(batch_df, batch_id):
    print(f"Processing batch {batch_id}...")
    
    # Preprocess text
    if 'title' in batch_df.columns and 'text' in batch_df.columns:
        batch_df['content'] = batch_df['title'].fillna('') + " " + batch_df['text'].fillna('')
    else:
        batch_df['content'] = batch_df['text'].fillna('')
    
    batch_df['content'] = batch_df['content'].str.lower()
    
    # Extract features
    X = vectorizer.transform(batch_df['content'])
    
    # Make predictions
    batch_df['prediction'] = model.predict(X)
    batch_df['prediction_proba'] = model.predict_proba(X)[:, 1]
    batch_df['prediction_time'] = pd.Timestamp.now()
    
    # Save batch results
    output_file = f"{output_dir}/batch_{batch_id}_results.csv"
    batch_df.to_csv(output_file, index=False)
    
    # Return batch results for aggregation
    return batch_df[['id', 'prediction', 'prediction_proba', 'prediction_time']]

# Function to simulate streaming
def simulate_streaming(stream_df, batch_size=5, delay_seconds=2):
    print(f"Starting streaming simulation with {len(stream_df)} records...")
    
    # Split data into batches
    num_batches = len(stream_df) // batch_size + (1 if len(stream_df) % batch_size > 0 else 0)
    
    all_results = []
    
    for i in range(num_batches):
        # Get batch
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(stream_df))
        batch = stream_df.iloc[start_idx:end_idx].copy()
        
        # Save batch to streaming directory
        batch_file = f"{stream_dir}/batch_{i}.csv"
        batch.to_csv(batch_file, index=False)
        print(f"Saved batch {i} to {batch_file}")
        
        # Process batch
        batch_results = process_batch(batch, i)
        all_results.append(batch_results)
        
        # Simulate delay between batches
        if i < num_batches - 1:
            print(f"Waiting {delay_seconds} seconds before next batch...")
            time.sleep(delay_seconds)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    return combined_results

# Run streaming simulation
results = simulate_streaming(stream_sample, batch_size=5, delay_seconds=2)

# Analyze results
print("\nStreaming Results Summary:")
print(f"Total records processed: {len(results)}")
print(f"Fake news detected: {len(results[results['prediction'] == 0])}")
print(f"Real news detected: {len(results[results['prediction'] == 1])}")

# Save aggregated results
results.to_csv(f"{output_dir}/aggregated_results.csv", index=False)

# Create visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x='prediction', data=results)
plt.title('Streaming Predictions')
plt.xlabel('Prediction (0=Fake, 1=Real)')
plt.ylabel('Count')
plt.savefig(f"{results_dir}/streaming_predictions.png")

# Create time series visualization
results['prediction_time'] = pd.to_datetime(results['prediction_time'])
results = results.sort_values('prediction_time')

# Group by minute and count predictions
time_series = results.set_index('prediction_time').resample('1S').prediction.value_counts().unstack().fillna(0)
if 0 not in time_series.columns:
    time_series[0] = 0
if 1 not in time_series.columns:
    time_series[1] = 0

plt.figure(figsize=(12, 6))
time_series.plot(figsize=(12, 6))
plt.title('Predictions Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend(['Fake', 'Real'])
plt.savefig(f"{results_dir}/streaming_time_series.png")

# Save results for Grafana
streaming_metrics = {
    "total_processed": len(results),
    "fake_count": int(len(results[results['prediction'] == 0])),
    "real_count": int(len(results[results['prediction'] == 1])),
    "fake_percentage": float(len(results[results['prediction'] == 0]) / len(results) * 100),
    "real_percentage": float(len(results[results['prediction'] == 1]) / len(results) * 100),
    "execution_time": time.time() - start_time
}

with open(f"{results_dir}/streaming_metrics.json", "w") as f:
    json.dump(streaming_metrics, f)

# Create a simple dashboard-like visualization
plt.figure(figsize=(15, 10))

# Predictions count
plt.subplot(2, 2, 1)
labels = ['Fake News', 'Real News']
sizes = [streaming_metrics['fake_count'], streaming_metrics['real_count']]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.axis('equal')
plt.title('Prediction Distribution')

# Prediction probabilities histogram
plt.subplot(2, 2, 2)
sns.histplot(results['prediction_proba'], bins=10)
plt.title('Prediction Probability Distribution')
plt.xlabel('Probability of being Real News')
plt.ylabel('Count')

# Time series
plt.subplot(2, 2, 3)
time_series.plot(ax=plt.gca())
plt.title('Predictions Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend(['Fake', 'Real'])

# Summary text
plt.subplot(2, 2, 4)
plt.axis('off')
summary_text = f"""
Streaming Pipeline Summary:
---------------------------
Total records processed: {streaming_metrics['total_processed']}
Fake news detected: {streaming_metrics['fake_count']} ({streaming_metrics['fake_percentage']:.1f}%)
Real news detected: {streaming_metrics['real_count']} ({streaming_metrics['real_percentage']:.1f}%)
Execution time: {streaming_metrics['execution_time']:.2f} seconds
"""
plt.text(0.1, 0.5, summary_text, fontsize=12)

plt.tight_layout()
plt.savefig(f"{results_dir}/streaming_dashboard.png")

print(f"\nStreaming simulation completed in {time.time() - start_time:.2f} seconds")
print(f"Results saved to {output_dir}")
print(f"Visualizations saved to {results_dir}")

# Last modified: May 29, 2025
