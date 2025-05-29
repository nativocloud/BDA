"""
Script to implement and run a simplified baseline model for fake news detection.
"""

import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

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

# Split data into training and testing sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['label'], test_size=0.3, random_state=42
)

# Feature extraction with TF-IDF
print("Extracting features...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Random Forest classifier
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Make predictions
print("Making predictions...")
y_pred = rf_model.predict(X_test_tfidf)

# Evaluate the model
print("Evaluating model...")
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Detailed classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Save the model and vectorizer
print("Saving model and vectorizer...")
import pickle
with open(f"{models_dir}/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open(f"{models_dir}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save results
results = {
    "model": "Random Forest",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "report": report,
    "execution_time": time.time() - start_time
}

with open(f"{results_dir}/baseline_results.txt", "w") as f:
    f.write(f"Model: {results['model']}\n")
    f.write(f"Accuracy: {results['accuracy']:.4f}\n")
    f.write(f"Precision: {results['precision']:.4f}\n")
    f.write(f"Recall: {results['recall']:.4f}\n")
    f.write(f"F1 Score: {results['f1']:.4f}\n")
    f.write(f"Execution Time: {results['execution_time']:.2f} seconds\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Create a visualization of the results
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
sns.barplot(x=metrics, y=values)
plt.title('Random Forest Model Performance')
plt.ylim(0, 1)
plt.savefig(f"{results_dir}/baseline_performance.png")

print(f"Results saved to {results_dir}")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")

# Display feature importance
feature_names = vectorizer.get_feature_names_out()
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
})
top_features = feature_importance.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Top 20 Important Features')
plt.tight_layout()
plt.savefig(f"{results_dir}/feature_importance.png")

print("Feature importance visualization saved.")
print("Baseline model implementation completed successfully.")

# Last modified: May 29, 2025
