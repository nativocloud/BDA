"""
Script to implement and run a simplified Naive Bayes model for fake news detection.
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import json

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

# Train a Naive Bayes classifier
print("Training Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Make predictions
print("Making predictions...")
y_pred = nb_model.predict(X_test_tfidf)
y_pred_prob = nb_model.predict_proba(X_test_tfidf)[:, 1]

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
with open(f"{models_dir}/nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)
with open(f"{models_dir}/nb_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save results
results = {
    "model": "Naive Bayes",
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "execution_time": time.time() - start_time
}

# Save results as text
with open(f"{results_dir}/nb_results.txt", "w") as f:
    f.write(f"Model: {results['model']}\n")
    f.write(f"Accuracy: {results['accuracy']:.4f}\n")
    f.write(f"Precision: {results['precision']:.4f}\n")
    f.write(f"Recall: {results['recall']:.4f}\n")
    f.write(f"F1 Score: {results['f1']:.4f}\n")
    f.write(f"Execution Time: {results['execution_time']:.2f} seconds\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Save results as JSON for Grafana
with open(f"{results_dir}/nb_results.json", "w") as f:
    json.dump(results, f)

# Create visualizations of the results
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
sns.barplot(x=metrics, y=values)
plt.title('Naive Bayes Model Performance')
plt.ylim(0, 1)
plt.savefig(f"{results_dir}/nb_performance.png")

# Create a comparison with the Random Forest model
# Load Random Forest results if available
rf_results_path = f"{results_dir}/baseline_results.txt"
if os.path.exists(rf_results_path):
    with open(rf_results_path, 'r') as f:
        rf_content = f.read()
        rf_accuracy = float(rf_content.split('Accuracy: ')[1].split('\n')[0])
        rf_precision = float(rf_content.split('Precision: ')[1].split('\n')[0])
        rf_recall = float(rf_content.split('Recall: ')[1].split('\n')[0])
        rf_f1 = float(rf_content.split('F1 Score: ')[1].split('\n')[0])
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    
    # Set up data for comparison
    models = ['Random Forest', 'Naive Bayes']
    accuracy_values = [rf_accuracy, accuracy]
    precision_values = [rf_precision, precision]
    recall_values = [rf_recall, recall]
    f1_values = [rf_f1, f1]
    
    # Set width of bars
    barWidth = 0.2
    
    # Set position of bars on X axis
    r1 = np.arange(len(models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    
    # Create bars
    plt.bar(r1, accuracy_values, width=barWidth, label='Accuracy')
    plt.bar(r2, precision_values, width=barWidth, label='Precision')
    plt.bar(r3, recall_values, width=barWidth, label='Recall')
    plt.bar(r4, f1_values, width=barWidth, label='F1 Score')
    
    # Add labels and legend
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Comparison')
    plt.xticks([r + barWidth*1.5 for r in range(len(models))], models)
    plt.legend()
    plt.ylim(0, 1)
    
    # Save comparison plot
    plt.savefig(f"{results_dir}/model_comparison.png")
    
    # Save comparison data for Grafana
    comparison_data = {
        "models": models,
        "metrics": {
            "accuracy": accuracy_values,
            "precision": precision_values,
            "recall": recall_values,
            "f1": f1_values
        }
    }
    
    with open(f"{results_dir}/model_comparison.json", "w") as f:
        json.dump(comparison_data, f)

print(f"Results saved to {results_dir}")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")
print("Naive Bayes model implementation completed successfully.")
