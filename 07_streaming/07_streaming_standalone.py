# %% [markdown]
# # Fake News Detection: Streaming Analysis
# 
# This notebook contains all the necessary code for streaming analysis in the fake news detection project. The code is organized into independent functions, without dependencies on external modules or classes, to facilitate execution in Databricks Community Edition.

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
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, from_json, window, count, when
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, FloatType, TimestampType

# %%
# Initialize Spark session optimized for Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsDetection_StreamingAnalysis") \
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
def load_model(model_path):
    """
    Load a trained machine learning model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        object: Loaded model
    """
    print(f"Loading model from {model_path}...")
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# %%
def load_vectorizer(vectorizer_path):
    """
    Load a trained TF-IDF vectorizer from disk.
    
    Args:
        vectorizer_path (str): Path to the saved vectorizer file
        
    Returns:
        object: Loaded vectorizer
    """
    print(f"Loading vectorizer from {vectorizer_path}...")
    
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        print(f"Vectorizer loaded successfully from {vectorizer_path}")
        return vectorizer
    
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return None

# %%
def load_streaming_data(data_path):
    """
    Load streaming data from a CSV file.
    
    Args:
        data_path (str): Path to the streaming data file
        
    Returns:
        DataFrame: Pandas DataFrame with streaming data
    """
    print(f"Loading streaming data from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from {data_path}")
        return df
    
    except Exception as e:
        print(f"Error loading streaming data: {e}")
        return None

# %% [markdown]
# ### Text Preprocessing Functions

# %%
def preprocess_text(df, text_column="text", title_column=None):
    """
    Preprocess text data for streaming analysis.
    
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
    
    return processed_df

# %% [markdown]
# ### Batch Processing Functions

# %%
def process_batch(batch_df, vectorizer, model, batch_id):
    """
    Process a batch of streaming data.
    
    Args:
        batch_df (DataFrame): Pandas DataFrame with batch data
        vectorizer: TF-IDF vectorizer
        model: Trained machine learning model
        batch_id (int): Batch identifier
        
    Returns:
        DataFrame: DataFrame with prediction results
    """
    print(f"Processing batch {batch_id}...")
    
    # Preprocess text
    processed_df = preprocess_text(batch_df)
    
    # Extract features
    X = vectorizer.transform(processed_df['content'])
    
    # Make predictions
    processed_df['prediction'] = model.predict(X)
    processed_df['prediction_proba'] = model.predict_proba(X)[:, 1]
    processed_df['prediction_time'] = pd.Timestamp.now()
    
    print(f"Processed batch {batch_id} with {len(processed_df)} records")
    
    return processed_df

# %%
def save_batch_results(batch_df, output_dir, batch_id):
    """
    Save batch processing results to disk.
    
    Args:
        batch_df (DataFrame): Pandas DataFrame with batch results
        output_dir (str): Directory to save results
        batch_id (int): Batch identifier
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save batch results
    output_file = f"{output_dir}/batch_{batch_id}_results.csv"
    batch_df.to_csv(output_file, index=False)
    
    print(f"Batch {batch_id} results saved to {output_file}")

# %% [markdown]
# ### Streaming Simulation Functions

# %%
def simulate_streaming(stream_df, vectorizer, model, batch_size=5, delay_seconds=2, 
                      stream_dir=None, output_dir=None):
    """
    Simulate a streaming pipeline with batched data processing.
    
    Args:
        stream_df (DataFrame): Pandas DataFrame with streaming data
        vectorizer: TF-IDF vectorizer
        model: Trained machine learning model
        batch_size (int): Number of records per batch
        delay_seconds (int): Delay between batches in seconds
        stream_dir (str): Directory to save batch files
        output_dir (str): Directory to save results
        
    Returns:
        DataFrame: DataFrame with aggregated results
    """
    print(f"Starting streaming simulation with {len(stream_df)} records...")
    
    # Create directories if provided
    if stream_dir:
        os.makedirs(stream_dir, exist_ok=True)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Split data into batches
    num_batches = len(stream_df) // batch_size + (1 if len(stream_df) % batch_size > 0 else 0)
    
    all_results = []
    
    for i in range(num_batches):
        # Get batch
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(stream_df))
        batch = stream_df.iloc[start_idx:end_idx].copy()
        
        # Save batch to streaming directory if provided
        if stream_dir:
            batch_file = f"{stream_dir}/batch_{i}.csv"
            batch.to_csv(batch_file, index=False)
            print(f"Saved batch {i} to {batch_file}")
        
        # Process batch
        batch_results = process_batch(batch, vectorizer, model, i)
        
        # Save batch results if output directory is provided
        if output_dir:
            save_batch_results(batch_results, output_dir, i)
        
        # Extract key results for aggregation
        all_results.append(batch_results[['id', 'prediction', 'prediction_proba', 'prediction_time']])
        
        # Simulate delay between batches
        if i < num_batches - 1:
            print(f"Waiting {delay_seconds} seconds before next batch...")
            time.sleep(delay_seconds)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    print(f"Streaming simulation completed with {len(combined_results)} total records processed")
    
    return combined_results

# %% [markdown]
# ### Analysis Functions

# %%
def analyze_streaming_results(results_df):
    """
    Analyze streaming prediction results.
    
    Args:
        results_df (DataFrame): Pandas DataFrame with prediction results
        
    Returns:
        dict: Dictionary with analysis metrics
    """
    print("Analyzing streaming results...")
    
    # Calculate metrics
    total_processed = len(results_df)
    fake_count = len(results_df[results_df['prediction'] == 0])
    real_count = len(results_df[results_df['prediction'] == 1])
    
    # Create metrics dictionary
    metrics = {
        "total_processed": total_processed,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_percentage": fake_count / total_processed * 100 if total_processed > 0 else 0,
        "real_percentage": real_count / total_processed * 100 if total_processed > 0 else 0,
        "execution_time": time.time() - start_time
    }
    
    # Print summary
    print("\nStreaming Results Summary:")
    print(f"Total records processed: {metrics['total_processed']}")
    print(f"Fake news detected: {metrics['fake_count']} ({metrics['fake_percentage']:.1f}%)")
    print(f"Real news detected: {metrics['real_count']} ({metrics['real_percentage']:.1f}%)")
    print(f"Execution time: {metrics['execution_time']:.2f} seconds")
    
    return metrics

# %%
def create_time_series_data(results_df, resample_freq='1S'):
    """
    Create time series data from streaming results.
    
    Args:
        results_df (DataFrame): Pandas DataFrame with prediction results
        resample_freq (str): Frequency for resampling time series data
        
    Returns:
        DataFrame: DataFrame with time series data
    """
    print(f"Creating time series data with {resample_freq} frequency...")
    
    # Ensure prediction_time is datetime
    results_df['prediction_time'] = pd.to_datetime(results_df['prediction_time'])
    
    # Sort by time
    sorted_df = results_df.sort_values('prediction_time')
    
    # Group by time and count predictions
    time_series = sorted_df.set_index('prediction_time').resample(resample_freq).prediction.value_counts().unstack().fillna(0)
    
    # Ensure all prediction values are present
    if 0 not in time_series.columns:
        time_series[0] = 0
    if 1 not in time_series.columns:
        time_series[1] = 0
    
    return time_series

# %% [markdown]
# ### Visualization Functions

# %%
def visualize_prediction_distribution(results_df, output_path=None):
    """
    Visualize the distribution of predictions.
    
    Args:
        results_df (DataFrame): Pandas DataFrame with prediction results
        output_path (str): Path to save the visualization (optional)
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='prediction', data=results_df)
    plt.title('Streaming Predictions')
    plt.xlabel('Prediction (0=Fake, 1=Real)')
    plt.ylabel('Count')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Prediction distribution visualization saved to {output_path}")
    
    plt.show()

# %%
def visualize_time_series(time_series_df, output_path=None):
    """
    Visualize predictions over time.
    
    Args:
        time_series_df (DataFrame): Pandas DataFrame with time series data
        output_path (str): Path to save the visualization (optional)
    """
    plt.figure(figsize=(12, 6))
    time_series_df.plot(figsize=(12, 6))
    plt.title('Predictions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.legend(['Fake', 'Real'])
    
    if output_path:
        plt.savefig(output_path)
        print(f"Time series visualization saved to {output_path}")
    
    plt.show()

# %%
def visualize_probability_distribution(results_df, output_path=None):
    """
    Visualize the distribution of prediction probabilities.
    
    Args:
        results_df (DataFrame): Pandas DataFrame with prediction results
        output_path (str): Path to save the visualization (optional)
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['prediction_proba'], bins=10)
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Probability of being Real News')
    plt.ylabel('Count')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Probability distribution visualization saved to {output_path}")
    
    plt.show()

# %%
def create_streaming_dashboard(results_df, metrics, time_series_df, output_path=None):
    """
    Create a dashboard-style visualization of streaming results.
    
    Args:
        results_df (DataFrame): Pandas DataFrame with prediction results
        metrics (dict): Dictionary with analysis metrics
        time_series_df (DataFrame): Pandas DataFrame with time series data
        output_path (str): Path to save the visualization (optional)
    """
    plt.figure(figsize=(15, 10))
    
    # Predictions count
    plt.subplot(2, 2, 1)
    labels = ['Fake News', 'Real News']
    sizes = [metrics['fake_count'], metrics['real_count']]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.axis('equal')
    plt.title('Prediction Distribution')
    
    # Prediction probabilities histogram
    plt.subplot(2, 2, 2)
    sns.histplot(results_df['prediction_proba'], bins=10, ax=plt.gca())
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Probability of being Real News')
    plt.ylabel('Count')
    
    # Time series
    plt.subplot(2, 2, 3)
    time_series_df.plot(ax=plt.gca())
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
    Total records processed: {metrics['total_processed']}
    Fake news detected: {metrics['fake_count']} ({metrics['fake_percentage']:.1f}%)
    Real news detected: {metrics['real_count']} ({metrics['real_percentage']:.1f}%)
    Execution time: {metrics['execution_time']:.2f} seconds
    """
    plt.text(0.1, 0.5, summary_text, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Streaming dashboard saved to {output_path}")
    
    plt.show()

# %% [markdown]
# ### Data Storage Functions

# %%
def save_streaming_results(results_df, output_path):
    """
    Save streaming results to disk.
    
    Args:
        results_df (DataFrame): Pandas DataFrame with prediction results
        output_path (str): Path to save the results
    """
    print(f"Saving streaming results to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    results_df.to_csv(output_path, index=False)
    
    print(f"Streaming results saved to {output_path}")

# %%
def save_streaming_metrics(metrics, output_path):
    """
    Save streaming metrics to disk.
    
    Args:
        metrics (dict): Dictionary with analysis metrics
        output_path (str): Path to save the metrics
    """
    print(f"Saving streaming metrics to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save metrics
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Streaming metrics saved to {output_path}")

# %% [markdown]
# ### Spark Streaming Functions

# %%
def create_spark_streaming_schema():
    """
    Create schema for Spark Streaming data.
    
    Returns:
        StructType: Spark schema for streaming data
    """
    return StructType([
        StructField("id", IntegerType(), False),
        StructField("title", StringType(), True),
        StructField("text", StringType(), True),
        StructField("label", IntegerType(), True)
    ])

# %%
def setup_spark_streaming(spark, input_path, schema, processing_time="5 seconds"):
    """
    Set up a Spark Streaming context for processing streaming data.
    
    Args:
        spark: Spark session
        input_path (str): Path to the streaming data directory
        schema: Schema for the streaming data
        processing_time (str): Processing time interval
        
    Returns:
        DataFrame: Spark streaming DataFrame
    """
    print(f"Setting up Spark Streaming from {input_path}...")
    
    # Create streaming DataFrame
    streaming_df = spark \
        .readStream \
        .schema(schema) \
        .option("maxFilesPerTrigger", 1) \
        .option("latestFirst", "true") \
        .json(input_path) \
        .withWatermark("timestamp", "10 seconds")
    
    return streaming_df

# %%
def process_spark_streaming(streaming_df, output_path, query_name="streaming_query"):
    """
    Process streaming data with Spark Streaming.
    
    Args:
        streaming_df: Spark streaming DataFrame
        output_path (str): Path to save the streaming results
        query_name (str): Name of the streaming query
        
    Returns:
        StreamingQuery: Spark streaming query
    """
    print(f"Processing streaming data to {output_path}...")
    
    # Define processing logic
    query = streaming_df \
        .writeStream \
        .outputMode("append") \
        .format("json") \
        .option("path", output_path) \
        .option("checkpointLocation", f"{output_path}/checkpoint") \
        .queryName(query_name) \
        .start()
    
    return query

# %% [markdown]
# ## Complete Streaming Pipeline

# %%
def run_streaming_pipeline(
    model_path="dbfs:/FileStore/fake_news_detection/models/rf_model.pkl",
    vectorizer_path="dbfs:/FileStore/fake_news_detection/models/tfidf_vectorizer.pkl",
    data_path="dbfs:/FileStore/fake_news_detection/data/stream_sample.csv",
    stream_dir="dbfs:/FileStore/fake_news_detection/data/streaming",
    output_dir="dbfs:/FileStore/fake_news_detection/data/streaming_output",
    results_dir="dbfs:/FileStore/fake_news_detection/results",
    batch_size=5,
    delay_seconds=2
):
    """
    Run the complete streaming pipeline for fake news detection.
    
    Args:
        model_path (str): Path to the trained model
        vectorizer_path (str): Path to the trained vectorizer
        data_path (str): Path to the streaming data
        stream_dir (str): Directory to save batch files
        output_dir (str): Directory to save batch results
        results_dir (str): Directory to save analysis results
        batch_size (int): Number of records per batch
        delay_seconds (int): Delay between batches in seconds
        
    Returns:
        dict: Dictionary with references to results
    """
    print("Starting streaming pipeline...")
    start_time = time.time()
    
    # Create directories
    for directory in [stream_dir, output_dir, results_dir]:
        try:
            # For Databricks
            dbutils.fs.mkdirs(directory.replace("dbfs:", ""))
        except:
            # For local environment
            local_dir = directory.replace("dbfs:/", "/tmp/")
            os.makedirs(local_dir, exist_ok=True)
    
    # 1. Load model and vectorizer
    model = load_model(model_path.replace("dbfs:/", "/dbfs/"))
    vectorizer = load_vectorizer(vectorizer_path.replace("dbfs:/", "/dbfs/"))
    
    if model is None or vectorizer is None:
        print("Error: Could not load model or vectorizer. Pipeline aborted.")
        return None
    
    # 2. Load streaming data
    stream_df = load_streaming_data(data_path.replace("dbfs:/", "/dbfs/"))
    
    if stream_df is None:
        print("Error: Could not load streaming data. Pipeline aborted.")
        return None
    
    # 3. Run streaming simulation
    local_stream_dir = stream_dir.replace("dbfs:/", "/tmp/")
    local_output_dir = output_dir.replace("dbfs:/", "/tmp/")
    
    results = simulate_streaming(
        stream_df, 
        vectorizer, 
        model, 
        batch_size=batch_size, 
        delay_seconds=delay_seconds,
        stream_dir=local_stream_dir,
        output_dir=local_output_dir
    )
    
    # 4. Analyze results
    metrics = analyze_streaming_results(results)
    
    # 5. Create time series data
    time_series = create_time_series_data(results)
    
    # 6. Create visualizations
    local_results_dir = results_dir.replace("dbfs:/", "/tmp/")
    
    visualize_prediction_distribution(
        results, 
        output_path=f"{local_results_dir}/streaming_predictions.png"
    )
    
    visualize_time_series(
        time_series, 
        output_path=f"{local_results_dir}/streaming_time_series.png"
    )
    
    visualize_probability_distribution(
        results, 
        output_path=f"{local_results_dir}/streaming_probabilities.png"
    )
    
    create_streaming_dashboard(
        results, 
        metrics, 
        time_series, 
        output_path=f"{local_results_dir}/streaming_dashboard.png"
    )
    
    # 7. Save results
    save_streaming_results(
        results, 
        output_path=f"{local_output_dir}/aggregated_results.csv"
    )
    
    save_streaming_metrics(
        metrics, 
        output_path=f"{local_results_dir}/streaming_metrics.json"
    )
    
    print(f"\nStreaming pipeline completed in {time.time() - start_time:.2f} seconds!")
    print(f"Results saved to {output_dir}")
    print(f"Visualizations saved to {results_dir}")
    
    return {
        "results": results,
        "metrics": metrics,
        "time_series": time_series
    }

# %% [markdown]
# ## Step-by-Step Tutorial

# %% [markdown]
# ### 1. Load Model and Vectorizer

# %%
# Define paths
model_path = "/tmp/models/rf_model.pkl"
vectorizer_path = "/tmp/models/tfidf_vectorizer.pkl"
data_path = "/tmp/data/stream_sample.csv"
stream_dir = "/tmp/data/streaming"
output_dir = "/tmp/data/streaming_output"
results_dir = "/tmp/results"

# Create directories
for directory in ["/tmp/models", "/tmp/data", stream_dir, output_dir, results_dir]:
    os.makedirs(directory, exist_ok=True)

# Load model and vectorizer (if available)
model = None
vectorizer = None

try:
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)
except:
    print("Model or vectorizer not found. For demonstration, we'll create simple ones.")
    
    # Create a simple model and vectorizer for demonstration
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    
    # Sample data for training
    texts = [
        "This is fake news about politics",
        "Real news about economy",
        "Fake story about celebrities",
        "True report on science"
    ]
    labels = [0, 1, 0, 1]  # 0 = fake, 1 = real
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts)
    
    # Create and fit model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, labels)
    
    # Save model and vectorizer
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("Created and saved simple model and vectorizer for demonstration")

# %% [markdown]
# ### 2. Create Sample Streaming Data

# %%
# Create sample streaming data if not available
try:
    stream_df = load_streaming_data(data_path)
    if stream_df is None:
        raise FileNotFoundError("Sample data not found")
except:
    print("Creating sample streaming data...")
    
    # Create sample data
    sample_data = {
        "id": list(range(20)),
        "title": [
            "Breaking News: Political Scandal",
            "Economy Shows Signs of Recovery",
            "Celebrity Caught in Controversy",
            "New Scientific Discovery",
            "Government Announces New Policy",
            "Stock Market Hits Record High",
            "Famous Actor in New Movie",
            "Health Study Reveals Surprising Results",
            "Sports Team Wins Championship",
            "Technology Company Launches Product",
            "Weather Forecast Predicts Storm",
            "Education Reform Proposed",
            "Environmental Crisis Worsens",
            "International Relations Strained",
            "Local Community Event Success",
            "Transportation System Upgrade",
            "Cultural Festival Announced",
            "Financial Markets Volatile",
            "Medical Breakthrough Reported",
            "Social Media Platform Changes"
        ],
        "text": [
            "Politicians involved in major scandal with evidence of corruption.",
            "Economic indicators show positive trends in job growth and consumer spending.",
            "Famous celebrity caught in scandal with questionable evidence.",
            "Scientists discover new species with potential medical applications.",
            "Government announces controversial policy that divides public opinion.",
            "Stock market reaches unprecedented levels despite economic concerns.",
            "A-list actor stars in upcoming blockbuster movie with record budget.",
            "New health study contradicts previous findings on popular supplements.",
            "Underdog team wins championship in surprising upset victory.",
            "Major tech company releases innovative product to mixed reviews.",
            "Meteorologists predict severe weather conditions for the weekend.",
            "Lawmakers propose significant changes to education system.",
            "Environmental scientists report accelerating climate change effects.",
            "Diplomatic tensions rise between countries after controversial statement.",
            "Local community event raises record funds for charity.",
            "City approves major upgrade to public transportation infrastructure.",
            "Annual cultural festival announces lineup of international artists.",
            "Financial markets experience significant fluctuations due to uncertainty.",
            "Researchers announce promising results in treatment development.",
            "Popular social media platform implements controversial policy changes."
        ],
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = fake, 1 = real
    }
    
    stream_df = pd.DataFrame(sample_data)
    
    # Save sample data
    stream_df.to_csv(data_path, index=False)
    
    print(f"Created and saved sample streaming data with {len(stream_df)} records")

# Display sample data
print("\nSample streaming data:")
print(stream_df.head(3))

# %% [markdown]
# ### 3. Process a Single Batch

# %%
# Process a single batch
if stream_df is not None and model is not None and vectorizer is not None:
    # Get first batch
    batch_size = 5
    first_batch = stream_df.iloc[:batch_size].copy()
    
    # Process batch
    batch_results = process_batch(first_batch, vectorizer, model, batch_id=0)
    
    # Display results
    print("\nBatch processing results:")
    print(batch_results[['id', 'title', 'prediction', 'prediction_proba']].head())
    
    # Save batch results
    save_batch_results(batch_results, output_dir, batch_id=0)

# %% [markdown]
# ### 4. Run Streaming Simulation

# %%
# Run streaming simulation with small batch size and minimal delay
if stream_df is not None and model is not None and vectorizer is not None:
    # Run simulation
    results = simulate_streaming(
        stream_df, 
        vectorizer, 
        model, 
        batch_size=5, 
        delay_seconds=1,  # Short delay for demonstration
        stream_dir=stream_dir,
        output_dir=output_dir
    )
    
    # Display results
    print("\nStreaming simulation results:")
    print(results.head())

# %% [markdown]
# ### 5. Analyze Streaming Results

# %%
# Analyze streaming results
if 'results' in locals():
    # Calculate metrics
    metrics = analyze_streaming_results(results)
    
    # Create time series data
    time_series = create_time_series_data(results)
    
    # Display time series data
    print("\nTime series data:")
    print(time_series.head())

# %% [markdown]
# ### 6. Visualize Results

# %%
# Create visualizations
if 'results' in locals() and 'metrics' in locals() and 'time_series' in locals():
    # Prediction distribution
    visualize_prediction_distribution(results)
    
    # Time series
    visualize_time_series(time_series)
    
    # Probability distribution
    visualize_probability_distribution(results)
    
    # Dashboard
    create_streaming_dashboard(results, metrics, time_series)

# %% [markdown]
# ### 7. Save Results

# %%
# Save results
if 'results' in locals() and 'metrics' in locals():
    # Save streaming results
    save_streaming_results(results, f"{output_dir}/aggregated_results.csv")
    
    # Save metrics
    save_streaming_metrics(metrics, f"{results_dir}/streaming_metrics.json")

# %% [markdown]
# ### 8. Complete Pipeline

# %%
# Run the complete streaming pipeline
pipeline_results = run_streaming_pipeline(
    model_path=model_path,
    vectorizer_path=vectorizer_path,
    data_path=data_path,
    stream_dir=stream_dir,
    output_dir=output_dir,
    results_dir=results_dir,
    batch_size=5,
    delay_seconds=1  # Short delay for demonstration
)

# %% [markdown]
# ## Important Notes
# 
# 1. **Streaming Purpose**: Streaming analysis enables real-time detection of fake news as it appears, allowing for immediate response and mitigation.
# 
# 2. **Batch Processing**: The implementation uses a batch processing approach to simulate streaming, which is well-suited for Databricks Community Edition.
# 
# 3. **Model Loading**: Pre-trained models are loaded from disk, allowing the streaming pipeline to leverage previously trained classifiers.
# 
# 4. **Visualization**: Real-time visualizations help monitor the stream of news and track trends in fake news detection.
# 
# 5. **Time Series Analysis**: Tracking predictions over time helps identify patterns and potential coordinated fake news campaigns.
# 
# 6. **Performance Considerations**: For production environments, consider:
#    - Using Spark Structured Streaming for true streaming capabilities
#    - Implementing checkpointing for fault tolerance
#    - Optimizing batch size based on available resources
# 
# 7. **Databricks Integration**: The code is optimized for Databricks Community Edition with appropriate configurations for memory and processing.
# 
# 8. **Simulation vs. Real Streaming**: This notebook demonstrates a simulation of streaming; in a production environment, you would connect to real data sources like Twitter API, RSS feeds, or news APIs.
