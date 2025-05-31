# %% [markdown]
# # Fake News Detection: Traditional Machine Learning Models
# 
# This notebook contains all the necessary code for implementing traditional machine learning models in the fake news detection project. The code is organized into independent functions, without dependencies on external modules or classes, to facilitate execution in Databricks Community Edition.

# %% [markdown]
# ## Setup and Imports

# %%
# Import necessary libraries
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, regexp_replace, length, udf, concat_ws, lower, explode, array
)
from pyspark.sql.types import StringType, IntegerType, DoubleType, ArrayType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# %%
# Initialize Spark session optimized for Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsDetection_TraditionalModels") \
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
def load_data_from_hive(fake_table_name="fake", true_table_name="real"):
    """
    Load data from Hive tables.
    
    Args:
        fake_table_name (str): Name of the Hive table with fake news
        true_table_name (str): Name of the Hive table with real news
        
    Returns:
        tuple: (real_df, fake_df) DataFrames with loaded data
    """
    print(f"Loading data from Hive tables '{true_table_name}' and '{fake_table_name}'...")
    
    # Check if tables exist
    tables = [row.tableName for row in spark.sql("SHOW TABLES").collect()]
    
    if true_table_name not in tables or fake_table_name not in tables:
        raise ValueError(f"Hive tables '{true_table_name}' and/or '{fake_table_name}' do not exist")
    
    # Load data from Hive tables
    real_df = spark.table(true_table_name)
    fake_df = spark.table(fake_table_name)
    
    # Register as temporary views for SQL queries
    real_df.createOrReplaceTempView("real_news")
    fake_df.createOrReplaceTempView("fake_news")
    
    # Display information about DataFrames
    print(f"Real news loaded: {real_df.count()} records")
    print(f"Fake news loaded: {fake_df.count()} records")
    
    return real_df, fake_df

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
def combine_datasets(real_df, fake_df):
    """
    Combine real and fake news datasets with labels.
    
    Args:
        real_df (DataFrame): DataFrame with real news
        fake_df (DataFrame): DataFrame with fake news
        
    Returns:
        DataFrame: Combined DataFrame with label column
    """
    # Add label column (1 for real, 0 for fake)
    real_with_label = real_df.withColumn("label", lit(1))
    fake_with_label = fake_df.withColumn("label", lit(0))
    
    # Combine datasets
    combined_df = real_with_label.union(fake_with_label)
    
    # Display class distribution
    print("Class distribution:")
    combined_df.groupBy("label").count().show()
    
    return combined_df

# %% [markdown]
# ### Memory Management Functions

# %%
def create_stratified_sample(df, sample_size_per_class=2000, seed=42):
    """
    Create a balanced sample with equal representation from each class.
    
    Args:
        df (DataFrame): DataFrame to sample from
        sample_size_per_class (int): Number of samples per class
        seed (int): Random seed for reproducibility
        
    Returns:
        DataFrame: DataFrame with balanced samples
    """
    # Get class counts
    class_counts = df.groupBy("label").count().collect()
    
    # Calculate sampling fractions
    fractions = {}
    for row in class_counts:
        label = row["label"]
        count = row["count"]
        fraction = min(1.0, sample_size_per_class / count)
        fractions[label] = fraction
    
    # Create stratified sample
    sampled_df = df.sampleBy("label", fractions, seed)
    
    print(f"Original class distribution:")
    df.groupBy("label").count().show()
    
    print(f"Sampled class distribution:")
    sampled_df.groupBy("label").count().show()
    
    return sampled_df

# %%
def prepare_working_dataset(df, use_full_dataset=True, sample_size_per_class=2000, seed=42):
    """
    Prepare working dataset based on available memory.
    
    Args:
        df (DataFrame): Input DataFrame
        use_full_dataset (bool): Whether to use the full dataset or a sample
        sample_size_per_class (int): Number of samples per class if sampling
        seed (int): Random seed for reproducibility
        
    Returns:
        DataFrame: Working DataFrame (full or sampled)
    """
    if use_full_dataset:
        working_df = df
        print("Using full dataset")
    else:
        # Create a balanced sample for development/testing
        working_df = create_stratified_sample(df, sample_size_per_class, seed)
        print("Using sampled dataset")

    # Cache the working dataset for faster processing
    working_df.cache()
    print(f"Working dataset size: {working_df.count()} records")
    
    return working_df

# %% [markdown]
# ### Text Preprocessing Functions

# %%
def preprocess_text(df, text_column="text", title_column="title"):
    """
    Preprocess text data for feature extraction.
    
    Args:
        df (DataFrame): Input DataFrame
        text_column (str): Name of the text column
        title_column (str): Name of the title column
        
    Returns:
        DataFrame: DataFrame with preprocessed text
    """
    # Combine title and text fields, and perform basic cleaning
    preprocessed_df = df.withColumn(
        "content", 
        concat_ws(" ", 
                  when(col(title_column).isNull(), "").otherwise(col(title_column)),
                  when(col(text_column).isNull(), "").otherwise(col(text_column))
        )
    )
    
    # Convert to lowercase and remove special characters
    preprocessed_df = preprocessed_df.withColumn("content", lower(col("content")))
    preprocessed_df = preprocessed_df.withColumn(
        "content", 
        regexp_replace(col("content"), "[^a-zA-Z0-9\\s]", " ")
    )
    
    # Remove extra whitespace
    preprocessed_df = preprocessed_df.withColumn(
        "content", 
        regexp_replace(col("content"), "\\s+", " ")
    )
    
    # Show sample of preprocessed content
    preprocessed_df.select("content", "label").show(5, truncate=50)
    
    return preprocessed_df

# %% [markdown]
# ### Data Splitting Functions

# %%
def split_train_test(df, train_ratio=0.7, seed=42):
    """
    Split data into training and testing sets.
    
    Args:
        df (DataFrame): Input DataFrame
        train_ratio (float): Ratio of training data (0-1)
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, test_df) Training and testing DataFrames
    """
    # Split data into training and testing sets
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    
    # Cache datasets for faster processing
    train_df.cache()
    test_df.cache()
    
    print(f"Training set size: {train_df.count()} records")
    print(f"Testing set size: {test_df.count()} records")
    
    # Check class distribution in training set
    print("Training set class distribution:")
    train_df.groupBy("label").count().show()
    
    # Check class distribution in testing set
    print("Testing set class distribution:")
    test_df.groupBy("label").count().show()
    
    return train_df, test_df

# %% [markdown]
# ### Feature Engineering Functions

# %%
def create_tfidf_pipeline(input_col="content", output_col="features", num_features=10000):
    """
    Create a TF-IDF feature extraction pipeline.
    
    Args:
        input_col (str): Input column name
        output_col (str): Output column name
        num_features (int): Number of features for HashingTF
        
    Returns:
        Pipeline: Spark ML Pipeline for TF-IDF feature extraction
    """
    # Define feature extraction pipeline
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=num_features)
    idf = IDF(inputCol="rawFeatures", outputCol=output_col)
    
    # Create feature extraction pipeline
    feature_pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
    
    return feature_pipeline

# %%
def extract_features(train_df, test_df, input_col="content", output_col="features", num_features=10000):
    """
    Extract TF-IDF features from training and testing data.
    
    Args:
        train_df (DataFrame): Training DataFrame
        test_df (DataFrame): Testing DataFrame
        input_col (str): Input column name
        output_col (str): Output column name
        num_features (int): Number of features for HashingTF
        
    Returns:
        tuple: (feature_model, train_features, test_features) - Fitted pipeline model and transformed DataFrames
    """
    # Create feature extraction pipeline
    feature_pipeline = create_tfidf_pipeline(input_col, output_col, num_features)
    
    # Fit the pipeline on the training data
    feature_model = feature_pipeline.fit(train_df)
    
    # Transform the training and testing data
    train_features = feature_model.transform(train_df)
    test_features = feature_model.transform(test_df)
    
    # Show sample of features
    train_features.select(input_col, "filtered", output_col, "label").show(2, truncate=50)
    
    return feature_model, train_features, test_features

# %% [markdown]
# ### Model Training and Evaluation Functions

# %%
def create_evaluators():
    """
    Create evaluators for model assessment.
    
    Returns:
        tuple: (accuracy_evaluator, f1_evaluator, auc_evaluator) - Evaluators for different metrics
    """
    # Define evaluators
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="f1"
    )
    
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label", 
        rawPredictionCol="rawPrediction", 
        metricName="areaUnderROC"
    )
    
    return accuracy_evaluator, f1_evaluator, auc_evaluator

# %%
def train_naive_bayes(train_features, test_features, num_folds=3):
    """
    Train a Naive Bayes classifier with cross-validation.
    
    Args:
        train_features (DataFrame): Training data with features
        test_features (DataFrame): Testing data with features
        num_folds (int): Number of folds for cross-validation
        
    Returns:
        tuple: (best_model, predictions, metrics) - Best model, predictions, and performance metrics
    """
    print("Training Naive Bayes model...")
    
    # Create Naive Bayes model
    nb = NaiveBayes(featuresCol="features", labelCol="label")
    
    # Define parameter grid for cross-validation
    paramGrid = ParamGridBuilder() \
        .addGrid(nb.smoothing, [0.1, 0.5, 1.0]) \
        .build()
    
    # Create evaluators
    accuracy_evaluator, f1_evaluator, auc_evaluator = create_evaluators()
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=nb,
        estimatorParamMaps=paramGrid,
        evaluator=f1_evaluator,
        numFolds=num_folds  # Use fewer folds for Community Edition (less resource intensive)
    )
    
    # Train model with cross-validation
    start_time = time.time()
    cv_model = cv.fit(train_features)
    training_time = time.time() - start_time
    print(f"Naive Bayes training time: {training_time:.2f} seconds")
    
    # Get best model
    best_model = cv_model.bestModel
    print(f"Best smoothing parameter: {best_model.getSmoothing()}")
    
    # Make predictions on test data
    predictions = best_model.transform(test_features)
    
    # Evaluate model
    accuracy = accuracy_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)
    
    print(f"Naive Bayes Accuracy: {accuracy:.4f}")
    print(f"Naive Bayes F1 Score: {f1:.4f}")
    print(f"Naive Bayes AUC: {auc:.4f}")
    
    # Collect metrics
    metrics = {
        "model": "Naive Bayes",
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
        "training_time": training_time
    }
    
    return best_model, predictions, metrics

# %%
def train_random_forest(train_features, test_features, num_folds=3):
    """
    Train a Random Forest classifier with cross-validation.
    
    Args:
        train_features (DataFrame): Training data with features
        test_features (DataFrame): Testing data with features
        num_folds (int): Number of folds for cross-validation
        
    Returns:
        tuple: (best_model, predictions, metrics) - Best model, predictions, and performance metrics
    """
    print("Training Random Forest model...")
    
    # Create Random Forest model
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")
    
    # Define parameter grid for cross-validation
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()
    
    # Create evaluators
    accuracy_evaluator, f1_evaluator, auc_evaluator = create_evaluators()
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=f1_evaluator,
        numFolds=num_folds  # Use fewer folds for Community Edition (less resource intensive)
    )
    
    # Train model with cross-validation
    start_time = time.time()
    cv_model = cv.fit(train_features)
    training_time = time.time() - start_time
    print(f"Random Forest training time: {training_time:.2f} seconds")
    
    # Get best model
    best_model = cv_model.bestModel
    print(f"Best numTrees: {best_model.getNumTrees}")
    print(f"Best maxDepth: {best_model.getMaxDepth()}")
    
    # Make predictions on test data
    predictions = best_model.transform(test_features)
    
    # Evaluate model
    accuracy = accuracy_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)
    
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print(f"Random Forest F1 Score: {f1:.4f}")
    print(f"Random Forest AUC: {auc:.4f}")
    
    # Collect metrics
    metrics = {
        "model": "Random Forest",
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
        "training_time": training_time
    }
    
    return best_model, predictions, metrics

# %%
def train_logistic_regression(train_features, test_features, num_folds=3):
    """
    Train a Logistic Regression classifier with cross-validation.
    
    Args:
        train_features (DataFrame): Training data with features
        test_features (DataFrame): Testing data with features
        num_folds (int): Number of folds for cross-validation
        
    Returns:
        tuple: (best_model, predictions, metrics) - Best model, predictions, and performance metrics
    """
    print("Training Logistic Regression model...")
    
    # Create Logistic Regression model
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    
    # Define parameter grid for cross-validation
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.3]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
    
    # Create evaluators
    accuracy_evaluator, f1_evaluator, auc_evaluator = create_evaluators()
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=f1_evaluator,
        numFolds=num_folds  # Use fewer folds for Community Edition (less resource intensive)
    )
    
    # Train model with cross-validation
    start_time = time.time()
    cv_model = cv.fit(train_features)
    training_time = time.time() - start_time
    print(f"Logistic Regression training time: {training_time:.2f} seconds")
    
    # Get best model
    best_model = cv_model.bestModel
    print(f"Best regParam: {best_model.getRegParam()}")
    print(f"Best elasticNetParam: {best_model.getElasticNetParam()}")
    
    # Make predictions on test data
    predictions = best_model.transform(test_features)
    
    # Evaluate model
    accuracy = accuracy_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)
    
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print(f"Logistic Regression F1 Score: {f1:.4f}")
    print(f"Logistic Regression AUC: {auc:.4f}")
    
    # Collect metrics
    metrics = {
        "model": "Logistic Regression",
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
        "training_time": training_time
    }
    
    return best_model, predictions, metrics

# %% [markdown]
# ### Feature Importance and Analysis Functions

# %%
def analyze_feature_importance(model, vocabulary=None, top_n=20):
    """
    Analyze feature importance from a trained model.
    
    Args:
        model: Trained model with feature importances
        vocabulary (list): List of feature names (optional)
        top_n (int): Number of top features to display
        
    Returns:
        DataFrame: DataFrame with feature importances
    """
    print("Analyzing feature importance...")
    
    # Check if model has feature importances
    if hasattr(model, 'featureImportances'):
        # Get feature importances from model
        feature_importances = model.featureImportances.toArray()
        
        # Get top N feature indices
        top_indices = np.argsort(-feature_importances)[:top_n]
        top_importances = feature_importances[top_indices]
        
        # Create DataFrame for visualization
        if vocabulary and len(vocabulary) >= max(top_indices):
            # If vocabulary is provided, use feature names
            feature_names = [vocabulary[i] for i in top_indices]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': top_importances
            })
        else:
            # Otherwise use feature indices
            importance_df = pd.DataFrame({
                'Feature Index': top_indices,
                'Importance': top_importances
            })
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['Importance'])
        
        if vocabulary and len(vocabulary) >= max(top_indices):
            plt.yticks(range(len(importance_df)), importance_df['Feature'])
        else:
            plt.yticks(range(len(importance_df)), [f"Feature {idx}" for idx in importance_df['Feature Index']])
            
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    else:
        print("Model does not have feature importances attribute.")
        return None

# %%
def compare_models(metrics_list):
    """
    Compare performance of multiple models.
    
    Args:
        metrics_list (list): List of model metrics dictionaries
        
    Returns:
        DataFrame: DataFrame with model comparison
    """
    # Create comparison DataFrame
    models = [metrics['model'] for metrics in metrics_list]
    accuracy = [metrics['accuracy'] for metrics in metrics_list]
    f1_score = [metrics['f1'] for metrics in metrics_list]
    auc_score = [metrics['auc'] for metrics in metrics_list]
    training_time = [metrics['training_time'] for metrics in metrics_list]
    
    comparison_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'F1 Score': f1_score,
        'AUC': auc_score,
        'Training Time (s)': training_time
    })
    
    # Display comparison
    print("Model Comparison:")
    print(comparison_df)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy, F1, and AUC
    plt.subplot(2, 1, 1)
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, accuracy, width, label='Accuracy')
    plt.bar(x, f1_score, width, label='F1 Score')
    plt.bar(x + width, auc_score, width, label='AUC')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot training time
    plt.subplot(2, 1, 2)
    plt.bar(x, training_time, color='green', alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Time Comparison')
    plt.xticks(x, models)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

# %% [markdown]
# ### Model Saving and Loading Functions

# %%
def save_model(model, path):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save
        path (str): Path where to save the model
    """
    print(f"Saving model to {path}...")
    
    try:
        model.write().overwrite().save(path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

# %%
def save_metrics(metrics_list, path):
    """
    Save model metrics to disk.
    
    Args:
        metrics_list (list): List of model metrics dictionaries
        path (str): Path where to save the metrics
    """
    print(f"Saving metrics to {path}...")
    
    try:
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_list)
        
        # Save to CSV
        metrics_df.to_csv(path, index=False)
        print("Metrics saved successfully.")
    except Exception as e:
        print(f"Error saving metrics: {e}")

# %% [markdown]
# ## Complete Modeling Pipeline

# %%
def train_and_evaluate_models(
    input_path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet",
    output_dir="dbfs:/FileStore/fake_news_detection/models",
    use_full_dataset=True,
    sample_size_per_class=2000,
    num_features=10000,
    num_folds=3
):
    """
    Complete pipeline for training and evaluating traditional machine learning models.
    
    Args:
        input_path (str): Path to preprocessed data
        output_dir (str): Directory to save models and results
        use_full_dataset (bool): Whether to use the full dataset or a sample
        sample_size_per_class (int): Number of samples per class if sampling
        num_features (int): Number of features for TF-IDF
        num_folds (int): Number of folds for cross-validation
        
    Returns:
        dict: Dictionary with references to trained models and results
    """
    print("Starting traditional machine learning modeling pipeline...")
    start_time = time.time()
    
    # Create output directories
    try:
        dbutils.fs.mkdirs(output_dir.replace("dbfs:", ""))
    except:
        print("Warning: Could not create directories. This is expected in local environments.")
    
    # 1. Load preprocessed data
    df = load_preprocessed_data(input_path)
    if df is None:
        print("Error: Could not load preprocessed data. Pipeline aborted.")
        return None
    
    # 2. Prepare working dataset
    working_df = prepare_working_dataset(df, use_full_dataset, sample_size_per_class)
    
    # 3. Preprocess text
    preprocessed_df = preprocess_text(working_df)
    
    # 4. Split data into training and testing sets
    train_df, test_df = split_train_test(preprocessed_df)
    
    # 5. Extract features
    feature_model, train_features, test_features = extract_features(
        train_df, test_df, input_col="content", output_col="features", num_features=num_features
    )
    
    # 6. Train and evaluate models
    metrics_list = []
    
    # 6.1. Naive Bayes
    nb_model, nb_predictions, nb_metrics = train_naive_bayes(train_features, test_features, num_folds)
    metrics_list.append(nb_metrics)
    
    # 6.2. Random Forest
    rf_model, rf_predictions, rf_metrics = train_random_forest(train_features, test_features, num_folds)
    metrics_list.append(rf_metrics)
    
    # 6.3. Logistic Regression
    lr_model, lr_predictions, lr_metrics = train_logistic_regression(train_features, test_features, num_folds)
    metrics_list.append(lr_metrics)
    
    # 7. Compare models
    comparison_df = compare_models(metrics_list)
    
    # 8. Analyze feature importance (for Random Forest)
    importance_df = analyze_feature_importance(rf_model)
    
    # 9. Save models
    save_model(nb_model, f"{output_dir}/naive_bayes_model")
    save_model(rf_model, f"{output_dir}/random_forest_model")
    save_model(lr_model, f"{output_dir}/logistic_regression_model")
    save_model(feature_model, f"{output_dir}/tfidf_feature_model")
    
    # 10. Save metrics
    save_metrics(metrics_list, f"{output_dir}/model_metrics.csv")
    
    print(f"\nTraditional machine learning modeling pipeline completed in {time.time() - start_time:.2f} seconds!")
    
    return {
        "naive_bayes_model": nb_model,
        "random_forest_model": rf_model,
        "logistic_regression_model": lr_model,
        "feature_model": feature_model,
        "metrics": metrics_list,
        "comparison": comparison_df,
        "importance": importance_df
    }

# %% [markdown]
# ## Step-by-Step Tutorial

# %% [markdown]
# ### 1. Load and Prepare Data

# %%
# Load preprocessed data
preprocessed_df = load_preprocessed_data()

# Prepare working dataset (use full dataset or sample)
if preprocessed_df:
    working_df = prepare_working_dataset(preprocessed_df, use_full_dataset=True)
    
    # Preprocess text
    preprocessed_df = preprocess_text(working_df)
    
    # Split data into training and testing sets
    train_df, test_df = split_train_test(preprocessed_df)

# %% [markdown]
# ### 2. Extract Features

# %%
# Extract TF-IDF features
if 'train_df' in locals() and 'test_df' in locals():
    feature_model, train_features, test_features = extract_features(
        train_df, test_df, input_col="content", output_col="features", num_features=10000
    )

# %% [markdown]
# ### 3. Train Naive Bayes Model

# %%
# Train Naive Bayes model
if 'train_features' in locals() and 'test_features' in locals():
    nb_model, nb_predictions, nb_metrics = train_naive_bayes(train_features, test_features)

# %% [markdown]
# ### 4. Train Random Forest Model

# %%
# Train Random Forest model
if 'train_features' in locals() and 'test_features' in locals():
    rf_model, rf_predictions, rf_metrics = train_random_forest(train_features, test_features)

# %% [markdown]
# ### 5. Train Logistic Regression Model

# %%
# Train Logistic Regression model
if 'train_features' in locals() and 'test_features' in locals():
    lr_model, lr_predictions, lr_metrics = train_logistic_regression(train_features, test_features)

# %% [markdown]
# ### 6. Compare Models

# %%
# Compare model performance
if 'nb_metrics' in locals() and 'rf_metrics' in locals() and 'lr_metrics' in locals():
    metrics_list = [nb_metrics, rf_metrics, lr_metrics]
    comparison_df = compare_models(metrics_list)

# %% [markdown]
# ### 7. Analyze Feature Importance

# %%
# Analyze feature importance from Random Forest model
if 'rf_model' in locals():
    importance_df = analyze_feature_importance(rf_model)

# %% [markdown]
# ### 8. Save Models and Results

# %%
# Save models and results
if 'nb_model' in locals() and 'rf_model' in locals() and 'lr_model' in locals():
    output_dir = "dbfs:/FileStore/fake_news_detection/models"
    
    # Save models
    save_model(nb_model, f"{output_dir}/naive_bayes_model")
    save_model(rf_model, f"{output_dir}/random_forest_model")
    save_model(lr_model, f"{output_dir}/logistic_regression_model")
    save_model(feature_model, f"{output_dir}/tfidf_feature_model")
    
    # Save metrics
    save_metrics(metrics_list, f"{output_dir}/model_metrics.csv")

# %% [markdown]
# ### 9. Complete Pipeline

# %%
# Run the complete modeling pipeline
results = train_and_evaluate_models(
    input_path="dbfs:/FileStore/fake_news_detection/preprocessed_data/preprocessed_news.parquet",
    output_dir="dbfs:/FileStore/fake_news_detection/models",
    use_full_dataset=True,
    sample_size_per_class=2000,
    num_features=10000,
    num_folds=3
)

# %% [markdown]
# ## Important Notes
# 
# 1. **Model Selection**: We implemented three traditional machine learning models (Naive Bayes, Random Forest, and Logistic Regression) that serve as strong baselines for fake news detection.
# 
# 2. **Feature Engineering**: We used TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features, which is a proven technique for text classification tasks.
# 
# 3. **Cross-Validation**: We used cross-validation to ensure robust model evaluation and hyperparameter tuning, which helps prevent overfitting.
# 
# 4. **Memory Management**: The code includes options for using the full dataset or a stratified sample, which is useful for environments with limited resources like Databricks Community Edition.
# 
# 5. **Vectorization**: All operations are vectorized using Spark's distributed processing capabilities, ensuring efficient computation even with large datasets.
# 
# 6. **Feature Importance**: We analyzed feature importance from the Random Forest model to understand what words are most predictive of fake news.
# 
# 7. **Model Comparison**: We compared the performance of all models using multiple metrics (accuracy, F1 score, AUC) to provide a comprehensive evaluation.
# 
# 8. **Databricks Integration**: The code is optimized for Databricks Community Edition with appropriate configurations for memory and processing.
