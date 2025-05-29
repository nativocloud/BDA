"""
Cross-validation and data leakage prevention utility functions for fake news detection project.
This module contains functions for ensuring proper cross-validation and preventing data leakage.
"""

import os
import numpy as np
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt

def create_stratified_cv_folds(df, label_col="label", n_splits=5, seed=42):
    """
    Create stratified cross-validation folds for a Spark DataFrame.
    
    Args:
        df: Spark DataFrame
        label_col (str): Column containing labels
        n_splits (int): Number of folds
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of (train_df, val_df) tuples
    """
    # Convert to pandas for stratification
    pandas_df = df.toPandas()
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Create folds
    folds = []
    for train_idx, val_idx in skf.split(pandas_df, pandas_df[label_col]):
        train_pandas = pandas_df.iloc[train_idx]
        val_pandas = pandas_df.iloc[val_idx]
        
        # Convert back to Spark DataFrames
        train_spark = df.sparkSession.createDataFrame(train_pandas)
        val_spark = df.sparkSession.createDataFrame(val_pandas)
        
        folds.append((train_spark, val_spark))
    
    return folds

def perform_stratified_cv(spark, pipeline, param_grid, df, label_col="label", n_splits=5, seed=42):
    """
    Perform stratified cross-validation for a Spark ML pipeline.
    
    Args:
        spark: Spark session
        pipeline: ML Pipeline
        param_grid: Parameter grid for tuning
        df: Spark DataFrame
        label_col (str): Column containing labels
        n_splits (int): Number of folds
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (best_model, cv_metrics)
    """
    # Create folds
    folds = create_stratified_cv_folds(df, label_col, n_splits, seed)
    
    # Initialize metrics
    all_metrics = []
    
    # Perform cross-validation
    for i, (train_df, val_df) in enumerate(folds):
        print(f"Fold {i+1}/{n_splits}")
        
        # Train model
        model = pipeline.fit(train_df)
        
        # Evaluate model
        predictions = model.transform(val_df)
        
        # Calculate metrics
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol=label_col, 
            rawPredictionCol="rawPrediction", 
            metricName="areaUnderROC"
        )
        
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol=label_col, 
            predictionCol="prediction", 
            metricName="accuracy"
        )
        
        auc = evaluator_auc.evaluate(predictions)
        accuracy = evaluator_acc.evaluate(predictions)
        
        # Store metrics
        all_metrics.append({
            "fold": i+1,
            "auc": auc,
            "accuracy": accuracy
        })
        
        print(f"Fold {i+1} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    
    # Calculate average metrics
    avg_auc = np.mean([m["auc"] for m in all_metrics])
    avg_accuracy = np.mean([m["accuracy"] for m in all_metrics])
    
    print(f"Average - AUC: {avg_auc:.4f}, Accuracy: {avg_accuracy:.4f}")
    
    # Train final model on all data
    final_model = pipeline.fit(df)
    
    return final_model, all_metrics

def time_based_train_test_split(df, time_col, train_ratio=0.8, seed=42):
    """
    Split data based on time to prevent data leakage.
    
    Args:
        df: DataFrame
        time_col (str): Column containing time/date information
        train_ratio (float): Ratio of training data
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, test_df)
    """
    # Sort by time
    sorted_df = df.orderBy(time_col)
    
    # Calculate split point
    count = sorted_df.count()
    split_point = int(count * train_ratio)
    
    # Split data
    train_df = sorted_df.limit(split_point)
    test_df = sorted_df.subtract(train_df)
    
    return train_df, test_df

def feature_leakage_check(train_df, test_df, feature_cols):
    """
    Check for potential feature leakage between train and test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        feature_cols (list): List of feature column names
        
    Returns:
        dict: Dictionary of leakage metrics
    """
    leakage_metrics = {}
    
    for col in feature_cols:
        # Get unique values
        train_values = set(train_df.select(col).distinct().rdd.flatMap(lambda x: x).collect())
        test_values = set(test_df.select(col).distinct().rdd.flatMap(lambda x: x).collect())
        
        # Calculate overlap
        overlap = train_values.intersection(test_values)
        overlap_ratio = len(overlap) / len(test_values) if len(test_values) > 0 else 0
        
        leakage_metrics[col] = {
            "overlap_count": len(overlap),
            "overlap_ratio": overlap_ratio
        }
    
    return leakage_metrics

def plot_cv_results(cv_metrics, metric_name="auc", save_path=None):
    """
    Plot cross-validation results.
    
    Args:
        cv_metrics (list): List of metrics dictionaries
        metric_name (str): Name of the metric to plot
        save_path (str): Path to save the plot
    """
    # Extract metrics
    folds = [m["fold"] for m in cv_metrics]
    metrics = [m[metric_name] for m in cv_metrics]
    avg_metric = np.mean(metrics)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(folds, metrics, color='skyblue')
    plt.axhline(y=avg_metric, color='r', linestyle='-', label=f'Average: {avg_metric:.4f}')
    
    plt.title(f'Cross-Validation Results - {metric_name.upper()}')
    plt.xlabel('Fold')
    plt.ylabel(metric_name.upper())
    plt.xticks(folds)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def create_nested_cv(df, outer_splits=5, inner_splits=3, seed=42):
    """
    Create nested cross-validation folds to prevent data leakage during hyperparameter tuning.
    
    Args:
        df: DataFrame
        outer_splits (int): Number of outer folds for model evaluation
        inner_splits (int): Number of inner folds for hyperparameter tuning
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of (train_idx, test_idx, inner_cv) tuples
    """
    # Convert to pandas for stratification
    pandas_df = df.toPandas()
    X = pandas_df.drop('label', axis=1)
    y = pandas_df['label']
    
    # Create outer folds
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    
    # Create nested CV structure
    nested_cv = []
    
    for train_idx, test_idx in outer_cv.split(X, y):
        # Create inner CV
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_cv_splits = list(inner_cv.split(X.iloc[train_idx], y.iloc[train_idx]))
        
        nested_cv.append((train_idx, test_idx, inner_cv_splits))
    
    return nested_cv

def perform_nested_cv(spark, pipeline_factory, param_grid_factory, df, nested_cv):
    """
    Perform nested cross-validation to prevent data leakage during hyperparameter tuning.
    
    Args:
        spark: Spark session
        pipeline_factory: Function that creates a pipeline
        param_grid_factory: Function that creates a parameter grid
        df: DataFrame
        nested_cv: Nested CV structure from create_nested_cv
        
    Returns:
        tuple: (best_models, outer_metrics, inner_metrics)
    """
    pandas_df = df.toPandas()
    
    best_models = []
    outer_metrics = []
    inner_metrics = []
    
    for fold_idx, (train_idx, test_idx, inner_cv_splits) in enumerate(nested_cv):
        print(f"Outer Fold {fold_idx+1}/{len(nested_cv)}")
        
        # Get outer train/test data
        outer_train = pandas_df.iloc[train_idx]
        outer_test = pandas_df.iloc[test_idx]
        
        # Convert to Spark DataFrames
        spark_outer_train = spark.createDataFrame(outer_train)
        spark_outer_test = spark.createDataFrame(outer_test)
        
        # Inner CV for hyperparameter tuning
        inner_fold_metrics = []
        
        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv_splits):
            print(f"  Inner Fold {inner_fold_idx+1}/{len(inner_cv_splits)}")
            
            # Get inner train/val data
            inner_train = outer_train.iloc[inner_train_idx]
            inner_val = outer_train.iloc[inner_val_idx]
            
            # Convert to Spark DataFrames
            spark_inner_train = spark.createDataFrame(inner_train)
            spark_inner_val = spark.createDataFrame(inner_val)
            
            # Create pipeline and param grid
            pipeline = pipeline_factory()
            param_grid = param_grid_factory()
            
            # Create cross-validator
            evaluator = BinaryClassificationEvaluator(
                labelCol="label", 
                rawPredictionCol="rawPrediction", 
                metricName="areaUnderROC"
            )
            
            cv = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=3  # This is just for hyperparameter selection within the inner fold
            )
            
            # Fit cross-validator
            cv_model = cv.fit(spark_inner_train)
            
            # Evaluate on inner validation set
            predictions = cv_model.transform(spark_inner_val)
            auc = evaluator.evaluate(predictions)
            
            inner_fold_metrics.append({
                "outer_fold": fold_idx+1,
                "inner_fold": inner_fold_idx+1,
                "auc": auc
            })
            
            print(f"    Inner Fold {inner_fold_idx+1} - AUC: {auc:.4f}")
        
        # Train final model for this outer fold using best hyperparameters
        pipeline = pipeline_factory()
        best_model = pipeline.fit(spark_outer_train)
        
        # Evaluate on outer test set
        predictions = best_model.transform(spark_outer_test)
        evaluator = BinaryClassificationEvaluator(
            labelCol="label", 
            rawPredictionCol="rawPrediction", 
            metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(predictions)
        
        outer_metrics.append({
            "fold": fold_idx+1,
            "auc": auc
        })
        
        best_models.append(best_model)
        inner_metrics.append(inner_fold_metrics)
        
        print(f"Outer Fold {fold_idx+1} - AUC: {auc:.4f}")
    
    return best_models, outer_metrics, inner_metrics

# Last modified: May 29, 2025
