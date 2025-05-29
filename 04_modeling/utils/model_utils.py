"""
Model utility functions for fake news detection project.
This module contains functions for creating, training, and evaluating models.
"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import numpy as np
import matplotlib.pyplot as plt
import os

def create_random_forest_model(num_trees=10, max_depth=5, seed=42):
    """
    Create a Random Forest classifier model.
    
    Args:
        num_trees (int): Number of trees in the forest
        max_depth (int): Maximum depth of each tree
        seed (int): Random seed for reproducibility
        
    Returns:
        RandomForestClassifier: Configured model
    """
    rf = RandomForestClassifier(
        labelCol="label", 
        featuresCol="features", 
        numTrees=num_trees, 
        maxDepth=max_depth,
        seed=seed
    )
    return rf

def create_naive_bayes_model():
    """
    Create a Naive Bayes classifier model.
    
    Returns:
        NaiveBayes: Configured model
    """
    nb = NaiveBayes(
        labelCol="label", 
        featuresCol="features", 
        modelType="multinomial"
    )
    return nb

def create_logistic_regression_model(max_iter=100, reg_param=0.3, elastic_net_param=0.8):
    """
    Create a Logistic Regression classifier model.
    
    Args:
        max_iter (int): Maximum number of iterations
        reg_param (float): Regularization parameter
        elastic_net_param (float): ElasticNet mixing parameter
        
    Returns:
        LogisticRegression: Configured model
    """
    lr = LogisticRegression(
        labelCol="label", 
        featuresCol="features", 
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_net_param
    )
    return lr

def evaluate_model(model, test_data, log_dir=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        test_data: Test dataset
        log_dir (str): Directory to save evaluation logs
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    predictions = model.transform(test_data)
    
    # Binary classification evaluator
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="label", 
        rawPredictionCol="rawPrediction", 
        metricName="areaUnderROC"
    )
    
    # Multiclass classification evaluator for accuracy
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    
    # Calculate metrics
    auc = evaluator_auc.evaluate(predictions)
    accuracy = evaluator_acc.evaluate(predictions)
    
    # Calculate precision, recall, and F1 score
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="weightedPrecision"
    )
    
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="weightedRecall"
    )
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction", 
        metricName="f1"
    )
    
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    
    # Create metrics dictionary
    metrics = {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # Log metrics
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "evaluation_metrics.txt"), "a") as f:
            f.write(f"Model: {type(model).__name__}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write("-" * 50 + "\n")
    
    return metrics

def perform_cross_validation(pipeline, param_grid, train_data, num_folds=3, seed=42):
    """
    Perform cross-validation to find the best model parameters.
    
    Args:
        pipeline: ML Pipeline
        param_grid: Parameter grid for tuning
        train_data: Training dataset
        num_folds (int): Number of cross-validation folds
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (best_model, cv_metrics)
    """
    # Create evaluator
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", 
        rawPredictionCol="rawPrediction", 
        metricName="areaUnderROC"
    )
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=num_folds,
        seed=seed
    )
    
    # Fit cross-validator
    cv_model = cv.fit(train_data)
    
    # Get the best model
    best_model = cv_model.bestModel
    
    # Get metrics for all models
    cv_metrics = {
        "avg_metrics": cv_model.avgMetrics,
        "best_model_index": np.argmax(cv_model.avgMetrics),
        "param_maps": cv_model.getEstimatorParamMaps()
    }
    
    return best_model, cv_metrics

def plot_model_comparison(metrics_dict, save_path=None):
    """
    Plot comparison of model metrics.
    
    Args:
        metrics_dict (dict): Dictionary of model metrics
        save_path (str): Path to save the plot
    """
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.15
    index = np.arange(len(metrics))
    
    # Plot bars for each model
    for i, model in enumerate(models):
        values = [metrics_dict[model][metric] for metric in metrics]
        ax.bar(index + i * bar_width, values, bar_width, label=model)
    
    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(index + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

# Last modified: May 29, 2025
