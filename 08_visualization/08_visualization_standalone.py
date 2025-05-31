# %% [markdown]
# # Fake News Detection: Visualization Analysis
# 
# This notebook contains all the necessary code for visualization analysis in the fake news detection project. The code is organized into independent functions, without dependencies on external modules or classes, to facilitate execution in Databricks Community Edition.

# %% [markdown]
# ## Setup and Imports

# %%
# Import necessary libraries
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, to_json, struct

# %%
# Initialize Spark session optimized for Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsDetection_VisualizationAnalysis") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

# Display Spark configuration
print(f"Spark version: {spark.version}")
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")

# %% [markdown]
# ## Reusable Functions

# %% [markdown]
# ### Configuration Functions

# %%
def create_visualization_config(theme='darkgrid', fig_size=(12, 8), dpi=300, 
                               grafana_export=True, grafana_export_path=None):
    """
    Create a configuration dictionary for visualizations.
    
    Args:
        theme (str): Visual theme for plots
        fig_size (tuple): Default figure size
        dpi (int): DPI for saved figures
        grafana_export (bool): Whether to export data for Grafana
        grafana_export_path (str): Path for Grafana data exports
        
    Returns:
        dict: Configuration dictionary
    """
    config = {
        'theme': theme,
        'fig_size': fig_size,
        'dpi': dpi,
        'grafana_export': grafana_export,
        'grafana_export_path': grafana_export_path
    }
    
    return config

# %%
def setup_visualization_environment(config=None, output_dir=None):
    """
    Set up the visualization environment with the specified configuration.
    
    Args:
        config (dict): Configuration dictionary
        output_dir (str): Directory to save visualization outputs
        
    Returns:
        tuple: (config, output_dir) - Updated configuration and output directory
    """
    # Default configuration
    if config is None:
        config = create_visualization_config()
    
    # Default output directory
    if output_dir is None:
        output_dir = '/tmp/fake_news_detection/logs/'
    
    # Set Grafana export path if not specified
    if config['grafana_export'] and config['grafana_export_path'] is None:
        config['grafana_export_path'] = os.path.join(output_dir, 'grafana')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    if config['grafana_export']:
        os.makedirs(config['grafana_export_path'], exist_ok=True)
    
    # Set visualization style
    sns.set_theme(style=config['theme'])
    plt.rcParams['figure.figsize'] = config['fig_size']
    
    return config, output_dir

# %% [markdown]
# ### Data Loading Functions

# %%
def load_model_metrics(metrics_path):
    """
    Load model metrics from a JSON file.
    
    Args:
        metrics_path (str): Path to the metrics JSON file
        
    Returns:
        dict: Dictionary with model metrics
    """
    print(f"Loading model metrics from {metrics_path}...")
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print(f"Successfully loaded metrics for {len(metrics)} models")
        return metrics
    
    except Exception as e:
        print(f"Error loading model metrics: {e}")
        return {}

# %%
def load_confusion_matrices(cm_path):
    """
    Load confusion matrices from a JSON file.
    
    Args:
        cm_path (str): Path to the confusion matrices JSON file
        
    Returns:
        dict: Dictionary with confusion matrices
    """
    print(f"Loading confusion matrices from {cm_path}...")
    
    try:
        with open(cm_path, 'r') as f:
            cm_data = json.load(f)
        
        # Convert lists to numpy arrays
        confusion_matrices = {}
        for model, cm in cm_data.items():
            confusion_matrices[model] = np.array(cm)
        
        print(f"Successfully loaded confusion matrices for {len(confusion_matrices)} models")
        return confusion_matrices
    
    except Exception as e:
        print(f"Error loading confusion matrices: {e}")
        return {}

# %%
def load_time_series_metrics(time_series_path):
    """
    Load time series metrics from a CSV file.
    
    Args:
        time_series_path (str): Path to the time series CSV file
        
    Returns:
        DataFrame: Pandas DataFrame with time series metrics
    """
    print(f"Loading time series metrics from {time_series_path}...")
    
    try:
        df = pd.read_csv(time_series_path)
        
        # Convert timestamp column to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Successfully loaded time series metrics with {len(df)} records")
        return df
    
    except Exception as e:
        print(f"Error loading time series metrics: {e}")
        return pd.DataFrame()

# %% [markdown]
# ### Basic Visualization Functions

# %%
def plot_confusion_matrix(cm, classes=None, title='Confusion Matrix', 
                         normalize=False, config=None, save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix to plot
        classes (list): List of class names
        title (str): Title for the plot
        normalize (bool): Whether to normalize the confusion matrix
        config (dict): Visualization configuration
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Use default config if not provided
    if config is None:
        config = create_visualization_config()
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=config['fig_size'])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Set tick marks and labels
    if classes is None:
        classes = ['Fake', 'Real']
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()

# %%
def plot_model_comparison(models_data, metric='accuracy', title='Model Comparison',
                         config=None, save_path=None):
    """
    Plot a comparison of multiple models based on a specified metric.
    
    Args:
        models_data (dict): Dictionary with model names as keys and metrics as values
        metric (str): Metric to compare
        title (str): Title for the plot
        config (dict): Visualization configuration
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Use default config if not provided
    if config is None:
        config = create_visualization_config()
    
    # Extract data
    models = list(models_data.keys())
    values = [data.get(metric, 0) for data in models_data.values()]
    
    # Create figure
    plt.figure(figsize=config['fig_size'])
    bars = plt.bar(models, values, color=sns.color_palette('viridis', len(models)))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel(metric.capitalize())
    plt.ylim(0, max(values) + 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    # Export data for Grafana if enabled
    if config['grafana_export'] and config['grafana_export_path']:
        export_data = pd.DataFrame({
            'model': models,
            metric: values
        })
        export_path = os.path.join(config['grafana_export_path'], f'model_comparison_{metric}.csv')
        export_data.to_csv(export_path, index=False)
        print(f"Model comparison data exported to {export_path}")
    
    return plt.gcf()

# %%
def plot_metrics_over_time(metrics_data, metrics=None, title='Metrics Over Time',
                          config=None, save_path=None):
    """
    Plot metrics over time for streaming evaluation.
    
    Args:
        metrics_data (DataFrame): DataFrame with 'timestamp' column and metric columns
        metrics (list): List of metric names to plot
        title (str): Title for the plot
        config (dict): Visualization configuration
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Use default config if not provided
    if config is None:
        config = create_visualization_config()
    
    # Check for timestamp column
    if 'timestamp' not in metrics_data.columns:
        raise ValueError("metrics_data must contain a 'timestamp' column")
    
    # Use all numeric columns if metrics not specified
    if metrics is None:
        metrics = [col for col in metrics_data.columns 
                  if col != 'timestamp' and pd.api.types.is_numeric_dtype(metrics_data[col])]
    
    # Create figure
    plt.figure(figsize=config['fig_size'])
    
    for metric in metrics:
        if metric in metrics_data.columns:
            plt.plot(metrics_data['timestamp'], metrics_data[metric], marker='o', label=metric)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')
        print(f"Metrics over time plot saved to {save_path}")
    
    # Export data for Grafana if enabled
    if config['grafana_export'] and config['grafana_export_path']:
        export_path = os.path.join(config['grafana_export_path'], 'metrics_over_time.csv')
        metrics_data.to_csv(export_path, index=False)
        print(f"Metrics over time data exported to {export_path}")
    
    return plt.gcf()

# %% [markdown]
# ### Advanced Visualization Functions

# %%
def plot_feature_importance(feature_names, importance_values, title='Feature Importance',
                           top_n=20, config=None, save_path=None):
    """
    Plot feature importance from a trained model.
    
    Args:
        feature_names (list): List of feature names
        importance_values (list): List of importance values
        title (str): Title for the plot
        top_n (int): Number of top features to display
        config (dict): Visualization configuration
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Use default config if not provided
    if config is None:
        config = create_visualization_config()
    
    # Create DataFrame for easier sorting
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    })
    
    # Sort by importance and get top N
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    # Create figure
    plt.figure(figsize=config['fig_size'])
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(title)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    # Export data for Grafana if enabled
    if config['grafana_export'] and config['grafana_export_path']:
        export_path = os.path.join(config['grafana_export_path'], 'feature_importance.csv')
        feature_importance.to_csv(export_path, index=False)
        print(f"Feature importance data exported to {export_path}")
    
    return plt.gcf()

# %%
def plot_roc_curve(fpr_dict, tpr_dict, auc_dict, title='ROC Curve',
                  config=None, save_path=None):
    """
    Plot ROC curves for multiple models.
    
    Args:
        fpr_dict (dict): Dictionary with model names as keys and false positive rates as values
        tpr_dict (dict): Dictionary with model names as keys and true positive rates as values
        auc_dict (dict): Dictionary with model names as keys and AUC values as values
        title (str): Title for the plot
        config (dict): Visualization configuration
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Use default config if not provided
    if config is None:
        config = create_visualization_config()
    
    # Create figure
    plt.figure(figsize=config['fig_size'])
    
    # Plot diagonal line for random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # Plot ROC curve for each model
    for model_name in fpr_dict.keys():
        if model_name in tpr_dict and model_name in auc_dict:
            plt.plot(fpr_dict[model_name], tpr_dict[model_name],
                    label=f'{model_name} (AUC = {auc_dict[model_name]:.3f})')
    
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')
        print(f"ROC curve plot saved to {save_path}")
    
    return plt.gcf()

# %%
def plot_precision_recall_curve(precision_dict, recall_dict, ap_dict, title='Precision-Recall Curve',
                               config=None, save_path=None):
    """
    Plot precision-recall curves for multiple models.
    
    Args:
        precision_dict (dict): Dictionary with model names as keys and precision values as values
        recall_dict (dict): Dictionary with model names as keys and recall values as values
        ap_dict (dict): Dictionary with model names as keys and average precision values as values
        title (str): Title for the plot
        config (dict): Visualization configuration
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Use default config if not provided
    if config is None:
        config = create_visualization_config()
    
    # Create figure
    plt.figure(figsize=config['fig_size'])
    
    # Plot precision-recall curve for each model
    for model_name in precision_dict.keys():
        if model_name in recall_dict and model_name in ap_dict:
            plt.plot(recall_dict[model_name], precision_dict[model_name],
                    label=f'{model_name} (AP = {ap_dict[model_name]:.3f})')
    
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')
        print(f"Precision-recall curve plot saved to {save_path}")
    
    return plt.gcf()

# %% [markdown]
# ### Interactive Visualization Functions

# %%
def create_interactive_confusion_matrix(cm, classes=None, title='Confusion Matrix',
                                       normalize=False, save_path=None):
    """
    Create an interactive confusion matrix with Plotly.
    
    Args:
        cm (numpy.ndarray): Confusion matrix to plot
        classes (list): List of class names
        title (str): Title for the plot
        normalize (bool): Whether to normalize the confusion matrix
        save_path (str): Path to save the HTML file
        
    Returns:
        plotly.graph_objects.Figure: The figure object
    """
    # Normalize if requested
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        z = cm_norm
        text = [[f'{val:.2f}' for val in row] for row in cm_norm]
    else:
        z = cm
        text = [[str(int(val)) for val in row] for row in cm]
    
    # Set default classes if not provided
    if classes is None:
        classes = ['Fake', 'Real']
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=classes,
        y=classes,
        colorscale='Blues',
        text=text,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label'),
        width=600,
        height=600
    )
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive confusion matrix saved to {save_path}")
    
    return fig

# %%
def create_interactive_model_comparison(models_data, metrics=None, title='Model Comparison',
                                       save_path=None):
    """
    Create an interactive model comparison with Plotly.
    
    Args:
        models_data (dict): Dictionary with model names as keys and metrics as values
        metrics (list): List of metrics to compare
        title (str): Title for the plot
        save_path (str): Path to save the HTML file
        
    Returns:
        plotly.graph_objects.Figure: The figure object
    """
    # Extract model names
    models = list(models_data.keys())
    
    # Use all metrics if not specified
    if metrics is None and len(models_data) > 0:
        # Get metrics from first model
        first_model = next(iter(models_data.values()))
        metrics = list(first_model.keys())
    
    # Create figure
    fig = go.Figure()
    
    # Add bar for each metric
    for metric in metrics:
        values = [data.get(metric, 0) for data in models_data.values()]
        fig.add_trace(go.Bar(
            x=models,
            y=values,
            name=metric,
            text=[f'{val:.3f}' for val in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title='Model'),
        yaxis=dict(title='Value', range=[0, 1]),
        barmode='group',
        width=800,
        height=500
    )
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive model comparison saved to {save_path}")
    
    return fig

# %%
def create_interactive_time_series(metrics_data, metrics=None, title='Metrics Over Time',
                                  save_path=None):
    """
    Create an interactive time series plot with Plotly.
    
    Args:
        metrics_data (DataFrame): DataFrame with 'timestamp' column and metric columns
        metrics (list): List of metrics to plot
        title (str): Title for the plot
        save_path (str): Path to save the HTML file
        
    Returns:
        plotly.graph_objects.Figure: The figure object
    """
    # Check for timestamp column
    if 'timestamp' not in metrics_data.columns:
        raise ValueError("metrics_data must contain a 'timestamp' column")
    
    # Use all numeric columns if metrics not specified
    if metrics is None:
        metrics = [col for col in metrics_data.columns 
                  if col != 'timestamp' and pd.api.types.is_numeric_dtype(metrics_data[col])]
    
    # Create figure
    fig = go.Figure()
    
    # Add line for each metric
    for metric in metrics:
        if metric in metrics_data.columns:
            fig.add_trace(go.Scatter(
                x=metrics_data['timestamp'],
                y=metrics_data[metric],
                mode='lines+markers',
                name=metric
            ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(title='Time'),
        yaxis=dict(title='Value'),
        width=800,
        height=500
    )
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive time series plot saved to {save_path}")
    
    return fig

# %%
def create_interactive_dashboard(models_data, confusion_matrices, metrics_over_time=None,
                                title='Fake News Detection Dashboard', save_path=None):
    """
    Create an interactive dashboard with Plotly.
    
    Args:
        models_data (dict): Dictionary with model names as keys and metrics as values
        confusion_matrices (dict): Dictionary with model names as keys and confusion matrices as values
        metrics_over_time (DataFrame): DataFrame with metrics over time
        title (str): Title for the dashboard
        save_path (str): Path to save the HTML file
        
    Returns:
        plotly.graph_objects.Figure: The figure object
    """
    # Create subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Accuracy Comparison', 'Model F1 Score Comparison',
                       'Confusion Matrix', 'Metrics Over Time'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
              [{'type': 'heatmap'}, {'type': 'scatter'}]]
    )
    
    # Add model comparison bars for accuracy
    models = list(models_data.keys())
    accuracies = [data.get('accuracy', 0) for data in models_data.values()]
    
    fig.add_trace(
        go.Bar(
            x=models, 
            y=accuracies, 
            name='Accuracy', 
            text=[f'{acc:.3f}' for acc in accuracies],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Add model comparison bars for F1 score
    f1_scores = [data.get('f1', 0) for data in models_data.values()]
    
    fig.add_trace(
        go.Bar(
            x=models, 
            y=f1_scores, 
            name='F1 Score',
            text=[f'{f1:.3f}' for f1 in f1_scores],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Add confusion matrix for the best model
    best_model = models[np.argmax(accuracies)] if accuracies else None
    if best_model and best_model in confusion_matrices:
        cm = confusion_matrices[best_model]
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Fake', 'Real'],
                y=['Fake', 'Real'],
                colorscale='Blues',
                showscale=True,
                text=[[str(int(val)) for val in row] for row in cm],
                texttemplate="%{text}",
                name='Confusion Matrix'
            ),
            row=2, col=1
        )
    
    # Add metrics over time if provided
    if metrics_over_time is not None and 'timestamp' in metrics_over_time.columns:
        metrics = [col for col in metrics_over_time.columns 
                  if col != 'timestamp' and pd.api.types.is_numeric_dtype(metrics_over_time[col])]
        
        for metric in metrics:
            fig.add_trace(
                go.Scatter(
                    x=metrics_over_time['timestamp'],
                    y=metrics_over_time[metric],
                    mode='lines+markers',
                    name=metric
                ),
                row=2, col=2
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes
    fig.update_yaxes(title_text='Accuracy', range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text='F1 Score', range=[0, 1], row=1, col=2)
    fig.update_xaxes(title_text='Predicted', row=2, col=1)
    fig.update_yaxes(title_text='Actual', row=2, col=1)
    fig.update_xaxes(title_text='Time', row=2, col=2)
    fig.update_yaxes(title_text='Value', row=2, col=2)
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    
    return fig

# %% [markdown]
# ### Grafana Integration Functions

# %%
def export_metrics_for_grafana(metrics_data, output_path, filename='metrics.json'):
    """
    Export metrics in a format suitable for Grafana.
    
    Args:
        metrics_data (dict or DataFrame): Metrics data to export
        output_path (str): Directory to save the exported file
        filename (str): Name of the output file
        
    Returns:
        str: Path to the exported file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Full path to the output file
    export_path = os.path.join(output_path, filename)
    
    # Convert DataFrame to dict if needed
    if isinstance(metrics_data, pd.DataFrame):
        # Convert to records format
        metrics_dict = metrics_data.to_dict(orient='records')
    else:
        metrics_dict = metrics_data
    
    # Export to JSON
    with open(export_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Metrics exported to {export_path}")
    return export_path

# %%
def setup_grafana_datasource(metrics_path, dashboard_title='Fake News Detection'):
    """
    Generate a Grafana datasource configuration.
    
    Args:
        metrics_path (str): Path to the metrics file
        dashboard_title (str): Title for the Grafana dashboard
        
    Returns:
        dict: Grafana datasource configuration
    """
    # Create datasource configuration
    datasource = {
        "name": "FakeNewsMetrics",
        "type": "json",
        "url": metrics_path,
        "access": "direct",
        "jsonData": {
            "timeField": "timestamp"
        }
    }
    
    # Create dashboard configuration
    dashboard = {
        "title": dashboard_title,
        "uid": "fake-news-detection",
        "panels": [
            {
                "title": "Model Accuracy",
                "type": "gauge",
                "datasource": "FakeNewsMetrics",
                "targets": [
                    {
                        "target": "accuracy"
                    }
                ]
            },
            {
                "title": "Fake vs Real News",
                "type": "piechart",
                "datasource": "FakeNewsMetrics",
                "targets": [
                    {
                        "target": "fake_count"
                    },
                    {
                        "target": "real_count"
                    }
                ]
            }
        ]
    }
    
    return {
        "datasource": datasource,
        "dashboard": dashboard
    }

# %% [markdown]
# ## Complete Visualization Pipeline

# %%
def run_visualization_pipeline(
    model_metrics_path="/tmp/fake_news_detection/results/model_metrics.json",
    confusion_matrices_path="/tmp/fake_news_detection/results/confusion_matrices.json",
    time_series_path="/tmp/fake_news_detection/results/metrics_over_time.csv",
    feature_importance_path="/tmp/fake_news_detection/results/feature_importance.json",
    output_dir="/tmp/fake_news_detection/visualizations",
    grafana_export=True
):
    """
    Run the complete visualization pipeline for fake news detection.
    
    Args:
        model_metrics_path (str): Path to the model metrics JSON file
        confusion_matrices_path (str): Path to the confusion matrices JSON file
        time_series_path (str): Path to the time series metrics CSV file
        feature_importance_path (str): Path to the feature importance JSON file
        output_dir (str): Directory to save visualization outputs
        grafana_export (bool): Whether to export data for Grafana
        
    Returns:
        dict: Dictionary with references to visualization results
    """
    print("Starting visualization pipeline...")
    
    # 1. Set up configuration
    config = create_visualization_config(
        grafana_export=grafana_export,
        grafana_export_path=os.path.join(output_dir, 'grafana')
    )
    
    config, output_dir = setup_visualization_environment(config, output_dir)
    
    # Create subdirectories
    static_dir = os.path.join(output_dir, 'static')
    interactive_dir = os.path.join(output_dir, 'interactive')
    
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(interactive_dir, exist_ok=True)
    
    # 2. Load data
    model_metrics = {}
    confusion_matrices = {}
    time_series_metrics = pd.DataFrame()
    feature_importance = {}
    
    # Try to load model metrics
    try:
        model_metrics = load_model_metrics(model_metrics_path)
    except:
        print("Warning: Could not load model metrics. Using sample data.")
        # Create sample data
        model_metrics = {
            "RandomForest": {"accuracy": 0.92, "precision": 0.91, "recall": 0.90, "f1": 0.90},
            "LogisticRegression": {"accuracy": 0.88, "precision": 0.87, "recall": 0.86, "f1": 0.86},
            "LSTM": {"accuracy": 0.94, "precision": 0.93, "recall": 0.92, "f1": 0.92}
        }
    
    # Try to load confusion matrices
    try:
        confusion_matrices = load_confusion_matrices(confusion_matrices_path)
    except:
        print("Warning: Could not load confusion matrices. Using sample data.")
        # Create sample data
        confusion_matrices = {
            "RandomForest": np.array([[450, 50], [40, 460]]),
            "LogisticRegression": np.array([[430, 70], [60, 440]]),
            "LSTM": np.array([[460, 40], [30, 470]])
        }
    
    # Try to load time series metrics
    try:
        time_series_metrics = load_time_series_metrics(time_series_path)
    except:
        print("Warning: Could not load time series metrics. Using sample data.")
        # Create sample data
        dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
        time_series_metrics = pd.DataFrame({
            'timestamp': dates,
            'accuracy': np.linspace(0.85, 0.95, 10),
            'precision': np.linspace(0.84, 0.94, 10),
            'recall': np.linspace(0.83, 0.93, 10),
            'f1': np.linspace(0.83, 0.93, 10)
        })
    
    # Try to load feature importance
    try:
        with open(feature_importance_path, 'r') as f:
            feature_importance = json.load(f)
    except:
        print("Warning: Could not load feature importance. Using sample data.")
        # Create sample data
        feature_importance = {
            "features": [f"feature_{i}" for i in range(20)],
            "importance": np.random.rand(20).tolist()
        }
    
    # 3. Create static visualizations
    
    # Model comparison plots
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_model_comparison(
            model_metrics, 
            metric=metric,
            title=f'Model Comparison - {metric.capitalize()}',
            config=config,
            save_path=os.path.join(static_dir, f'model_comparison_{metric}.png')
        )
    
    # Confusion matrix for each model
    for model, cm in confusion_matrices.items():
        plot_confusion_matrix(
            cm,
            classes=['Fake', 'Real'],
            title=f'Confusion Matrix - {model}',
            config=config,
            save_path=os.path.join(static_dir, f'confusion_matrix_{model}.png')
        )
        
        # Also create normalized version
        plot_confusion_matrix(
            cm,
            classes=['Fake', 'Real'],
            title=f'Normalized Confusion Matrix - {model}',
            normalize=True,
            config=config,
            save_path=os.path.join(static_dir, f'confusion_matrix_{model}_normalized.png')
        )
    
    # Metrics over time
    if not time_series_metrics.empty and 'timestamp' in time_series_metrics.columns:
        plot_metrics_over_time(
            time_series_metrics,
            title='Performance Metrics Over Time',
            config=config,
            save_path=os.path.join(static_dir, 'metrics_over_time.png')
        )
    
    # Feature importance
    if 'features' in feature_importance and 'importance' in feature_importance:
        plot_feature_importance(
            feature_importance['features'],
            feature_importance['importance'],
            title='Feature Importance',
            config=config,
            save_path=os.path.join(static_dir, 'feature_importance.png')
        )
    
    # 4. Create interactive visualizations
    
    # Interactive model comparison
    create_interactive_model_comparison(
        model_metrics,
        metrics=['accuracy', 'precision', 'recall', 'f1'],
        title='Model Performance Comparison',
        save_path=os.path.join(interactive_dir, 'model_comparison.html')
    )
    
    # Interactive confusion matrix for best model
    best_model = max(model_metrics.items(), key=lambda x: x[1].get('accuracy', 0))[0]
    if best_model in confusion_matrices:
        create_interactive_confusion_matrix(
            confusion_matrices[best_model],
            classes=['Fake', 'Real'],
            title=f'Confusion Matrix - {best_model}',
            save_path=os.path.join(interactive_dir, 'confusion_matrix.html')
        )
    
    # Interactive time series
    if not time_series_metrics.empty and 'timestamp' in time_series_metrics.columns:
        create_interactive_time_series(
            time_series_metrics,
            title='Performance Metrics Over Time',
            save_path=os.path.join(interactive_dir, 'metrics_over_time.html')
        )
    
    # Interactive dashboard
    create_interactive_dashboard(
        model_metrics,
        confusion_matrices,
        time_series_metrics,
        title='Fake News Detection Dashboard',
        save_path=os.path.join(interactive_dir, 'dashboard.html')
    )
    
    # 5. Export data for Grafana if enabled
    if grafana_export:
        # Export model metrics
        export_metrics_for_grafana(
            model_metrics,
            config['grafana_export_path'],
            'model_metrics.json'
        )
        
        # Export time series metrics
        if not time_series_metrics.empty:
            time_series_metrics.to_csv(
                os.path.join(config['grafana_export_path'], 'time_series_metrics.csv'),
                index=False
            )
        
        # Generate Grafana configuration
        grafana_config = setup_grafana_datasource(
            os.path.join(config['grafana_export_path'], 'model_metrics.json'),
            'Fake News Detection Dashboard'
        )
        
        with open(os.path.join(config['grafana_export_path'], 'grafana_config.json'), 'w') as f:
            json.dump(grafana_config, f, indent=2)
    
    print(f"Visualization pipeline completed!")
    print(f"Static visualizations saved to: {static_dir}")
    print(f"Interactive visualizations saved to: {interactive_dir}")
    if grafana_export:
        print(f"Grafana data exported to: {config['grafana_export_path']}")
    
    return {
        "static_dir": static_dir,
        "interactive_dir": interactive_dir,
        "grafana_dir": config['grafana_export_path'] if grafana_export else None,
        "model_metrics": model_metrics,
        "confusion_matrices": confusion_matrices,
        "time_series_metrics": time_series_metrics
    }

# %% [markdown]
# ## Step-by-Step Tutorial

# %% [markdown]
# ### 1. Set Up Visualization Environment

# %%
# Create configuration
config = create_visualization_config(
    theme='darkgrid',
    fig_size=(10, 6),
    dpi=300,
    grafana_export=True,
    grafana_export_path='/tmp/fake_news_detection/grafana'
)

# Set up environment
config, output_dir = setup_visualization_environment(
    config,
    output_dir='/tmp/fake_news_detection/visualizations'
)

print(f"Visualization environment set up with output directory: {output_dir}")
print(f"Configuration: {config}")

# %% [markdown]
# ### 2. Create Sample Data for Demonstration

# %%
# Create sample model metrics
model_metrics = {
    "RandomForest": {"accuracy": 0.92, "precision": 0.91, "recall": 0.90, "f1": 0.90},
    "LogisticRegression": {"accuracy": 0.88, "precision": 0.87, "recall": 0.86, "f1": 0.86},
    "LSTM": {"accuracy": 0.94, "precision": 0.93, "recall": 0.92, "f1": 0.92}
}

# Create sample confusion matrices
confusion_matrices = {
    "RandomForest": np.array([[450, 50], [40, 460]]),
    "LogisticRegression": np.array([[430, 70], [60, 440]]),
    "LSTM": np.array([[460, 40], [30, 470]])
}

# Create sample time series data
dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
time_series_metrics = pd.DataFrame({
    'timestamp': dates,
    'accuracy': np.linspace(0.85, 0.95, 10),
    'precision': np.linspace(0.84, 0.94, 10),
    'recall': np.linspace(0.83, 0.93, 10),
    'f1': np.linspace(0.83, 0.93, 10)
})

# Create sample feature importance
feature_names = [f"feature_{i}" for i in range(20)]
importance_values = np.random.rand(20)
importance_values = importance_values / importance_values.sum()  # Normalize

# %% [markdown]
# ### 3. Create Basic Visualizations

# %%
# Plot confusion matrix
cm_fig = plot_confusion_matrix(
    confusion_matrices["RandomForest"],
    classes=['Fake', 'Real'],
    title='Confusion Matrix - RandomForest',
    config=config
)

# Plot normalized confusion matrix
cm_norm_fig = plot_confusion_matrix(
    confusion_matrices["RandomForest"],
    classes=['Fake', 'Real'],
    title='Normalized Confusion Matrix - RandomForest',
    normalize=True,
    config=config
)

# %% [markdown]
# ### 4. Compare Model Performance

# %%
# Plot model comparison for accuracy
acc_fig = plot_model_comparison(
    model_metrics,
    metric='accuracy',
    title='Model Comparison - Accuracy',
    config=config
)

# Plot model comparison for F1 score
f1_fig = plot_model_comparison(
    model_metrics,
    metric='f1',
    title='Model Comparison - F1 Score',
    config=config
)

# %% [markdown]
# ### 5. Visualize Metrics Over Time

# %%
# Plot metrics over time
time_fig = plot_metrics_over_time(
    time_series_metrics,
    metrics=['accuracy', 'precision', 'recall', 'f1'],
    title='Performance Metrics Over Time',
    config=config
)

# %% [markdown]
# ### 6. Visualize Feature Importance

# %%
# Plot feature importance
feat_fig = plot_feature_importance(
    feature_names,
    importance_values,
    title='Feature Importance',
    top_n=10,
    config=config
)

# %% [markdown]
# ### 7. Create Interactive Visualizations

# %%
# Create interactive confusion matrix
interactive_cm = create_interactive_confusion_matrix(
    confusion_matrices["RandomForest"],
    classes=['Fake', 'Real'],
    title='Interactive Confusion Matrix - RandomForest'
)

# Create interactive model comparison
interactive_models = create_interactive_model_comparison(
    model_metrics,
    metrics=['accuracy', 'precision', 'recall', 'f1'],
    title='Interactive Model Performance Comparison'
)

# Create interactive time series
interactive_time = create_interactive_time_series(
    time_series_metrics,
    metrics=['accuracy', 'f1'],
    title='Interactive Performance Metrics Over Time'
)

# %% [markdown]
# ### 8. Create Interactive Dashboard

# %%
# Create interactive dashboard
dashboard = create_interactive_dashboard(
    model_metrics,
    confusion_matrices,
    time_series_metrics,
    title='Fake News Detection Dashboard'
)

# %% [markdown]
# ### 9. Export Data for Grafana

# %%
# Export model metrics for Grafana
metrics_path = export_metrics_for_grafana(
    model_metrics,
    config['grafana_export_path'],
    'model_metrics.json'
)

# Export time series metrics for Grafana
time_series_metrics.to_csv(
    os.path.join(config['grafana_export_path'], 'time_series_metrics.csv'),
    index=False
)

# Generate Grafana configuration
grafana_config = setup_grafana_datasource(
    metrics_path,
    'Fake News Detection Dashboard'
)

print(f"Grafana configuration generated: {grafana_config}")

# %% [markdown]
# ### 10. Run Complete Visualization Pipeline

# %%
# Run the complete visualization pipeline
results = run_visualization_pipeline(
    output_dir='/tmp/fake_news_detection/visualizations',
    grafana_export=True
)

# %% [markdown]
# ## Important Notes
# 
# 1. **Visualization Purpose**: Visualizations help interpret model performance, identify patterns in fake news, and communicate results effectively to stakeholders.
# 
# 2. **Static vs. Interactive**: This notebook provides both static visualizations (using Matplotlib/Seaborn) and interactive visualizations (using Plotly) to serve different needs.
# 
# 3. **Grafana Integration**: For real-time monitoring, data can be exported in formats compatible with Grafana dashboards.
# 
# 4. **Customization**: All visualization functions accept configuration parameters to customize appearance and behavior.
# 
# 5. **Performance Considerations**: For large datasets, consider:
#    - Using sampling for visualizations
#    - Limiting the number of features in importance plots
#    - Using static visualizations instead of interactive ones
# 
# 6. **Databricks Integration**: The code is optimized for Databricks Community Edition with appropriate configurations for memory and processing.
# 
# 7. **Saving Visualizations**: All visualizations can be saved to disk for sharing or inclusion in reports.
# 
# 8. **Dashboard Creation**: The interactive dashboard combines multiple visualizations into a single comprehensive view of model performance.
