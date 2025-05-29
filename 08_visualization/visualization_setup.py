"""
Visualization Setup for Fake News Detection

This module provides utilities for creating visualizations of fake news data,
including data quality metrics, feature distributions, and model performance.

The implementation uses Spark's distributed processing capabilities to ensure scalability,
with visualization outputs compatible with Databricks notebooks.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, count, when, isnan, isnull, countDistinct, 
    year, month, dayofmonth, dayofweek, 
    length, explode, split, avg, sum, max, min, stddev
)
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import io
import base64
from datetime import datetime

class VisualizationSetup:
    """
    A class for creating visualizations of fake news data.
    
    This class provides methods for visualizing data quality metrics,
    feature distributions, temporal patterns, and model performance.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize the VisualizationSetup.
        
        Args:
            spark (SparkSession): The Spark session to use for data processing.
        """
        self.spark = spark
        
        # Set default plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set(font_scale=1.2)
        
        # Default figure size
        self.fig_width = 12
        self.fig_height = 8
    
    def plot_data_quality_metrics(self, df: DataFrame) -> None:
        """
        Plot data quality metrics for the DataFrame.
        
        This method visualizes completeness, validity, and other quality metrics
        for each column in the DataFrame.
        
        Args:
            df (DataFrame): The input DataFrame with validation columns.
        """
        # Calculate completeness metrics
        completeness_metrics = []
        
        for column in df.columns:
            if not column.startswith("std_") and not column.endswith("_valid"):
                # Count non-null values
                non_null_count = df.filter(~isnull(col(column))).count()
                total_count = df.count()
                completeness = non_null_count / total_count if total_count > 0 else 0
                
                # Check if validation column exists
                validity = None
                if f"{column}_valid" in df.columns:
                    valid_count = df.filter(col(f"{column}_valid") == True).count()
                    validity = valid_count / total_count if total_count > 0 else 0
                
                completeness_metrics.append({
                    "column": column,
                    "completeness": completeness,
                    "validity": validity
                })
        
        # Convert to pandas for visualization
        metrics_df = pd.DataFrame(completeness_metrics)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot completeness
        metrics_df.plot(
            x="column", 
            y="completeness", 
            kind="bar", 
            color="skyblue", 
            ax=ax, 
            label="Completeness"
        )
        
        # Plot validity if available
        if not metrics_df["validity"].isna().all():
            metrics_df.plot(
                x="column", 
                y="validity", 
                kind="bar", 
                color="orange", 
                ax=ax, 
                label="Validity"
            )
        
        # Set labels and title
        ax.set_xlabel("Column")
        ax.set_ylabel("Percentage")
        ax.set_title("Data Quality Metrics by Column")
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Display the plot
        plt.show()
    
    def plot_temporal_distribution(self, df: DataFrame, date_column: str = "std_date") -> None:
        """
        Plot temporal distribution of fake news articles.
        
        This method visualizes the distribution of articles over time,
        including trends by year, month, and day of week.
        
        Args:
            df (DataFrame): The input DataFrame with a date column.
            date_column (str): The name of the standardized date column.
        """
        # Ensure date components are available
        if "year" not in df.columns or "month" not in df.columns or "day_of_week" not in df.columns:
            df = df.withColumn("year", year(col(date_column)))
            df = df.withColumn("month", month(col(date_column)))
            df = df.withColumn("day_of_week", dayofweek(col(date_column)))
        
        # Create distribution by year
        year_counts = df.groupBy("year").count().orderBy("year").toPandas()
        
        # Create distribution by month
        month_counts = df.groupBy("month").count().orderBy("month").toPandas()
        
        # Create distribution by day of week
        dow_counts = df.groupBy("day_of_week").count().orderBy("day_of_week").toPandas()
        
        # Map day of week numbers to names
        dow_names = {
            1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
            5: "Thursday", 6: "Friday", 7: "Saturday"
        }
        dow_counts["day_name"] = dow_counts["day_of_week"].map(dow_names)
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(self.fig_width * 1.5, self.fig_height))
        
        # Plot distribution by year
        ax1.bar(year_counts["year"], year_counts["count"], color="skyblue")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Number of Articles")
        ax1.set_title("Articles by Year")
        ax1.tick_params(axis="x", rotation=45)
        
        # Plot distribution by month
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        month_counts["month_name"] = month_counts["month"].map(month_names)
        ax2.bar(month_counts["month_name"], month_counts["count"], color="orange")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Number of Articles")
        ax2.set_title("Articles by Month")
        
        # Plot distribution by day of week
        ax3.bar(dow_counts["day_name"], dow_counts["count"], color="green")
        ax3.set_xlabel("Day of Week")
        ax3.set_ylabel("Number of Articles")
        ax3.set_title("Articles by Day of Week")
        ax3.tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_text_length_distribution(self, df: DataFrame, text_column: str = "std_text", title_column: str = "std_title") -> None:
        """
        Plot distribution of text and title lengths.
        
        This method visualizes the distribution of text and title lengths,
        which can help identify outliers and understand the data better.
        
        Args:
            df (DataFrame): The input DataFrame with text columns.
            text_column (str): The name of the standardized text column.
            title_column (str): The name of the standardized title column.
        """
        # Calculate text and title lengths
        df_with_lengths = df.withColumn("text_length", length(col(text_column)))
        df_with_lengths = df_with_lengths.withColumn("title_length", length(col(title_column)))
        
        # Convert to pandas for visualization
        text_lengths = df_with_lengths.select("text_length").toPandas()
        title_lengths = df_with_lengths.select("title_length").toPandas()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_width, self.fig_height))
        
        # Plot text length distribution
        sns.histplot(text_lengths["text_length"], bins=50, kde=True, ax=ax1, color="skyblue")
        ax1.set_xlabel("Text Length (characters)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Text Lengths")
        
        # Plot title length distribution
        sns.histplot(title_lengths["title_length"], bins=30, kde=True, ax=ax2, color="orange")
        ax2.set_xlabel("Title Length (characters)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Title Lengths")
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and display summary statistics
        text_stats = df_with_lengths.select(
            min("text_length").alias("min"),
            max("text_length").alias("max"),
            avg("text_length").alias("mean"),
            stddev("text_length").alias("stddev")
        ).toPandas()
        
        title_stats = df_with_lengths.select(
            min("title_length").alias("min"),
            max("title_length").alias("max"),
            avg("title_length").alias("mean"),
            stddev("title_length").alias("stddev")
        ).toPandas()
        
        print("Text Length Statistics:")
        print(f"  Min: {text_stats['min'][0]:.1f}")
        print(f"  Max: {text_stats['max'][0]:.1f}")
        print(f"  Mean: {text_stats['mean'][0]:.1f}")
        print(f"  StdDev: {text_stats['stddev'][0]:.1f}")
        
        print("\nTitle Length Statistics:")
        print(f"  Min: {title_stats['min'][0]:.1f}")
        print(f"  Max: {title_stats['max'][0]:.1f}")
        print(f"  Mean: {title_stats['mean'][0]:.1f}")
        print(f"  StdDev: {title_stats['stddev'][0]:.1f}")
    
    def plot_label_distribution(self, df: DataFrame, label_column: str = "label") -> None:
        """
        Plot distribution of fake vs. real news labels.
        
        This method visualizes the class distribution in the dataset,
        which is important for understanding class imbalance.
        
        Args:
            df (DataFrame): The input DataFrame with a label column.
            label_column (str): The name of the label column.
        """
        # Count instances by label
        label_counts = df.groupBy(label_column).count().toPandas()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot label distribution
        colors = ["skyblue", "orange"]
        ax.bar(label_counts[label_column].astype(str), label_counts["count"], color=colors)
        
        # Set labels and title
        ax.set_xlabel("Label (0 = Real, 1 = Fake)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Real vs. Fake News Articles")
        
        # Add count and percentage labels
        total = label_counts["count"].sum()
        for i, count in enumerate(label_counts["count"]):
            percentage = count / total * 100
            ax.text(i, count + 0.01 * total, f"{count}\n({percentage:.1f}%)", 
                    ha="center", va="bottom", fontweight="bold")
        
        plt.tight_layout()
        plt.show()
    
    def plot_top_sources(self, df: DataFrame, source_column: str = "std_source", top_n: int = 10) -> None:
        """
        Plot top news sources in the dataset.
        
        This method visualizes the most common news sources,
        which can help identify patterns in fake news distribution.
        
        Args:
            df (DataFrame): The input DataFrame with a source column.
            source_column (str): The name of the standardized source column.
            top_n (int): The number of top sources to display.
        """
        # Count articles by source
        source_counts = df.filter(col(source_column).isNotNull()) \
                          .groupBy(source_column) \
                          .count() \
                          .orderBy("count", ascending=False) \
                          .limit(top_n) \
                          .toPandas()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot source distribution
        source_counts.plot(
            x=source_column, 
            y="count", 
            kind="bar", 
            color="skyblue", 
            ax=ax
        )
        
        # Set labels and title
        ax.set_xlabel("Source")
        ax.set_ylabel("Count")
        ax.set_title(f"Top {top_n} News Sources")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    
    def plot_word_frequency(self, df: DataFrame, text_column: str = "std_text", top_n: int = 20) -> None:
        """
        Plot frequency of top words in the dataset.
        
        This method visualizes the most common words in the text,
        which can help identify important features for classification.
        
        Args:
            df (DataFrame): The input DataFrame with a text column.
            text_column (str): The name of the standardized text column.
            top_n (int): The number of top words to display.
        """
        # Tokenize text
        tokenized_df = df.withColumn("words", split(col(text_column), " "))
        
        # Explode words into separate rows
        words_df = tokenized_df.select(explode("words").alias("word"))
        
        # Count word frequencies
        word_counts = words_df.filter(length("word") > 3) \
                             .groupBy("word") \
                             .count() \
                             .orderBy("count", ascending=False) \
                             .limit(top_n) \
                             .toPandas()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot word frequencies
        word_counts.plot(
            x="word", 
            y="count", 
            kind="bar", 
            color="skyblue", 
            ax=ax
        )
        
        # Set labels and title
        ax.set_xlabel("Word")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Top {top_n} Words by Frequency")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df: DataFrame, numeric_columns: List[str]) -> None:
        """
        Plot correlation matrix for numeric features.
        
        This method visualizes the correlations between numeric features,
        which can help identify relationships and potential multicollinearity.
        
        Args:
            df (DataFrame): The input DataFrame with numeric columns.
            numeric_columns (List[str]): List of numeric column names.
        """
        # Select only numeric columns
        numeric_df = df.select(numeric_columns)
        
        # Convert to pandas for visualization
        pd_df = numeric_df.toPandas()
        
        # Calculate correlation matrix
        corr_matrix = pd_df.corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot correlation matrix
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        
        # Set title
        ax.set_title("Correlation Matrix of Numeric Features")
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Plot performance metrics for different models.
        
        This method visualizes the performance of different models,
        allowing for easy comparison.
        
        Args:
            metrics (Dict[str, Dict[str, float]]): Dictionary of model metrics.
                Format: {model_name: {metric_name: value}}
        """
        # Extract model names and metrics
        model_names = list(metrics.keys())
        metric_names = list(metrics[model_names[0]].keys())
        
        # Create a DataFrame for visualization
        data = []
        for model in model_names:
            for metric in metric_names:
                data.append({
                    "Model": model,
                    "Metric": metric,
                    "Value": metrics[model][metric]
                })
        
        metrics_df = pd.DataFrame(data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot model performance
        sns.barplot(x="Model", y="Value", hue="Metric", data=metrics_df, ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        
        # Adjust legend position
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        """
        Plot confusion matrix for a model.
        
        This method visualizes the confusion matrix,
        showing true positives, false positives, true negatives, and false negatives.
        
        Args:
            cm (np.ndarray): Confusion matrix as a 2x2 numpy array.
            model_name (str): Name of the model.
        """
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {model_name}")
        
        # Set tick labels
        ax.set_xticklabels(["Real", "Fake"])
        ax.set_yticklabels(["Real", "Fake"])
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importances: List[float], model_name: str) -> None:
        """
        Plot feature importance for a model.
        
        This method visualizes the importance of different features,
        which can help identify the most predictive features.
        
        Args:
            feature_names (List[str]): List of feature names.
            importances (List[float]): List of feature importance scores.
            model_name (str): Name of the model.
        """
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values("Importance", ascending=False)
        
        # Take top 20 features
        top_features = importance_df.head(20)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot feature importance
        sns.barplot(x="Importance", y="Feature", data=top_features, ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Feature Importance - {model_name}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self, train_sizes: List[int], train_scores: List[float], 
                           test_scores: List[float], model_name: str) -> None:
        """
        Plot learning curve for a model.
        
        This method visualizes how model performance changes with training set size,
        which can help identify overfitting or underfitting.
        
        Args:
            train_sizes (List[int]): List of training set sizes.
            train_scores (List[float]): List of training scores.
            test_scores (List[float]): List of test scores.
            model_name (str): Name of the model.
        """
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot learning curve
        ax.plot(train_sizes, train_scores, "o-", color="skyblue", label="Training Score")
        ax.plot(train_sizes, test_scores, "o-", color="orange", label="Validation Score")
        
        # Set labels and title
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Score")
        ax.set_title(f"Learning Curve - {model_name}")
        
        # Add legend
        ax.legend(loc="best")
        
        # Add grid
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, fpr: Dict[str, List[float]], tpr: Dict[str, List[float]], 
                      auc: Dict[str, float]) -> None:
        """
        Plot ROC curve for multiple models.
        
        This method visualizes the Receiver Operating Characteristic curve,
        which shows the trade-off between true positive rate and false positive rate.
        
        Args:
            fpr (Dict[str, List[float]]): Dictionary of false positive rates by model.
            tpr (Dict[str, List[float]]): Dictionary of true positive rates by model.
            auc (Dict[str, float]): Dictionary of AUC scores by model.
        """
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot ROC curve for each model
        for model_name in fpr.keys():
            ax.plot(fpr[model_name], tpr[model_name], 
                   label=f"{model_name} (AUC = {auc[model_name]:.3f})")
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        
        # Set labels and title
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        
        # Add legend
        ax.legend(loc="lower right")
        
        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        # Add grid
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, precision: Dict[str, List[float]], 
                                   recall: Dict[str, List[float]], 
                                   ap: Dict[str, float]) -> None:
        """
        Plot precision-recall curve for multiple models.
        
        This method visualizes the precision-recall curve,
        which is useful for imbalanced classification problems.
        
        Args:
            precision (Dict[str, List[float]]): Dictionary of precision values by model.
            recall (Dict[str, List[float]]): Dictionary of recall values by model.
            ap (Dict[str, float]): Dictionary of average precision scores by model.
        """
        # Create plot
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot precision-recall curve for each model
        for model_name in precision.keys():
            ax.plot(recall[model_name], precision[model_name], 
                   label=f"{model_name} (AP = {ap[model_name]:.3f})")
        
        # Set labels and title
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve Comparison")
        
        # Add legend
        ax.legend(loc="lower left")
        
        # Set axis limits
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        # Add grid
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def create_dashboard(self, df: DataFrame, metrics: Dict[str, Dict[str, float]] = None) -> str:
        """
        Create a comprehensive dashboard for fake news data.
        
        This method combines multiple visualizations into a single dashboard,
        providing an overview of the data and model performance.
        
        Args:
            df (DataFrame): The input DataFrame with processed data.
            metrics (Dict[str, Dict[str, float]]): Optional dictionary of model metrics.
            
        Returns:
            str: HTML string containing the dashboard.
        """
        # Create a list to store plot images
        plot_images = []
        
        # Data quality metrics
        plt.figure(figsize=(self.fig_width, self.fig_height))
        self.plot_data_quality_metrics(df)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close()
        
        # Temporal distribution
        plt.figure(figsize=(self.fig_width * 1.5, self.fig_height))
        self.plot_temporal_distribution(df)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close()
        
        # Text length distribution
        plt.figure(figsize=(self.fig_width, self.fig_height))
        self.plot_text_length_distribution(df)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close()
        
        # Label distribution
        plt.figure(figsize=(self.fig_width, self.fig_height))
        self.plot_label_distribution(df)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close()
        
        # Top sources
        plt.figure(figsize=(self.fig_width, self.fig_height))
        self.plot_top_sources(df)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close()
        
        # Word frequency
        plt.figure(figsize=(self.fig_width, self.fig_height))
        self.plot_word_frequency(df)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_images.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close()
        
        # Model performance (if provided)
        if metrics is not None:
            plt.figure(figsize=(self.fig_width, self.fig_height))
            self.plot_model_performance(metrics)
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_images.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close()
        
        # Create HTML dashboard
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fake News Detection Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                .dashboard {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                }}
                .plot {{
                    margin: 10px;
                    padding: 15px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
                .plot img {{
                    max-width: 100%;
                    height: auto;
                }}
                .plot h2 {{
                    color: #555;
                    font-size: 18px;
                    margin-top: 0;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 20px;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <h1>Fake News Detection Dashboard</h1>
            <div class="dashboard">
        """
        
        # Add plots to dashboard
        plot_titles = [
            "Data Quality Metrics",
            "Temporal Distribution",
            "Text Length Distribution",
            "Label Distribution",
            "Top News Sources",
            "Word Frequency",
            "Model Performance Comparison"
        ]
        
        for i, (image, title) in enumerate(zip(plot_images, plot_titles)):
            html += f"""
                <div class="plot">
                    <h2>{title}</h2>
                    <img src="data:image/png;base64,{image}" alt="{title}">
                </div>
            """
        
        # Add footer
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html += f"""
            </div>
            <div class="footer">
                <p>Generated on {current_time} | Total Records: {df.count()}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save_dashboard_to_html(self, df: DataFrame, output_path: str, 
                              metrics: Dict[str, Dict[str, float]] = None) -> None:
        """
        Save dashboard to an HTML file.
        
        This method creates a dashboard and saves it to an HTML file.
        
        Args:
            df (DataFrame): The input DataFrame with processed data.
            output_path (str): Path to save the HTML file.
            metrics (Dict[str, Dict[str, float]]): Optional dictionary of model metrics.
        """
        # Create dashboard
        html = self.create_dashboard(df, metrics)
        
        # Save to file
        with open(output_path, "w") as f:
            f.write(html)
        
        print(f"Dashboard saved to {output_path}")
    
    def export_to_grafana(self, df: DataFrame, metrics: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Export visualization data for Grafana.
        
        This method prepares data in a format suitable for Grafana dashboards,
        enabling real-time monitoring of fake news detection.
        
        Args:
            df (DataFrame): The input DataFrame with processed data.
            metrics (Dict[str, Dict[str, float]]): Optional dictionary of model metrics.
            
        Returns:
            Dict[str, Any]: Dictionary of data for Grafana.
        """
        # Calculate data quality metrics
        completeness_metrics = {}
        validity_metrics = {}
        
        for column in df.columns:
            if not column.startswith("std_") and not column.endswith("_valid"):
                # Count non-null values
                non_null_count = df.filter(~isnull(col(column))).count()
                total_count = df.count()
                completeness = non_null_count / total_count if total_count > 0 else 0
                completeness_metrics[column] = completeness
                
                # Check if validation column exists
                if f"{column}_valid" in df.columns:
                    valid_count = df.filter(col(f"{column}_valid") == True).count()
                    validity = valid_count / total_count if total_count > 0 else 0
                    validity_metrics[column] = validity
        
        # Calculate temporal distribution
        temporal_metrics = {}
        
        # Distribution by year
        year_counts = df.groupBy("year").count().orderBy("year").collect()
        temporal_metrics["year"] = {row["year"]: row["count"] for row in year_counts}
        
        # Distribution by month
        month_counts = df.groupBy("month").count().orderBy("month").collect()
        temporal_metrics["month"] = {row["month"]: row["count"] for row in month_counts}
        
        # Distribution by day of week
        dow_counts = df.groupBy("day_of_week").count().orderBy("day_of_week").collect()
        temporal_metrics["day_of_week"] = {row["day_of_week"]: row["count"] for row in dow_counts}
        
        # Calculate label distribution
        label_counts = df.groupBy("label").count().collect()
        label_metrics = {row["label"]: row["count"] for row in label_counts}
        
        # Prepare Grafana data
        grafana_data = {
            "data_quality": {
                "completeness": completeness_metrics,
                "validity": validity_metrics
            },
            "temporal_distribution": temporal_metrics,
            "label_distribution": label_metrics,
            "total_records": df.count(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add model metrics if provided
        if metrics is not None:
            grafana_data["model_performance"] = metrics
        
        return grafana_data


def create_visualizations(spark: SparkSession, df: DataFrame, output_path: str = None) -> None:
    """
    Create visualizations for fake news data.
    
    This function creates various visualizations for fake news data,
    providing insights into data quality, distributions, and patterns.
    
    Args:
        spark (SparkSession): The Spark session to use for data processing.
        df (DataFrame): The input DataFrame with processed data.
        output_path (str): Optional path to save the dashboard HTML file.
    """
    # Create visualization setup
    viz = VisualizationSetup(spark)
    
    # Plot data quality metrics
    viz.plot_data_quality_metrics(df)
    
    # Plot temporal distribution
    viz.plot_temporal_distribution(df)
    
    # Plot text length distribution
    viz.plot_text_length_distribution(df)
    
    # Plot label distribution
    viz.plot_label_distribution(df)
    
    # Plot top sources
    viz.plot_top_sources(df)
    
    # Plot word frequency
    viz.plot_word_frequency(df)
    
    # Save dashboard to HTML if output path is provided
    if output_path is not None:
        viz.save_dashboard_to_html(df, output_path)


# Example usage:
# from pyspark.sql import SparkSession
# 
# # Create a Spark session
# spark = SparkSession.builder.appName("FakeNewsVisualization").getOrCreate()
# 
# # Load processed data
# df = spark.read.parquet("/path/to/processed_data.parquet")
# 
# # Create visualizations
# create_visualizations(spark, df, output_path="/path/to/dashboard.html")
# 
# # Create visualization setup for more customized plots
# viz = VisualizationSetup(spark)
# 
# # Plot model performance
# model_metrics = {
#     "Logistic Regression": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85},
#     "Random Forest": {"accuracy": 0.88, "precision": 0.90, "recall": 0.85, "f1": 0.87},
#     "LSTM": {"accuracy": 0.92, "precision": 0.94, "recall": 0.90, "f1": 0.92}
# }
# viz.plot_model_performance(model_metrics)
# 
# # Plot confusion matrix
# confusion_matrix = np.array([[120, 30], [20, 130]])
# viz.plot_confusion_matrix(confusion_matrix, "Random Forest")
# 
# # Plot feature importance
# feature_names = ["word_count", "contains_question", "sentiment_score", "avg_word_length", "title_length"]
# importances = [0.35, 0.25, 0.20, 0.15, 0.05]
# viz.plot_feature_importance(feature_names, importances, "Random Forest")
