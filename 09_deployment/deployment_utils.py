"""
Deployment Utilities for Fake News Detection Pipeline

This module provides utilities for deploying the fake news detection pipeline
to Databricks Community Edition, including packaging, initialization scripts,
and deployment instructions.

The implementation ensures compatibility with Databricks Community Edition
and provides a streamlined deployment process.
"""

import os
import shutil
import zipfile
import json
import subprocess
import base64
from typing import Dict, List, Optional, Any, Tuple
import datetime
import re

class DeploymentUtils:
    """
    A class for deploying the fake news detection pipeline to Databricks.
    
    This class provides methods for packaging code, creating initialization scripts,
    and deploying to Databricks Community Edition.
    """
    
    def __init__(self, project_dir: str, output_dir: str = None):
        """
        Initialize the DeploymentUtils.
        
        Args:
            project_dir (str): The root directory of the project.
            output_dir (str): The directory to store deployment artifacts.
        """
        self.project_dir = os.path.abspath(project_dir)
        self.output_dir = os.path.abspath(output_dir) if output_dir else os.path.join(self.project_dir, "deployment")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define component directories
        self.components = [
            "01_data_ingestion",
            "02_preprocessing",
            "03_feature_engineering",
            "04_modeling",
            "05_graph_analysis",
            "06_clustering",
            "07_streaming",
            "08_visualization",
            "09_deployment"
        ]
        
        # Define required Python packages
        self.required_packages = [
            "pyspark==3.3.2",
            "numpy",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "nltk",
            "graphframes"
        ]
    
    def create_deployment_package(self, include_data: bool = False) -> str:
        """
        Create a deployment package for Databricks.
        
        This method packages the project code and dependencies into a zip file
        that can be uploaded to Databricks.
        
        Args:
            include_data (bool): Whether to include data files in the package.
            
        Returns:
            str: Path to the created deployment package.
        """
        # Create a timestamp for the package name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"fake_news_detection_{timestamp}.zip"
        package_path = os.path.join(self.output_dir, package_name)
        
        # Create a temporary directory for packaging
        temp_dir = os.path.join(self.output_dir, "temp_package")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Copy Python files and notebooks to the temporary directory
            for component in self.components:
                component_dir = os.path.join(self.project_dir, component)
                if os.path.exists(component_dir):
                    # Create component directory in the temporary directory
                    temp_component_dir = os.path.join(temp_dir, component)
                    os.makedirs(temp_component_dir, exist_ok=True)
                    
                    # Copy Python files
                    for file in os.listdir(component_dir):
                        if file.endswith(".py") or file.endswith(".ipynb"):
                            src_file = os.path.join(component_dir, file)
                            dst_file = os.path.join(temp_component_dir, file)
                            shutil.copy2(src_file, dst_file)
            
            # Copy documentation
            docs_dir = os.path.join(self.project_dir, "docs")
            if os.path.exists(docs_dir):
                temp_docs_dir = os.path.join(temp_dir, "docs")
                os.makedirs(temp_docs_dir, exist_ok=True)
                
                for file in os.listdir(docs_dir):
                    if file.endswith(".md"):
                        src_file = os.path.join(docs_dir, file)
                        dst_file = os.path.join(temp_docs_dir, file)
                        shutil.copy2(src_file, dst_file)
            
            # Include data files if requested
            if include_data:
                data_dir = os.path.join(self.project_dir, "data")
                if os.path.exists(data_dir):
                    temp_data_dir = os.path.join(temp_dir, "data")
                    os.makedirs(temp_data_dir, exist_ok=True)
                    
                    for file in os.listdir(data_dir):
                        if file.endswith(".csv") or file.endswith(".parquet"):
                            src_file = os.path.join(data_dir, file)
                            dst_file = os.path.join(temp_data_dir, file)
                            shutil.copy2(src_file, dst_file)
            
            # Create requirements.txt
            with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
                f.write("\n".join(self.required_packages))
            
            # Create README.md with deployment instructions
            with open(os.path.join(temp_dir, "README.md"), "w") as f:
                f.write(self._generate_readme())
            
            # Create initialization script
            with open(os.path.join(temp_dir, "init_script.sh"), "w") as f:
                f.write(self._generate_init_script())
            
            # Create the zip file
            with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
            
            print(f"Deployment package created at: {package_path}")
            return package_path
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_databricks_notebook(self, component: str, output_path: Optional[str] = None) -> str:
        """
        Create a Databricks notebook for a specific component.
        
        This method generates a Databricks notebook that imports and uses
        the code from a specific component.
        
        Args:
            component (str): The component to create a notebook for.
            output_path (str): Optional path to save the notebook.
            
        Returns:
            str: Path to the created notebook.
        """
        # Validate component
        if component not in self.components:
            raise ValueError(f"Invalid component: {component}. Must be one of {self.components}")
        
        # Determine component directory
        component_dir = os.path.join(self.project_dir, component)
        if not os.path.exists(component_dir):
            raise ValueError(f"Component directory not found: {component_dir}")
        
        # Find Python files in the component directory
        py_files = [f for f in os.listdir(component_dir) if f.endswith(".py")]
        if not py_files:
            raise ValueError(f"No Python files found in component directory: {component_dir}")
        
        # Generate notebook content
        notebook_content = self._generate_notebook_content(component, py_files)
        
        # Determine output path
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{component}_notebook.ipynb")
        
        # Write notebook to file
        with open(output_path, "w") as f:
            f.write(notebook_content)
        
        print(f"Databricks notebook created at: {output_path}")
        return output_path
    
    def create_all_databricks_notebooks(self) -> List[str]:
        """
        Create Databricks notebooks for all components.
        
        This method generates Databricks notebooks for all components
        in the project.
        
        Returns:
            List[str]: Paths to the created notebooks.
        """
        notebook_paths = []
        
        for component in self.components:
            try:
                notebook_path = self.create_databricks_notebook(component)
                notebook_paths.append(notebook_path)
            except ValueError as e:
                print(f"Warning: {str(e)}")
        
        return notebook_paths
    
    def create_deployment_instructions(self, output_path: Optional[str] = None) -> str:
        """
        Create detailed deployment instructions.
        
        This method generates a markdown document with detailed instructions
        for deploying the fake news detection pipeline to Databricks.
        
        Args:
            output_path (str): Optional path to save the instructions.
            
        Returns:
            str: Path to the created instructions.
        """
        # Generate instructions content
        instructions_content = self._generate_deployment_instructions()
        
        # Determine output path
        if output_path is None:
            output_path = os.path.join(self.output_dir, "deployment_instructions.md")
        
        # Write instructions to file
        with open(output_path, "w") as f:
            f.write(instructions_content)
        
        print(f"Deployment instructions created at: {output_path}")
        return output_path
    
    def convert_py_to_notebook(self, py_file: str, output_path: Optional[str] = None) -> str:
        """
        Convert a Python file to a Jupyter notebook.
        
        This method converts a Python file to a Jupyter notebook using jupytext.
        
        Args:
            py_file (str): Path to the Python file.
            output_path (str): Optional path to save the notebook.
            
        Returns:
            str: Path to the created notebook.
        """
        # Validate Python file
        if not os.path.exists(py_file):
            raise ValueError(f"Python file not found: {py_file}")
        
        # Determine output path
        if output_path is None:
            output_path = os.path.splitext(py_file)[0] + ".ipynb"
        
        # Convert Python file to notebook using jupytext
        try:
            subprocess.run(["jupytext", "--to", "notebook", py_file, "-o", output_path], check=True)
            print(f"Notebook created at: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert Python file to notebook: {str(e)}")
        except FileNotFoundError:
            raise RuntimeError("jupytext not found. Please install it with: pip install jupytext")
    
    def convert_all_py_to_notebooks(self) -> List[str]:
        """
        Convert all Python files to Jupyter notebooks.
        
        This method converts all Python files in the project to Jupyter notebooks.
        
        Returns:
            List[str]: Paths to the created notebooks.
        """
        notebook_paths = []
        
        for component in self.components:
            component_dir = os.path.join(self.project_dir, component)
            if os.path.exists(component_dir):
                for file in os.listdir(component_dir):
                    if file.endswith(".py"):
                        py_file = os.path.join(component_dir, file)
                        try:
                            notebook_path = self.convert_py_to_notebook(py_file)
                            notebook_paths.append(notebook_path)
                        except (ValueError, RuntimeError) as e:
                            print(f"Warning: {str(e)}")
        
        return notebook_paths
    
    def _generate_readme(self) -> str:
        """
        Generate README content for the deployment package.
        
        Returns:
            str: README content.
        """
        return f"""# Fake News Detection Pipeline

## Overview

This package contains the code and resources for deploying a fake news detection pipeline
to Databricks Community Edition. The pipeline includes data ingestion, preprocessing,
feature engineering, modeling, and visualization components.

## Directory Structure

- `01_data_ingestion/`: Data loading and preparation
- `02_preprocessing/`: Text cleaning and normalization
- `03_feature_engineering/`: Feature extraction
- `04_modeling/`: ML model implementation
- `05_graph_analysis/`: GraphX-based analysis
- `06_clustering/`: Clustering and topic modeling
- `07_streaming/`: Real-time detection pipeline
- `08_visualization/`: Dashboard setup
- `09_deployment/`: Databricks deployment utilities
- `docs/`: Documentation and tutorials
- `requirements.txt`: Required Python packages
- `init_script.sh`: Initialization script for Databricks

## Deployment Instructions

1. Upload this package to Databricks:
   - Go to Databricks workspace
   - Click on "Data" in the sidebar
   - Click "Create Table" and then "Upload File"
   - Select this zip file and upload it
   - Note the DBFS path where the file is uploaded

2. Create a new cluster:
   - Go to "Compute" in the sidebar
   - Click "Create Cluster"
   - Configure with appropriate settings (see deployment_instructions.md)
   - Add initialization script (copy content from init_script.sh)
   - Start the cluster

3. Import notebooks:
   - Go to "Workspace" in the sidebar
   - Click on the dropdown next to your username
   - Select "Import"
   - Import the notebooks from this package

4. Run the pipeline:
   - Start with the data ingestion notebook
   - Follow the sequence of notebooks as described in the documentation

For detailed instructions, see `deployment_instructions.md`.

## Requirements

{os.linesep.join(self.required_packages)}

## Created on

{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    def _generate_init_script(self) -> str:
        """
        Generate initialization script for Databricks.
        
        Returns:
            str: Initialization script content.
        """
        return """#!/bin/bash

# Initialization script for Fake News Detection Pipeline

# Install required Python packages
pip install --upgrade pip
pip install pyspark==3.3.2 numpy pandas scikit-learn matplotlib seaborn nltk graphframes

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

# Set up environment variables
export PYSPARK_PYTHON=/databricks/python3/bin/python3
export PYSPARK_DRIVER_PYTHON=/databricks/python3/bin/python3

# Create directories
mkdir -p /dbfs/FileStore/fake_news_detection/data
mkdir -p /dbfs/FileStore/fake_news_detection/models
mkdir -p /dbfs/FileStore/fake_news_detection/logs

# Log initialization completion
echo "Fake News Detection Pipeline initialization completed at $(date)" > /dbfs/FileStore/fake_news_detection/logs/init_log.txt

exit 0
"""
    
    def _generate_notebook_content(self, component: str, py_files: List[str]) -> str:
        """
        Generate notebook content for a component.
        
        Args:
            component (str): The component name.
            py_files (List[str]): List of Python files in the component.
            
        Returns:
            str: Notebook content in JSON format.
        """
        # Create notebook cells
        cells = []
        
        # Add title cell
        component_title = component.split("_")[-1].capitalize()
        title_cell = {
            "cell_type": "markdown",
            "source": [f"# Fake News Detection - {component_title}\n\n",
                      f"This notebook implements the {component_title} component of the fake news detection pipeline.\n\n",
                      "## Setup\n\n",
                      "First, let's set up the Spark session with appropriate configuration for our workload."]
        }
        cells.append(title_cell)
        
        # Add Spark session setup cell
        spark_setup_cell = {
            "cell_type": "code",
            "source": ["# Create a Spark session with appropriate configuration for data processing\n",
                      "# - appName: Identifies this application in the Spark UI and logs\n",
                      "# - master: In Databricks, this is automatically configured\n",
                      "# - config: We add various configurations to optimize Spark for our workload\n",
                      "from pyspark.sql import SparkSession\n\n",
                      "spark = SparkSession.builder \\\n",
                      f"    .appName(\"FakeNews{component_title}\") \\\n",
                      "    .config(\"spark.sql.files.maxPartitionBytes\", \"128MB\") \\\n",
                      "    .config(\"spark.sql.shuffle.partitions\", \"200\") \\\n",
                      "    .config(\"spark.memory.offHeap.enabled\", \"true\") \\\n",
                      "    .config(\"spark.memory.offHeap.size\", \"1g\") \\\n",
                      "    .getOrCreate()\n\n",
                      "# Display Spark version information\n",
                      "print(f\"Spark version: {spark.version}\")\n"],
            "metadata": {},
            "execution_count": None,
            "outputs": []
        }
        cells.append(spark_setup_cell)
        
        # Add explanation cell for Spark configuration
        spark_explanation_cell = {
            "cell_type": "markdown",
            "source": ["## Spark Configuration Explanation\n\n",
                      "The Spark session is configured with specific parameters to optimize performance:\n\n",
                      "- **appName**: Sets a meaningful name for identification in the Spark UI and logs\n",
                      "- **spark.sql.files.maxPartitionBytes**: Controls partition size when reading files, optimizing parallelism\n",
                      "- **spark.sql.shuffle.partitions**: Sets partition count for shuffle operations, balancing parallelism with overhead\n",
                      "- **spark.memory.offHeap.enabled**: Enables off-heap memory to reduce GC overhead and improve performance\n",
                      "- **spark.memory.offHeap.size**: Allocates 1GB of off-heap memory, appropriate for text processing\n\n",
                      "These settings are optimized for text processing in Databricks Community Edition."]
        }
        cells.append(spark_explanation_cell)
        
        # Add import cells for each Python file
        for py_file in py_files:
            module_name = os.path.splitext(py_file)[0]
            
            # Add markdown cell explaining the module
            module_explanation = {
                "cell_type": "markdown",
                "source": [f"## {module_name.replace('_', ' ').title()}\n\n",
                          f"Now we'll import and use the `{module_name}` module, which provides functionality for {component_title.lower()}."]
            }
            cells.append(module_explanation)
            
            # Add code cell to import the module
            import_cell = {
                "cell_type": "code",
                "source": [f"# Import the {module_name} module\n",
                          f"from {component}.{module_name} import *\n\n",
                          f"# Display available functions and classes\n",
                          f"print(f\"Available in {module_name}: {{[name for name in dir() if not name.startswith('_') and name not in globals()['_oh']]}}\")"],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            }
            cells.append(import_cell)
        
        # Add usage example cell
        usage_cell = {
            "cell_type": "markdown",
            "source": ["## Usage Example\n\n",
                      "Here's an example of how to use the imported modules:"]
        }
        cells.append(usage_cell)
        
        # Add code cell with usage example
        example_code = self._generate_example_code(component, py_files)
        example_cell = {
            "cell_type": "code",
            "source": example_code,
            "metadata": {},
            "execution_count": None,
            "outputs": []
        }
        cells.append(example_cell)
        
        # Create notebook JSON
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.10"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return json.dumps(notebook, indent=2)
    
    def _generate_example_code(self, component: str, py_files: List[str]) -> List[str]:
        """
        Generate example code for a component.
        
        Args:
            component (str): The component name.
            py_files (List[str]): List of Python files in the component.
            
        Returns:
            List[str]: Example code lines.
        """
        code_lines = ["# Example usage of the imported modules\n"]
        
        if component == "01_data_ingestion":
            code_lines.extend([
                "# Load data from CSV files\n",
                "data_path = \"/dbfs/FileStore/fake_news_detection/data/\"\n",
                "df = spark.read.csv(data_path + \"fake_news.csv\", header=True, inferSchema=True)\n\n",
                "# Display the first few rows\n",
                "df.show(5)\n\n",
                "# Get basic statistics\n",
                "df.describe().show()\n\n",
                "# Count rows by label\n",
                "df.groupBy(\"label\").count().show()"
            ])
        elif component == "02_preprocessing":
            code_lines.extend([
                "# Load sample data\n",
                "data_path = \"/dbfs/FileStore/fake_news_detection/data/\"\n",
                "df = spark.read.csv(data_path + \"fake_news.csv\", header=True, inferSchema=True)\n\n",
                "# Process dates\n",
                "from datetime import datetime\n",
                "df_with_dates = process_dates(df, date_column=\"publish_date\")\n\n",
                "# Validate and clean data\n",
                "df_cleaned, metrics = validate_and_clean_data(df_with_dates)\n\n",
                "# Show data quality metrics\n",
                "print(f\"Data quality metrics: {metrics}\")\n\n",
                "# Show processed data\n",
                "df_cleaned.select(\"title\", \"std_title\", \"title_valid\", \"year\", \"month\", \"day\").show(5)"
            ])
        elif component == "03_feature_engineering":
            code_lines.extend([
                "# Load preprocessed data\n",
                "data_path = \"/dbfs/FileStore/fake_news_detection/data/\"\n",
                "df = spark.read.parquet(data_path + \"preprocessed_data.parquet\")\n\n",
                "# Extract features\n",
                "from pyspark.ml.feature import HashingTF, IDF, Tokenizer\n",
                "from pyspark.sql.functions import col, udf\n",
                "from pyspark.sql.types import ArrayType, StringType, FloatType\n\n",
                "# Tokenize text\n",
                "tokenizer = Tokenizer(inputCol=\"std_text\", outputCol=\"words\")\n",
                "df_tokenized = tokenizer.transform(df)\n\n",
                "# Create TF-IDF features\n",
                "hashingTF = HashingTF(inputCol=\"words\", outputCol=\"tf\", numFeatures=10000)\n",
                "df_tf = hashingTF.transform(df_tokenized)\n",
                "idf = IDF(inputCol=\"tf\", outputCol=\"tfidf\")\n",
                "idf_model = idf.fit(df_tf)\n",
                "df_tfidf = idf_model.transform(df_tf)\n\n",
                "# Show features\n",
                "df_tfidf.select(\"std_title\", \"tfidf\").show(5, truncate=False)"
            ])
        elif component == "04_modeling":
            code_lines.extend([
                "# Load feature data\n",
                "data_path = \"/dbfs/FileStore/fake_news_detection/data/\"\n",
                "df = spark.read.parquet(data_path + \"features.parquet\")\n\n",
                "# Split data into training and testing sets\n",
                "train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)\n\n",
                "# Train logistic regression model\n",
                "from pyspark.ml.classification import LogisticRegression\n",
                "from pyspark.ml.evaluation import BinaryClassificationEvaluator\n\n",
                "lr = LogisticRegression(featuresCol=\"tfidf\", labelCol=\"label\")\n",
                "lr_model = lr.fit(train_df)\n\n",
                "# Make predictions\n",
                "predictions = lr_model.transform(test_df)\n\n",
                "# Evaluate model\n",
                "evaluator = BinaryClassificationEvaluator(labelCol=\"label\")\n",
                "auc = evaluator.evaluate(predictions)\n",
                "print(f\"AUC: {auc}\")\n\n",
                "# Show predictions\n",
                "predictions.select(\"std_title\", \"label\", \"prediction\", \"probability\").show(5)"
            ])
        elif component == "05_graph_analysis":
            code_lines.extend([
                "# Load preprocessed data\n",
                "data_path = \"/dbfs/FileStore/fake_news_detection/data/\"\n",
                "df = spark.read.parquet(data_path + \"preprocessed_data.parquet\")\n\n",
                "# Extract entities\n",
                "from pyspark.sql.functions import explode, split, col\n",
                "import nltk\n",
                "nltk.download('punkt')\n",
                "nltk.download('averaged_perceptron_tagger')\n",
                "nltk.download('maxent_ne_chunker')\n",
                "nltk.download('words')\n\n",
                "# Create entity graph\n",
                "from graphframes import GraphFrame\n\n",
                "# Extract entities (simplified example)\n",
                "df_entities = df.withColumn(\"entities\", split(col(\"std_text\"), \" \"))\n",
                "df_exploded = df_entities.select(\"id\", explode(\"entities\").alias(\"entity\"))\n\n",
                "# Create vertices\n",
                "vertices = df_exploded.select(\"entity\").distinct().withColumnRenamed(\"entity\", \"id\")\n\n",
                "# Create edges\n",
                "edges = df_exploded.join(df_exploded, \"id\").filter(\"entity != entity_1\")\\\n",
                "    .select(col(\"entity\").alias(\"src\"), col(\"entity_1\").alias(\"dst\"))\n\n",
                "# Create graph\n",
                "g = GraphFrame(vertices, edges)\n\n",
                "# Run PageRank\n",
                "results = g.pageRank(resetProbability=0.15, tol=0.01)\n",
                "results.vertices.select(\"id\", \"pagerank\").orderBy(\"pagerank\", ascending=False).show(10)"
            ])
        elif component == "06_clustering":
            code_lines.extend([
                "# Load feature data\n",
                "data_path = \"/dbfs/FileStore/fake_news_detection/data/\"\n",
                "df = spark.read.parquet(data_path + \"features.parquet\")\n\n",
                "# Perform clustering\n",
                "from pyspark.ml.clustering import KMeans\n",
                "from pyspark.ml.evaluation import ClusteringEvaluator\n\n",
                "# Train KMeans model\n",
                "kmeans = KMeans(featuresCol=\"tfidf\", k=5, seed=42)\n",
                "kmeans_model = kmeans.fit(df)\n\n",
                "# Make predictions\n",
                "predictions = kmeans_model.transform(df)\n\n",
                "# Evaluate clustering\n",
                "evaluator = ClusteringEvaluator(featuresCol=\"tfidf\")\n",
                "silhouette = evaluator.evaluate(predictions)\n",
                "print(f\"Silhouette with squared euclidean distance: {silhouette}\")\n\n",
                "# Show cluster centers\n",
                "centers = kmeans_model.clusterCenters()\n",
                "print(\"Cluster Centers:\")\n",
                "for i, center in enumerate(centers):\n",
                "    print(f\"Cluster {i}: {center[:10]}...\")\n\n",
                "# Show predictions\n",
                "predictions.select(\"std_title\", \"prediction\").show(5)"
            ])
        elif component == "07_streaming":
            code_lines.extend([
                "# Set up streaming context\n",
                "from pyspark.sql import functions as F\n",
                "from pyspark.sql.types import StructType, StructField, StringType, IntegerType\n",
                "from pyspark.ml.classification import LogisticRegressionModel\n\n",
                "# Define schema for streaming data\n",
                "schema = StructType([\n",
                "    StructField(\"id\", StringType(), True),\n",
                "    StructField(\"title\", StringType(), True),\n",
                "    StructField(\"text\", StringType(), True),\n",
                "    StructField(\"author\", StringType(), True),\n",
                "    StructField(\"publish_date\", StringType(), True)\n",
                "])\n\n",
                "# Create streaming DataFrame\n",
                "streaming_df = spark.readStream \\\n",
                "    .format(\"csv\") \\\n",
                "    .schema(schema) \\\n",
                "    .option(\"header\", \"true\") \\\n",
                "    .option(\"maxFilesPerTrigger\", 1) \\\n",
                "    .load(\"/dbfs/FileStore/fake_news_detection/streaming/\")\n\n",
                "# Load pre-trained model\n",
                "model_path = \"/dbfs/FileStore/fake_news_detection/models/lr_model\"\n",
                "model = LogisticRegressionModel.load(model_path)\n\n",
                "# Process streaming data (simplified)\n",
                "def process_batch(batch_df, batch_id):\n",
                "    # Preprocess batch\n",
                "    processed_df = preprocess_batch(batch_df)\n",
                "    \n",
                "    # Make predictions\n",
                "    predictions = model.transform(processed_df)\n",
                "    \n",
                "    # Save results\n",
                "    predictions.select(\"id\", \"title\", \"prediction\", \"probability\") \\\n",
                "        .write.mode(\"append\") \\\n",
                "        .parquet(\"/dbfs/FileStore/fake_news_detection/results/\")\n\n",
                "# Start streaming query\n",
                "query = streaming_df.writeStream \\\n",
                "    .foreachBatch(process_batch) \\\n",
                "    .outputMode(\"update\") \\\n",
                "    .trigger(processingTime=\"1 minute\") \\\n",
                "    .start()\n\n",
                "# Wait for termination\n",
                "# query.awaitTermination()"
            ])
        elif component == "08_visualization":
            code_lines.extend([
                "# Load processed data\n",
                "data_path = \"/dbfs/FileStore/fake_news_detection/data/\"\n",
                "df = spark.read.parquet(data_path + \"preprocessed_data.parquet\")\n\n",
                "# Create visualization setup\n",
                "viz = VisualizationSetup(spark)\n\n",
                "# Plot data quality metrics\n",
                "viz.plot_data_quality_metrics(df)\n\n",
                "# Plot temporal distribution\n",
                "viz.plot_temporal_distribution(df)\n\n",
                "# Plot text length distribution\n",
                "viz.plot_text_length_distribution(df)\n\n",
                "# Plot label distribution\n",
                "viz.plot_label_distribution(df)\n\n",
                "# Plot top sources\n",
                "viz.plot_top_sources(df)\n\n",
                "# Plot word frequency\n",
                "viz.plot_word_frequency(df)\n\n",
                "# Create dashboard\n",
                "dashboard_html = viz.create_dashboard(df)\n",
                "displayHTML(dashboard_html)"
            ])
        elif component == "09_deployment":
            code_lines.extend([
                "# Set up deployment utilities\n",
                "project_dir = \"/dbfs/FileStore/fake_news_detection/\"\n",
                "output_dir = \"/dbfs/FileStore/fake_news_detection/deployment/\"\n\n",
                "# Create deployment utils instance\n",
                "deployment = DeploymentUtils(project_dir, output_dir)\n\n",
                "# Create deployment package\n",
                "package_path = deployment.create_deployment_package(include_data=False)\n",
                "print(f\"Deployment package created at: {package_path}\")\n\n",
                "# Create deployment instructions\n",
                "instructions_path = deployment.create_deployment_instructions()\n",
                "print(f\"Deployment instructions created at: {instructions_path}\")\n\n",
                "# Display instructions\n",
                "with open(instructions_path, 'r') as f:\n",
                "    instructions = f.read()\n",
                "displayHTML(f\"<pre>{instructions}</pre>\")"
            ])
        else:
            code_lines.extend([
                "# Example usage for this component\n",
                "print(f\"This is an example for the {component} component.\")\n\n",
                "# TODO: Add specific examples for this component"
            ])
        
        return code_lines
    
    def _generate_deployment_instructions(self) -> str:
        """
        Generate detailed deployment instructions.
        
        Returns:
            str: Deployment instructions content.
        """
        return f"""# Fake News Detection Pipeline - Deployment Instructions

## Overview

This document provides detailed instructions for deploying the fake news detection pipeline
to Databricks Community Edition. The pipeline includes data ingestion, preprocessing,
feature engineering, modeling, and visualization components.

## Prerequisites

- Databricks Community Edition account
- Basic knowledge of Databricks and Spark
- The deployment package created with `create_deployment_package()`

## Step 1: Set Up Databricks Community Edition

1. Sign up for Databricks Community Edition at https://databricks.com/try-databricks
2. Log in to your Databricks account
3. Familiarize yourself with the Databricks interface

## Step 2: Create a Cluster

1. Go to "Compute" in the sidebar
2. Click "Create Cluster"
3. Configure the cluster with the following settings:
   - Cluster Name: FakeNewsDetection
   - Cluster Mode: Single Node
   - Databricks Runtime Version: 12.2 LTS (Scala 2.12, Spark 3.3.2)
   - Node Type: Standard_DS3_v2 (or equivalent)
   - Terminate after: 60 minutes of inactivity
4. Under "Advanced Options" > "Init Scripts", add a new init script:
   - Copy the content from `init_script.sh` in the deployment package
   - This script installs required packages and sets up the environment
5. Click "Create Cluster"
6. Wait for the cluster to start

## Step 3: Upload Data and Code

1. Go to "Data" in the sidebar
2. Click "Create Table" and then "Upload File"
3. Upload the deployment package zip file
4. Note the DBFS path where the file is uploaded (e.g., `/FileStore/tables/fake_news_detection_20250528_123456.zip`)
5. Create a new notebook to unzip the package:
   ```python
   # Unzip the deployment package
   dbutils.fs.mkdirs("/FileStore/fake_news_detection")
   import zipfile
   import os
   
   # Update this path to match your uploaded zip file
   zip_path = "/dbfs/FileStore/tables/fake_news_detection_20250528_123456.zip"
   extract_path = "/dbfs/FileStore/fake_news_detection"
   
   with zipfile.ZipFile(zip_path, 'r') as zip_ref:
       zip_ref.extractall(extract_path)
   
   print("Files extracted to:", extract_path)
   dbutils.fs.ls("/FileStore/fake_news_detection")
   ```
6. Run the notebook to extract the files

## Step 4: Upload Sample Data

1. Go to "Data" in the sidebar
2. Click "Create Table" and then "Upload File"
3. Upload your fake news dataset CSV files
4. Note the DBFS paths where the files are uploaded

## Step 5: Import Notebooks

1. Go to "Workspace" in the sidebar
2. Click on the dropdown next to your username
3. Select "Import"
4. In the "Import Notebooks" dialog:
   - Select "File" as the source
   - Upload the notebook files from the deployment package
   - Choose a workspace folder to import to
5. Click "Import"
6. Repeat for all notebook files

## Step 6: Configure Notebooks

1. Open each imported notebook
2. Update data paths to match your DBFS paths
3. Attach the notebook to your cluster

## Step 7: Run the Pipeline

Run the notebooks in the following order:

1. `01_data_ingestion_notebook.ipynb`: Load and prepare data
2. `02_preprocessing_notebook.ipynb`: Clean and normalize text
3. `03_feature_engineering_notebook.ipynb`: Extract features
4. `04_modeling_notebook.ipynb`: Train and evaluate models
5. `05_graph_analysis_notebook.ipynb`: Analyze entity relationships
6. `06_clustering_notebook.ipynb`: Perform topic clustering
7. `07_streaming_notebook.ipynb`: Set up real-time detection
8. `08_visualization_notebook.ipynb`: Create visualizations and dashboards

## Step 8: Schedule Jobs (Optional)

1. Go to "Workflows" in the sidebar
2. Click "Create Job"
3. Add tasks for each notebook in the pipeline
4. Configure the job schedule
5. Click "Create"

## Troubleshooting

### Common Issues

1. **Package Installation Failures**:
   - Check the cluster logs for error messages
   - Ensure the init script is properly configured
   - Try installing packages manually in a notebook

2. **Memory Issues**:
   - Adjust Spark configuration parameters
   - Reduce the size of the dataset
   - Use more efficient data processing techniques

3. **Permission Issues**:
   - Ensure you have the necessary permissions
   - Check file paths and access rights

### Getting Help

- Databricks Documentation: https://docs.databricks.com/
- Databricks Community Forum: https://community.databricks.com/
- Spark Documentation: https://spark.apache.org/docs/3.3.2/

## Limitations of Databricks Community Edition

Databricks Community Edition has the following limitations:

1. **Cluster Size**: Limited to a single node
2. **Compute Resources**: Limited CPU and memory
3. **Storage**: Limited DBFS storage
4. **Job Scheduling**: Limited job scheduling capabilities
5. **API Access**: No API access
6. **Cluster Lifetime**: Clusters terminate after inactivity

These limitations may affect the performance and scalability of the fake news detection pipeline.
Consider using a paid Databricks tier for production deployments.

## Created on

{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""


def create_deployment_package(project_dir: str, output_dir: str = None, include_data: bool = False) -> str:
    """
    Create a deployment package for Databricks.
    
    This function creates a deployment package containing the project code,
    dependencies, and documentation.
    
    Args:
        project_dir (str): The root directory of the project.
        output_dir (str): The directory to store deployment artifacts.
        include_data (bool): Whether to include data files in the package.
        
    Returns:
        str: Path to the created deployment package.
    """
    deployment_utils = DeploymentUtils(project_dir, output_dir)
    return deployment_utils.create_deployment_package(include_data)


def create_deployment_instructions(project_dir: str, output_dir: str = None) -> str:
    """
    Create detailed deployment instructions.
    
    This function generates a markdown document with detailed instructions
    for deploying the fake news detection pipeline to Databricks.
    
    Args:
        project_dir (str): The root directory of the project.
        output_dir (str): The directory to store deployment artifacts.
        
    Returns:
        str: Path to the created instructions.
    """
    deployment_utils = DeploymentUtils(project_dir, output_dir)
    return deployment_utils.create_deployment_instructions()


def convert_py_to_notebook(py_file: str, output_path: str = None) -> str:
    """
    Convert a Python file to a Jupyter notebook.
    
    This function converts a Python file to a Jupyter notebook using jupytext.
    
    Args:
        py_file (str): Path to the Python file.
        output_path (str): Optional path to save the notebook.
        
    Returns:
        str: Path to the created notebook.
    """
    deployment_utils = DeploymentUtils("", "")
    return deployment_utils.convert_py_to_notebook(py_file, output_path)


# Example usage:
# project_dir = "/home/ubuntu/fake_news_detection"
# output_dir = "/home/ubuntu/deployment"
# 
# # Create deployment package
# package_path = create_deployment_package(project_dir, output_dir)
# print(f"Deployment package created at: {package_path}")
# 
# # Create deployment instructions
# instructions_path = create_deployment_instructions(project_dir, output_dir)
# print(f"Deployment instructions created at: {instructions_path}")
# 
# # Convert Python file to notebook
# py_file = "/home/ubuntu/fake_news_detection/02_preprocessing/text_preprocessor.py"
# notebook_path = convert_py_to_notebook(py_file)
# print(f"Notebook created at: {notebook_path}")

# Last modified: May 29, 2025
