# %% [markdown]
# # Fake News Detection: Deployment
# 
# This notebook contains all the necessary code for deploying the fake news detection project to Databricks Community Edition. The code is organized into independent functions, without dependencies on external modules or classes, to facilitate execution in Databricks Community Edition.

# %% [markdown]
# ## Setup and Imports

# %%
# Import necessary libraries
import os
import sys
import shutil
import json
import zipfile
import subprocess
from pathlib import Path

# %% [markdown]
# ## Reusable Functions

# %% [markdown]
# ### Configuration Functions

# %%
def create_deployment_config(include_dirs=None, exclude_dirs=None, include_files=None, 
                            exclude_files=None, databricks_workspace=None):
    """
    Create a configuration dictionary for deployment.
    
    Args:
        include_dirs (list): Directories to include in deployment
        exclude_dirs (list): Directories to exclude from deployment
        include_files (list): Specific files to include
        exclude_files (list): Specific files to exclude
        databricks_workspace (str): Databricks workspace path
        
    Returns:
        dict: Configuration dictionary
    """
    # Default configuration
    config = {
        'include_dirs': include_dirs if include_dirs is not None else [],
        'exclude_dirs': exclude_dirs if exclude_dirs is not None else ['logs', '.git', '__pycache__', '.ipynb_checkpoints'],
        'include_files': include_files if include_files is not None else [],
        'exclude_files': exclude_files if exclude_files is not None else ['.DS_Store', '*.pyc', '*.pyo', '*.pyd', '.Python', '*.so'],
        'databricks_workspace': databricks_workspace if databricks_workspace is not None else '/Shared/fake_news_detection'
    }
    
    return config

# %%
def get_project_root(current_file=None):
    """
    Get the root directory of the project.
    
    Args:
        current_file (str): Path to the current file
        
    Returns:
        str: Absolute path to the project root directory
    """
    if current_file is None:
        # Use the current working directory
        current_path = os.getcwd()
    else:
        # Use the directory of the current file
        current_path = os.path.dirname(os.path.abspath(current_file))
    
    # Find the project root (assuming it's the parent of the current directory)
    project_root = os.path.abspath(os.path.join(current_path, '..'))
    
    # Check if we're already at the project root
    if os.path.basename(current_path) == 'BDA':
        project_root = current_path
    
    print(f"Project root identified as: {project_root}")
    return project_root

# %% [markdown]
# ### Package Creation Functions

# %%
def should_include_dir(dirname, config):
    """
    Check if a directory should be included in the deployment package.
    
    Args:
        dirname (str): Directory name to check
        config (dict): Deployment configuration
        
    Returns:
        bool: True if the directory should be included, False otherwise
    """
    # Check if explicitly included
    if config['include_dirs'] and dirname in config['include_dirs']:
        return True
    
    # Check if explicitly excluded
    if dirname in config['exclude_dirs']:
        return False
    
    # If include_dirs is empty, include all directories not explicitly excluded
    return not config['include_dirs']

# %%
def should_include_file(filename, config):
    """
    Check if a file should be included in the deployment package.
    
    Args:
        filename (str): File name to check
        config (dict): Deployment configuration
        
    Returns:
        bool: True if the file should be included, False otherwise
    """
    # Check if explicitly included
    if config['include_files'] and any(
        filename.endswith(pattern[1:]) if pattern.startswith('*.') else filename == pattern
        for pattern in config['include_files']
    ):
        return True
    
    # Check if explicitly excluded
    if any(
        filename.endswith(pattern[1:]) if pattern.startswith('*.') else filename == pattern
        for pattern in config['exclude_files']
    ):
        return False
    
    # If include_files is empty, include all files not explicitly excluded
    return not config['include_files']

# %%
def create_deployment_package(project_root, config, output_path=None):
    """
    Create a deployment package (zip file) containing all necessary files.
    
    Args:
        project_root (str): Root directory of the project
        config (dict): Deployment configuration
        output_path (str): Path to save the deployment package
            
    Returns:
        str: Path to the created deployment package
    """
    if output_path is None:
        output_path = os.path.join(project_root, 'deployment_package.zip')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    print(f"Creating deployment package from {project_root}...")
    print(f"Output path: {output_path}")
    
    # Create zip file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through project directory
        for root, dirs, files in os.walk(project_root):
            # Filter directories
            dirs[:] = [d for d in dirs if should_include_dir(d, config)]
            
            # Get relative path
            rel_path = os.path.relpath(root, project_root)
            if rel_path == '.':
                rel_path = ''
            
            # Add files
            for file in files:
                if should_include_file(file, config):
                    file_path = os.path.join(root, file)
                    arcname = os.path.join(rel_path, file)
                    zipf.write(file_path, arcname)
                    print(f"Added: {arcname}")
    
    print(f"Deployment package created at: {output_path}")
    return output_path

# %% [markdown]
# ### Databricks Integration Functions

# %%
def generate_databricks_init_script(project_root, output_path=None):
    """
    Generate an initialization script for Databricks.
    
    Args:
        project_root (str): Root directory of the project
        output_path (str): Path to save the initialization script
            
    Returns:
        str: Path to the created initialization script
    """
    if output_path is None:
        output_path = os.path.join(project_root, 'databricks_init.sh')
    
    # Create initialization script
    script_content = """#!/bin/bash
# Databricks initialization script for Fake News Detection project

# Install required packages
pip install --upgrade pip
pip install nltk scikit-learn pandas numpy matplotlib seaborn plotly

# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:/dbfs/FileStore/fake_news_detection

# Create necessary directories
mkdir -p /dbfs/FileStore/fake_news_detection/logs
mkdir -p /dbfs/FileStore/fake_news_detection/models
mkdir -p /dbfs/FileStore/fake_news_detection/data

echo "Initialization complete"
"""
    
    # Write script to file
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(output_path, 0o755)
    
    print(f"Databricks initialization script created at: {output_path}")
    return output_path

# %%
def generate_deployment_instructions(project_root, output_path=None):
    """
    Generate deployment instructions for Databricks Community Edition.
    
    Args:
        project_root (str): Root directory of the project
        output_path (str): Path to save the instructions
            
    Returns:
        str: Path to the created instructions file
    """
    if output_path is None:
        output_path = os.path.join(project_root, 'deployment_instructions.md')
    
    # Create instructions
    instructions = """# Deployment Instructions for Fake News Detection Project

## Prerequisites
- Databricks Community Edition account
- Web browser

## Step 1: Upload Deployment Package
1. Log in to your Databricks Community Edition account
2. Navigate to the Data tab in the left sidebar
3. Click on "Create" > "File Upload"
4. Select the deployment package zip file (`deployment_package.zip`)
5. Upload to `/FileStore/fake_news_detection/`

## Step 2: Extract the Package
1. Create a new notebook
2. Add the following code to extract the package:

```python
# Extract deployment package
import zipfile
zip_path = "/dbfs/FileStore/fake_news_detection/deployment_package.zip"
extract_path = "/dbfs/FileStore/fake_news_detection/"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Extraction complete")
```

3. Run the cell to extract the package

## Step 3: Upload Data Files
1. Navigate to the Data tab
2. Upload the following data files to `/FileStore/fake_news_detection/data/`:
   - `True.csv`
   - `Fake.csv`
   - `stream1.csv` (if using streaming pipeline)

## Step 4: Set Up Environment
1. Create a new cluster with the following configuration:
   - Databricks Runtime: 11.3 LTS (includes Apache Spark 3.3.0, Scala 2.12)
   - Python Version: 3.9
   - Worker Type: Standard_DS3_v2
   - Driver Type: Standard_DS3_v2
   - Workers: 1-2 (Community Edition limitation)

2. Create a new notebook and attach it to the cluster
3. Run the initialization script:

```python
# Run initialization script
dbutils.fs.cp("file:/dbfs/FileStore/fake_news_detection/databricks_init.sh", "dbfs:/databricks/init/fake_news_init.sh")
```

4. Restart the cluster to apply the initialization script

## Step 5: Run the Pipeline
1. Navigate to the Workspace tab
2. Create a new folder called "Fake News Detection"
3. Import the notebooks from the deployment package:
   - 01_data_ingestion_standalone.ipynb
   - 02_preprocessing_standalone.ipynb
   - 03_feature_engineering_standalone.ipynb
   - 04_traditional_models_standalone.ipynb
   - 05_graph_analysis_standalone.ipynb
   - 06_clustering_standalone.ipynb
   - 07_streaming_standalone.ipynb
   - 08_visualization_standalone.ipynb

4. Run the notebooks in sequence to execute the pipeline

## Step 6: Monitor and Visualize Results
1. The pipeline will generate results in `/FileStore/fake_news_detection/logs/`
2. Visualization dashboards will be available in the notebooks
3. For Grafana integration, follow the instructions in the visualization notebook

## Troubleshooting
- If you encounter package import errors, ensure the initialization script ran successfully
- For memory issues, try reducing the sample size in the data ingestion notebook
- Check the logs for detailed error messages

## Community Edition Limitations
- Limited compute resources (max 15GB memory)
- No API access
- Limited cluster size
- No MLflow integration
- No job scheduler

## Workarounds for Limitations
- Use smaller data samples
- Implement simplified models
- Use file-based metrics export instead of direct Grafana integration
- Run notebooks manually instead of using job scheduler
"""
    
    # Write instructions to file
    with open(output_path, 'w') as f:
        f.write(instructions)
    
    print(f"Deployment instructions created at: {output_path}")
    return output_path

# %%
def create_databricks_notebook_config(project_root, output_path=None):
    """
    Create configuration for Databricks notebooks.
    
    Args:
        project_root (str): Root directory of the project
        output_path (str): Path to save the configuration
            
    Returns:
        str: Path to the created configuration file
    """
    if output_path is None:
        output_path = os.path.join(project_root, 'notebook_config.json')
    
    # Create configuration
    config = {
        "notebooks": [
            {
                "path": "01_data_ingestion_standalone.ipynb",
                "description": "Data loading and preparation",
                "depends_on": []
            },
            {
                "path": "02_preprocessing_standalone.ipynb",
                "description": "Text preprocessing and cleaning",
                "depends_on": ["01_data_ingestion_standalone.ipynb"]
            },
            {
                "path": "03_feature_engineering_standalone.ipynb",
                "description": "Feature extraction and engineering",
                "depends_on": ["02_preprocessing_standalone.ipynb"]
            },
            {
                "path": "04_traditional_models_standalone.ipynb",
                "description": "Traditional machine learning models",
                "depends_on": ["03_feature_engineering_standalone.ipynb"]
            },
            {
                "path": "05_graph_analysis_standalone.ipynb",
                "description": "Graph-based analysis",
                "depends_on": ["03_feature_engineering_standalone.ipynb"]
            },
            {
                "path": "06_clustering_standalone.ipynb",
                "description": "Clustering analysis",
                "depends_on": ["03_feature_engineering_standalone.ipynb"]
            },
            {
                "path": "07_streaming_standalone.ipynb",
                "description": "Streaming analysis",
                "depends_on": ["04_traditional_models_standalone.ipynb"]
            },
            {
                "path": "08_visualization_standalone.ipynb",
                "description": "Visualization and dashboards",
                "depends_on": ["04_traditional_models_standalone.ipynb", "05_graph_analysis_standalone.ipynb", 
                              "06_clustering_standalone.ipynb", "07_streaming_standalone.ipynb"]
            }
        ],
        "data": {
            "input": [
                {
                    "path": "/FileStore/fake_news_detection/data/True.csv",
                    "description": "True news articles"
                },
                {
                    "path": "/FileStore/fake_news_detection/data/Fake.csv",
                    "description": "Fake news articles"
                }
            ],
            "output": [
                {
                    "path": "/FileStore/fake_news_detection/models/",
                    "description": "Trained models"
                },
                {
                    "path": "/FileStore/fake_news_detection/logs/",
                    "description": "Logs and metrics"
                }
            ]
        },
        "cluster": {
            "runtime": "11.3 LTS",
            "python_version": "3.9",
            "worker_type": "Standard_DS3_v2",
            "driver_type": "Standard_DS3_v2",
            "workers": 2
        }
    }
    
    # Write configuration to file
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Databricks notebook configuration created at: {output_path}")
    return output_path

# %% [markdown]
# ### Deployment Validation Functions

# %%
def validate_deployment_package(package_path):
    """
    Validate a deployment package by checking its contents.
    
    Args:
        package_path (str): Path to the deployment package
            
    Returns:
        bool: True if the package is valid, False otherwise
    """
    print(f"Validating deployment package: {package_path}")
    
    # Check if package exists
    if not os.path.exists(package_path):
        print(f"Error: Package not found at {package_path}")
        return False
    
    # Check if package is a zip file
    if not zipfile.is_zipfile(package_path):
        print(f"Error: {package_path} is not a valid zip file")
        return False
    
    # Check package contents
    required_files = [
        'databricks_init.sh',
        'deployment_instructions.md',
        'notebook_config.json'
    ]
    
    required_dirs = [
        '01_data_ingestion',
        '02_preprocessing',
        '03_feature_engineering',
        '04_modeling',
        '05_graph_analysis',
        '06_clustering',
        '07_streaming',
        '08_visualization'
    ]
    
    # Extract file list from zip
    with zipfile.ZipFile(package_path, 'r') as zipf:
        file_list = zipf.namelist()
    
    # Check required files
    missing_files = []
    for req_file in required_files:
        if not any(f.endswith(req_file) for f in file_list):
            missing_files.append(req_file)
    
    # Check required directories
    missing_dirs = []
    for req_dir in required_dirs:
        if not any(f.startswith(req_dir + '/') for f in file_list):
            missing_dirs.append(req_dir)
    
    # Report validation results
    if missing_files:
        print(f"Warning: Missing required files: {', '.join(missing_files)}")
    
    if missing_dirs:
        print(f"Warning: Missing required directories: {', '.join(missing_dirs)}")
    
    is_valid = not (missing_files or missing_dirs)
    
    if is_valid:
        print("Validation successful: Deployment package contains all required files and directories")
    else:
        print("Validation failed: Deployment package is missing required files or directories")
    
    return is_valid

# %%
def check_databricks_compatibility(project_root):
    """
    Check if the project is compatible with Databricks Community Edition.
    
    Args:
        project_root (str): Root directory of the project
            
    Returns:
        dict: Dictionary with compatibility check results
    """
    print("Checking Databricks Community Edition compatibility...")
    
    compatibility_issues = []
    
    # Check for large files (>100MB)
    large_files = []
    for root, dirs, files in os.walk(project_root):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
                large_files.append(os.path.relpath(file_path, project_root))
    
    if large_files:
        compatibility_issues.append(f"Large files detected (>100MB): {', '.join(large_files)}")
    
    # Check for unsupported libraries
    unsupported_libraries = []
    requirements_path = os.path.join(project_root, 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            requirements = f.read().splitlines()
        
        # List of libraries known to be problematic in Databricks CE
        problematic_libs = ['tensorflow', 'pytorch', 'torch', 'keras', 'ray', 'dask']
        
        for lib in problematic_libs:
            if any(req.startswith(lib) for req in requirements):
                unsupported_libraries.append(lib)
    
    if unsupported_libraries:
        compatibility_issues.append(f"Potentially problematic libraries detected: {', '.join(unsupported_libraries)}")
    
    # Check for GPU dependencies
    gpu_dependencies = False
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py') or file.endswith('.ipynb'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if 'cuda' in content.lower() or 'gpu' in content.lower():
                            gpu_dependencies = True
                            break
                except:
                    pass
    
    if gpu_dependencies:
        compatibility_issues.append("GPU dependencies detected, which are not supported in Databricks Community Edition")
    
    # Prepare result
    result = {
        "compatible": len(compatibility_issues) == 0,
        "issues": compatibility_issues
    }
    
    # Print results
    if result["compatible"]:
        print("Project is compatible with Databricks Community Edition")
    else:
        print("Project may have compatibility issues with Databricks Community Edition:")
        for issue in result["issues"]:
            print(f"- {issue}")
    
    return result

# %% [markdown]
# ## Complete Deployment Pipeline

# %%
def run_deployment_pipeline(project_root=None, output_dir=None, include_dirs=None, exclude_dirs=None):
    """
    Run the complete deployment pipeline for the fake news detection project.
    
    Args:
        project_root (str): Root directory of the project
        output_dir (str): Directory to save deployment artifacts
        include_dirs (list): Directories to include in deployment
        exclude_dirs (list): Directories to exclude from deployment
        
    Returns:
        dict: Dictionary with references to deployment artifacts
    """
    print("Starting deployment pipeline...")
    
    # 1. Set up configuration
    if project_root is None:
        project_root = get_project_root()
    
    if output_dir is None:
        output_dir = os.path.join(project_root, 'deployment')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create deployment configuration
    config = create_deployment_config(
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
        databricks_workspace='/Shared/fake_news_detection'
    )
    
    # 2. Check Databricks compatibility
    compatibility = check_databricks_compatibility(project_root)
    
    if not compatibility["compatible"]:
        print("Warning: Project may have compatibility issues with Databricks Community Edition")
        print("Continuing with deployment process...")
    
    # 3. Generate deployment artifacts
    
    # Create deployment package
    package_path = create_deployment_package(
        project_root,
        config,
        output_path=os.path.join(output_dir, 'deployment_package.zip')
    )
    
    # Generate Databricks initialization script
    init_script_path = generate_databricks_init_script(
        project_root,
        output_path=os.path.join(output_dir, 'databricks_init.sh')
    )
    
    # Generate deployment instructions
    instructions_path = generate_deployment_instructions(
        project_root,
        output_path=os.path.join(output_dir, 'deployment_instructions.md')
    )
    
    # Create notebook configuration
    notebook_config_path = create_databricks_notebook_config(
        project_root,
        output_path=os.path.join(output_dir, 'notebook_config.json')
    )
    
    # 4. Validate deployment package
    is_valid = validate_deployment_package(package_path)
    
    if not is_valid:
        print("Warning: Deployment package validation failed")
        print("Review the validation results and fix any issues before deploying")
    
    # 5. Prepare result
    result = {
        "package_path": package_path,
        "init_script_path": init_script_path,
        "instructions_path": instructions_path,
        "notebook_config_path": notebook_config_path,
        "is_valid": is_valid,
        "compatibility": compatibility
    }
    
    print(f"Deployment pipeline completed!")
    print(f"Deployment artifacts saved to: {output_dir}")
    
    return result

# %% [markdown]
# ## Step-by-Step Tutorial

# %% [markdown]
# ### 1. Set Up Deployment Environment

# %%
# Define project root and output directory
project_root = get_project_root()
output_dir = os.path.join(project_root, 'deployment')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Create deployment configuration
config = create_deployment_config(
    exclude_dirs=['logs', '.git', '__pycache__', '.ipynb_checkpoints', 'venv'],
    databricks_workspace='/Shared/fake_news_detection'
)

print(f"Deployment environment set up with output directory: {output_dir}")
print(f"Configuration: {config}")

# %% [markdown]
# ### 2. Check Databricks Compatibility

# %%
# Check if the project is compatible with Databricks Community Edition
compatibility = check_databricks_compatibility(project_root)

if compatibility["compatible"]:
    print("✅ Project is compatible with Databricks Community Edition")
else:
    print("⚠️ Project may have compatibility issues with Databricks Community Edition:")
    for issue in compatibility["issues"]:
        print(f"  - {issue}")

# %% [markdown]
# ### 3. Create Deployment Package

# %%
# Create deployment package
package_path = create_deployment_package(
    project_root,
    config,
    output_path=os.path.join(output_dir, 'deployment_package.zip')
)

# %% [markdown]
# ### 4. Generate Databricks Initialization Script

# %%
# Generate Databricks initialization script
init_script_path = generate_databricks_init_script(
    project_root,
    output_path=os.path.join(output_dir, 'databricks_init.sh')
)

# %% [markdown]
# ### 5. Generate Deployment Instructions

# %%
# Generate deployment instructions
instructions_path = generate_deployment_instructions(
    project_root,
    output_path=os.path.join(output_dir, 'deployment_instructions.md')
)

# %% [markdown]
# ### 6. Create Notebook Configuration

# %%
# Create notebook configuration
notebook_config_path = create_databricks_notebook_config(
    project_root,
    output_path=os.path.join(output_dir, 'notebook_config.json')
)

# %% [markdown]
# ### 7. Validate Deployment Package

# %%
# Validate deployment package
is_valid = validate_deployment_package(package_path)

if is_valid:
    print("✅ Deployment package validation successful")
else:
    print("⚠️ Deployment package validation failed")
    print("   Review the validation results and fix any issues before deploying")

# %% [markdown]
# ### 8. Run Complete Deployment Pipeline

# %%
# Run the complete deployment pipeline
result = run_deployment_pipeline(
    project_root=project_root,
    output_dir=output_dir
)

# Print deployment results
print("\nDeployment Results:")
print(f"Package: {result['package_path']}")
print(f"Init Script: {result['init_script_path']}")
print(f"Instructions: {result['instructions_path']}")
print(f"Notebook Config: {result['notebook_config_path']}")
print(f"Valid: {'Yes' if result['is_valid'] else 'No'}")
print(f"Compatible: {'Yes' if result['compatibility']['compatible'] else 'No'}")

# %% [markdown]
# ## Important Notes
# 
# 1. **Databricks Community Edition Limitations**: The Community Edition has several limitations:
#    - Limited compute resources (max 15GB memory)
#    - No API access
#    - Limited cluster size
#    - No MLflow integration
#    - No job scheduler
# 
# 2. **Deployment Package**: The deployment package contains all necessary files for running the fake news detection pipeline in Databricks. It includes:
#    - Standalone notebooks for each phase
#    - Initialization script for setting up the environment
#    - Configuration files for notebooks and clusters
#    - Deployment instructions
# 
# 3. **Standalone Notebooks**: The deployment uses standalone notebooks that don't rely on external modules or classes. This ensures compatibility with Databricks Community Edition.
# 
# 4. **Data Upload**: You'll need to manually upload the data files to Databricks FileStore. The deployment instructions include steps for this process.
# 
# 5. **Environment Setup**: The initialization script installs required packages and sets up the environment. You'll need to run this script when creating a new cluster.
# 
# 6. **Execution Order**: The notebooks should be run in the specified order to ensure proper data flow through the pipeline.
# 
# 7. **Troubleshooting**: If you encounter issues, check the logs and review the deployment instructions for troubleshooting tips.
# 
# 8. **Performance Optimization**: For better performance in Databricks Community Edition:
#    - Use smaller data samples
#    - Implement simplified models
#    - Optimize Spark configurations
#    - Use checkpointing to save intermediate results
