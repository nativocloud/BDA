# Databricks Setup Guide for BDA Project

This guide provides comprehensive instructions for setting up and using the BDA project in Databricks environments.

## Directory Structure

The BDA project uses a structured approach with numeric prefixes for organization:

```
BDA/
├── BDA01_data_ingestion/
│   ├── hive_data_ingestion.py
│   └── data_loader.py
├── BDA02_preprocessing/
│   ├── enhanced_text_preprocessor.py
│   ├── date_utils.py
│   └── data_validation_utils.py
└── ...
```

## Setup Instructions

### 1. Dependencies Installation

Add this to the beginning of your notebooks:

```python
# Install required packages (only needs to be run once per cluster)
%pip install nltk scikit-learn pandas numpy matplotlib seaborn

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')  # Required for some tokenization operations
nltk.download('stopwords')
nltk.download('wordnet')
```

### 2. Module Imports

Use this pattern for importing your modules:

```python
# Import modules using %run
%run "./BDA01_data_ingestion/hive_data_ingestion"
%run "./BDA02_preprocessing/enhanced_text_preprocessor"
%run "./BDA02_preprocessing/date_utils"
%run "./BDA02_preprocessing/data_validation_utils"

# Now you can use the classes directly
ingestion = HiveDataIngestion()
preprocessor = EnhancedTextPreprocessor()
validator = DateValidator()
```

### 3. For Production Use

For a more maintainable approach in production:

1. Create a `requirements.txt` file in your project root:
```
nltk>=3.6.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

2. Install at the cluster level through Databricks UI:
   - Go to Compute > Your Cluster > Libraries
   - Click "Install New" and add the required packages

## Common Issues and Solutions

### Import Path Issues

If you encounter import errors:

1. **Check the path in %run commands**: Make sure the path is correct relative to your notebook location
2. **Use absolute paths if needed**: `%run "/Users/your_username/BDA/BDA01_data_ingestion/hive_data_ingestion"`
3. **Verify file existence**: Ensure the file exists at the specified path

### Syntax Errors

If you see syntax errors like `SyntaxError: EOF while scanning triple-quoted string literal`:

1. Check for unclosed triple-quoted strings (`"""`) in your Python files
2. Ensure all docstrings and multi-line strings are properly closed

### Missing Dependencies

If you see errors like `ModuleNotFoundError: No module named 'nltk'`:

1. Run the dependency installation code at the beginning of your notebook
2. Restart your cluster if needed after installing packages
3. Check for any version conflicts between packages

### NLTK Resource Errors

If you see errors like `Resource punkt_tab not found`:

1. Make sure to download all required NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

2. If you're still having issues, try downloading all NLTK data:
```python
nltk.download('all')
```

## Best Practices

1. **Use %run for imports**: This is the most reliable method for importing modules with numeric prefixes in Databricks
2. **Install dependencies at cluster level**: For production use, install dependencies at the cluster level
3. **Use consistent paths**: Maintain consistent relative or absolute paths in your imports
4. **Document dependencies**: Keep a requirements.txt file updated with all dependencies

## Example Workflow

Here's a complete example workflow for a fake news detection pipeline:

```python
# Install dependencies
%pip install nltk scikit-learn pandas numpy matplotlib seaborn

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Import modules
%run "./BDA01_data_ingestion/hive_data_ingestion"
%run "./BDA02_preprocessing/enhanced_text_preprocessor"

# Initialize components
ingestion = HiveDataIngestion()
preprocessor = EnhancedTextPreprocessor()

# Load data
real_df, fake_df = ingestion.load_data_from_hive()
combined_df = ingestion.combine_datasets(real_df, fake_df)

# Create a balanced sample
sample_df = ingestion.create_balanced_sample(combined_df, sample_size=10000)

# Preprocess the data
preprocessed_df = preprocessor.preprocess_dataframe(sample_df)
```

This guide should help you successfully set up and use the BDA project in Databricks environments.

# Last modified: May 29, 2025
