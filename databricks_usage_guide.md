# Databricks Usage Guide for BDA Project

This guide explains how to properly import and use the BDA project modules in Databricks environments.

## Package Structure

The BDA project maintains a clear organizational structure with numeric prefixes (e.g., `01_data_ingestion`, `02_preprocessing`). While this provides excellent organization, it creates challenges for Python's import system since module names cannot start with numbers.

## Import Methods in Databricks

### Method 1: Using the Import Helper (Recommended)

We've created a special import helper that allows you to import modules with numeric prefixes:

```python
# Import the helper
from BDA.databricks_imports import import_module

# Import specific modules
hive_ingestion = import_module('/Workspace/Repos/your_username/BDA/01_data_ingestion/hive_data_ingestion.py')
text_preprocessor = import_module('/Workspace/Repos/your_username/BDA/02_preprocessing/enhanced_text_preprocessor.py')

# Use the imported classes
ingestion = hive_ingestion.HiveDataIngestion()
preprocessor = text_preprocessor.EnhancedTextPreprocessor()
```

### Method 2: Using Databricks %run Command

For simpler cases, you can use Databricks' built-in `%run` command:

```python
# Add notebook-relative imports
%run ../01_data_ingestion/hive_data_ingestion
%run ../02_preprocessing/enhanced_text_preprocessor

# Use the imported classes
ingestion = HiveDataIngestion()
preprocessor = EnhancedTextPreprocessor()
```

## Example: Setting Up a Complete Pipeline

Here's how to set up a complete fake news detection pipeline in Databricks:

```python
# Import necessary modules
from BDA.databricks_imports import import_module

# Import data ingestion
hive_ingestion = import_module('/Workspace/Repos/your_username/BDA/01_data_ingestion/hive_data_ingestion.py')
ingestion = hive_ingestion.HiveDataIngestion()

# Import preprocessing
text_preprocessor = import_module('/Workspace/Repos/your_username/BDA/02_preprocessing/enhanced_text_preprocessor.py')
preprocessor = text_preprocessor.EnhancedTextPreprocessor()

# Load and preprocess data
real_df, fake_df = ingestion.load_data_from_hive()
combined_df = ingestion.combine_datasets(real_df, fake_df)

# Create a balanced sample
sample_df = ingestion.create_balanced_sample(combined_df, sample_size=10000)

# Preprocess the data
preprocessed_df = preprocessor.preprocess_dataframe(sample_df)
```

## Community vs. Standard Edition Considerations

Both Databricks Community and Standard editions can use the same import approaches. The main difference is in resource constraints:

- **Community Edition**: Limited to 1 driver, 15.3 GB memory, 2 cores
- **Standard Edition**: More resources available, but code should be compatible with both

The import helper works identically in both environments.

## Best Practices

1. **Use the import helper** for complex imports or when you need to access multiple modules
2. **Use %run** for simple, one-off imports in notebooks
3. **Keep __init__.py files** in all directories to maintain proper package structure
4. **Document import patterns** in notebook comments for team reference
