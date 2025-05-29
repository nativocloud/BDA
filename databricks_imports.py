# Helper module for Databricks imports
import importlib.util
import sys

def import_module(module_path):
    """
    Import a module from a path that may contain numeric prefixes
    Example: import_module('BDA/01_data_ingestion/hive_data_ingestion.py')
    """
    spec = importlib.util.spec_from_file_location(
        module_path.split('/')[-1].replace('.py', ''), 
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

# Example usage in Databricks notebooks:
# from BDA.databricks_imports import import_module
# hive_ingestion = import_module('/Workspace/Repos/username/BDA/01_data_ingestion/hive_data_ingestion.py')
# HiveDataIngestion = hive_ingestion.HiveDataIngestion
