"""
Funções de salvamento para o pipeline integrado de processamento de texto.
Inclui funções para salvar em tabelas Hive e arquivos Parquet.
"""

from pyspark.sql import DataFrame
import os

def save_to_hive_table(df, table_name, partition_by=None):
    """
    Salva um DataFrame em uma tabela Hive de forma segura, verificando se a tabela já existe.
    
    Args:
        df (DataFrame): DataFrame a ser salvo
        table_name (str): Nome da tabela Hive
        partition_by (str, opcional): Coluna para particionamento
        
    Returns:
        bool: True se o salvamento foi bem-sucedido
    """
    print(f"Safely saving DataFrame to Hive table: {table_name}")
    
    # Verificar se a localização da tabela existe
    location = f"dbfs:/user/hive/warehouse/{table_name}"
    print(f"Checking if location exists: {location}")
    
    try:
        # Verificar se a localização existe usando dbutils (ambiente Databricks)
        location_exists = False
        try:
            location_exists = dbutils.fs.ls(location) is not None
        except:
            location_exists = False
        
        if location_exists:
            print(f"Location exists. Removing directory: {location}")
            dbutils.fs.rm(location, True)
            print("Directory removed successfully.")
        
        # Salvar o DataFrame como tabela Hive
        print(f"Saving DataFrame to table '{table_name}'...")
        
        if partition_by:
            df.write.mode("overwrite").partitionBy(partition_by).saveAsTable(table_name)
            print(f"DataFrame saved to table '{table_name}' with partitioning on '{partition_by}'.")
        else:
            df.write.mode("overwrite").saveAsTable(table_name)
            print(f"DataFrame saved to table '{table_name}'.")
        
        # Verificar se a tabela foi criada
        if spark.catalog.tableExists(table_name):
            print(f"Verified: Table '{table_name}' exists.")
            
            # Mostrar informações da tabela
            print("\nTable information:")
            spark.sql(f"DESCRIBE {table_name}").show(truncate=False)
            
            return True
        else:
            print(f"Error: Table '{table_name}' was not created.")
            return False
            
    except Exception as e:
        print(f"Error saving to Hive table: {str(e)}")
        return False

def save_to_parquet(df, path, partition_by=None):
    """
    Salva um DataFrame em formato Parquet.
    
    Args:
        df (DataFrame): DataFrame a ser salvo
        path (str): Caminho para salvar o arquivo Parquet
        partition_by (str, opcional): Coluna para particionamento
        
    Returns:
        bool: True se o salvamento foi bem-sucedido
    """
    print(f"Saving DataFrame to {path}...")
    
    try:
        # Salvar o DataFrame como Parquet
        if partition_by:
            df.write.mode("overwrite").partitionBy(partition_by).parquet(path)
        else:
            df.write.mode("overwrite").parquet(path)
        
        print(f"DataFrame saved to {path}")
        return True
        
    except Exception as e:
        print(f"Error saving to Parquet: {str(e)}")
        return False

def complete_pipeline_with_persistence(fake_path, true_path, directories, cache=True):
    """
    Executa o pipeline completo de processamento de texto e persistência de dados.
    
    Args:
        fake_path (str): Caminho para o arquivo CSV com notícias falsas
        true_path (str): Caminho para o arquivo CSV com notícias verdadeiras
        directories (dict): Dicionário com os diretórios do projeto
        cache (bool): Se deve usar cache durante o processamento
        
    Returns:
        DataFrame: DataFrame processado
    """
    from integrated_text_processing import load_csv_files, combine_datasets, complete_text_processing
    
    print("Starting complete pipeline with persistence...")
    
    # Etapa 1: Carregar dados
    fake_df, true_df = load_csv_files(fake_path, true_path, cache=cache)
    
    # Etapa 2: Combinar datasets
    combined_df = combine_datasets(fake_df, true_df, cache=cache)
    
    # Etapa 3: Processamento completo de texto (pré-processamento, tokenização, remoção de stopwords)
    processed_df = complete_text_processing(combined_df, cache=cache)
    
    # Etapa 4: Salvar em tabela Hive
    save_to_hive_table(processed_df, "processed_news", partition_by="label")
    
    # Etapa 5: Salvar em Parquet
    parquet_path = f"{directories['processed_data']}/processed_news.parquet"
    save_to_parquet(processed_df, parquet_path, partition_by="label")
    
    # Etapa 6: Liberar memória
    if cache:
        print("Cleaning up memory...")
        try:
            fake_df.unpersist()
            true_df.unpersist()
            combined_df.unpersist()
            print("Memory cleanup complete.")
        except:
            print("Note: Could not unpersist some DataFrames.")
    
    print("Complete pipeline with persistence finished successfully.")
    return processed_df
