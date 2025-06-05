# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pipeline Integrado Completo: Ingestão, Pré-processamento e Persistência
#
# Este notebook demonstra o pipeline integrado completo para detecção de fake news, incluindo:
# 1. Ingestão de dados
# 2. Pré-processamento de texto
# 3. Tokenização
# 4. Remoção de stopwords
# 5. Persistência em tabelas Hive e arquivos Parquet

# %% [markdown]
# ## Configuração e Importações

# %%
# Importar bibliotecas necessárias
import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, lower, regexp_replace, regexp_extract, trim, when
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Importar funções do pipeline integrado
from integrated_text_processing import (
    preprocess_text,
    tokenize_text,
    remove_stopwords,
    complete_text_processing
)

# Importar funções de persistência
from integrated_save_functions import (
    save_to_hive_table,
    save_to_parquet,
    complete_pipeline_with_persistence
)

# %% [markdown]
# ## Criar Sessão Spark

# %%
# Criar uma sessão Spark com configuração otimizada para Databricks Community Edition
spark = SparkSession.builder \
    .appName("FakeNewsDetection_IntegratedPipeline") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

# Exibir configuração do Spark
print(f"Spark version: {spark.version}")
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")

# %% [markdown]
# ## Criar Estrutura de Diretórios

# %%
def create_directory_structure(base_dir="/dbfs/FileStore/fake_news_detection"):
    """
    Cria a estrutura de diretórios necessária para o projeto de detecção de fake news.
    
    Args:
        base_dir (str): Diretório base para o projeto
        
    Returns:
        dict: Dicionário com caminhos para todos os diretórios criados
    """
    print(f"Criando estrutura de diretórios em {base_dir}...")
    
    # Definir caminhos de diretórios
    directories = {
        "data": f"{base_dir}/data",
        "raw_data": f"{base_dir}/data/raw",
        "processed_data": f"{base_dir}/data/processed",
        "sample_data": f"{base_dir}/data/sample",
        "models": f"{base_dir}/models",
        "logs": f"{base_dir}/logs",
        "visualizations": f"{base_dir}/visualizations",
        "temp": f"{base_dir}/temp"
    }
    
    # Criar diretórios
    for dir_name, dir_path in directories.items():
        # Usar dbutils no ambiente Databricks
        try:
            dbutils.fs.mkdirs(dir_path)
            print(f"Diretório criado: {dir_path}")
        except NameError:
            # Fallback para ambientes não-Databricks
            os.makedirs(dir_path.replace("/dbfs", ""), exist_ok=True)
            print(f"Diretório criado: {dir_path} (modo local)")
    
    print("Estrutura de diretórios criada com sucesso")
    return directories

# Criar diretórios
directories = create_directory_structure()

# %% [markdown]
# ## Carregar Dados

# %%
def load_csv_files(fake_path, true_path, cache=True):
    """
    Carrega arquivos CSV contendo notícias falsas e verdadeiras.
    
    Args:
        fake_path (str): Caminho para o arquivo CSV com notícias falsas
        true_path (str): Caminho para o arquivo CSV com notícias verdadeiras
        cache (bool): Se deve usar cache durante o carregamento
        
    Returns:
        tuple: (fake_df, true_df) DataFrames com dados carregados
    """
    print(f"Carregando arquivos CSV de {fake_path} e {true_path}...")
    
    # Carregar arquivos CSV
    fake_df = spark.read.csv(fake_path, header=True, inferSchema=True)
    true_df = spark.read.csv(true_path, header=True, inferSchema=True)
    
    # Adicionar rótulos (0 para falso, 1 para verdadeiro)
    fake_df = fake_df.withColumn("label", lit(0))
    true_df = true_df.withColumn("label", lit(1))
    
    # Cache DataFrames se solicitado
    if cache:
        fake_df.cache()
        true_df.cache()
        # Forçar materialização
        fake_count = fake_df.count()
        true_count = true_df.count()
    
    # Mostrar informações sobre os DataFrames
    print(f"Notícias falsas carregadas: {fake_df.count()} registros")
    print(f"Notícias verdadeiras carregadas: {true_df.count()} registros")
    
    return fake_df, true_df

# %% [markdown]
# ## Combinar Datasets

# %%
def combine_datasets(fake_df, true_df, cache=True):
    """
    Combina datasets de notícias falsas e verdadeiras em um único DataFrame.
    
    Args:
        fake_df: DataFrame com notícias falsas
        true_df: DataFrame com notícias verdadeiras
        cache (bool): Se deve usar cache durante a combinação
        
    Returns:
        DataFrame: DataFrame combinado com notícias falsas e verdadeiras
    """
    print("Combinando datasets de notícias falsas e verdadeiras...")
    
    # Combinar datasets
    combined_df = fake_df.union(true_df)
    
    # Cache o DataFrame combinado se solicitado
    if cache:
        combined_df.cache()
        # Forçar materialização
        combined_count = combined_df.count()
    
    print(f"Dataset combinado criado com {combined_df.count()} registros")
    
    return combined_df

# %% [markdown]
# ## Exemplo 1: Pipeline Passo a Passo
# 
# Este exemplo mostra como executar o pipeline passo a passo, com controle explícito sobre cada etapa.

# %%
# Definir caminhos (atualize com seus caminhos reais)
fake_path = "/path/to/Fake.csv"
true_path = "/path/to/True.csv"

# Etapa 1: Carregar dados
# fake_df, true_df = load_csv_files(fake_path, true_path, cache=True)

# Etapa 2: Combinar datasets
# combined_df = combine_datasets(fake_df, true_df, cache=True)

# Etapa 3: Pré-processamento de texto
# preprocessed_df = preprocess_text(combined_df, cache=True)

# Etapa 4: Tokenização
# tokenized_df = tokenize_text(preprocessed_df, text_column="text", output_column="tokens")

# Etapa 5: Remoção de stopwords
# processed_df = remove_stopwords(tokenized_df, tokens_column="tokens", output_column="filtered_tokens")

# Etapa 6: Salvar em tabela Hive
# save_to_hive_table(processed_df, "processed_news", partition_by="label")

# Etapa 7: Salvar em Parquet
# parquet_path = f"{directories['processed_data']}/processed_news.parquet"
# save_to_parquet(processed_df, parquet_path, partition_by="label")

# Etapa 8: Liberar memória
# fake_df.unpersist()
# true_df.unpersist()
# combined_df.unpersist()
# preprocessed_df.unpersist()

# %% [markdown]
# ## Exemplo 2: Pipeline Integrado com Processamento de Texto
# 
# Este exemplo usa a função `complete_text_processing` para executar o pré-processamento, tokenização e remoção de stopwords em uma única chamada.

# %%
# Definir caminhos (atualize com seus caminhos reais)
fake_path = "/path/to/Fake.csv"
true_path = "/path/to/True.csv"

# Etapa 1: Carregar dados
# fake_df, true_df = load_csv_files(fake_path, true_path, cache=True)

# Etapa 2: Combinar datasets
# combined_df = combine_datasets(fake_df, true_df, cache=True)

# Etapa 3: Processamento completo de texto (pré-processamento, tokenização, remoção de stopwords)
# processed_df = complete_text_processing(combined_df, cache=True)

# Etapa 4: Salvar em tabela Hive
# save_to_hive_table(processed_df, "processed_news", partition_by="label")

# Etapa 5: Salvar em Parquet
# parquet_path = f"{directories['processed_data']}/processed_news.parquet"
# save_to_parquet(processed_df, parquet_path, partition_by="label")

# Etapa 6: Liberar memória
# fake_df.unpersist()
# true_df.unpersist()
# combined_df.unpersist()

# %% [markdown]
# ## Exemplo 3: Pipeline Completo com Persistência
# 
# Este exemplo usa a função `complete_pipeline_with_persistence` para executar todo o pipeline em uma única chamada, incluindo persistência em Hive e Parquet.

# %%
# Definir caminhos (atualize com seus caminhos reais)
fake_path = "/path/to/Fake.csv"
true_path = "/path/to/True.csv"

# Executar o pipeline completo com persistência
# processed_df = complete_pipeline_with_persistence(fake_path, true_path, directories, cache=True)

# %% [markdown]
# ## Examinar os Resultados

# %%
# Exibir schema
# processed_df.printSchema()

# Mostrar dados de exemplo
# display(processed_df.select("text", "tokens", "filtered_tokens", "label", "location", "news_source").limit(5))

# Contar registros por rótulo
# display(processed_df.groupBy("label").count().orderBy("label"))

# %% [markdown]
# ## Abordagem com Pipeline API
# 
# Uma abordagem alternativa é usar a API Pipeline do Spark ML:

# %%
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF

def create_pipeline_api_approach(include_features=True):
    """
    Cria um pipeline de processamento de texto usando a API Pipeline do Spark ML.
    
    Args:
        include_features (bool): Se deve incluir etapas de extração de características
        
    Returns:
        Pipeline: Pipeline Spark ML para processamento de texto
    """
    # Definir transformadores
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    
    # Criar estágios do pipeline
    stages = [tokenizer, remover]
    
    # Opcionalmente adicionar extração de características
    if include_features:
        hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        stages.extend([hashingTF, idf])
    
    # Criar e retornar o pipeline
    return Pipeline(stages=stages)

# Exemplo de uso (comentado)
# pipeline = create_pipeline_api_approach(include_features=True)
# model = pipeline.fit(preprocessed_df)
# processed_df = model.transform(preprocessed_df)

# %% [markdown]
# ## Conclusão
# 
# Este notebook demonstra o pipeline integrado completo para detecção de fake news, combinando ingestão de dados, pré-processamento, tokenização, remoção de stopwords e persistência em uma única fase.
# 
# Benefícios desta abordagem:
# 1. Redução de computação ao eliminar processamento redundante
# 2. Melhoria na eficiência de memória através de cache estratégico
# 3. Fluxo de trabalho simplificado com menos etapas
# 4. Desempenho aprimorado em ambientes com recursos limitados como Databricks Community Edition
# 5. Persistência eficiente em múltiplos formatos (Hive e Parquet)
# 
# Os dados processados estão agora prontos para engenharia de características e treinamento de modelos.
