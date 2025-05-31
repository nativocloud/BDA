# %% [markdown]
# # Fake News Detection: Data Ingestion
# 
# Este notebook contém todo o código necessário para carregar, processar e preparar os dados para o projeto de deteção de notícias falsas. O código está organizado em funções independentes, sem dependências de módulos externos ou classes, para facilitar a execução no Databricks Community Edition.

# %% [markdown]
# ## Setup and Imports

# %%
# Importar bibliotecas necessárias
import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, lower, regexp_replace, rand, when, concat
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Inicializar a sessão Spark com suporte ao Hive
spark = SparkSession.builder \
    .appName("FakeNewsDetection") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "8g") \
    .enableHiveSupport() \
    .getOrCreate()

# Mostrar a versão do Spark
print(f"Spark version: {spark.version}")
print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")

# %% [markdown]
# ## Funções Reutilizáveis

# %% [markdown]
# ### Funções de Carregamento de Dados

# %%
def load_csv_files(fake_path, true_path):
    """
    Carrega os ficheiros CSV de notícias falsas e verdadeiras.
    
    Args:
        fake_path (str): Caminho para o ficheiro CSV de notícias falsas
        true_path (str): Caminho para o ficheiro CSV de notícias verdadeiras
        
    Returns:
        tuple: (fake_df, true_df) DataFrames com os dados carregados
    """
    print(f"A carregar ficheiros CSV de {fake_path} e {true_path}...")
    
    # Carregar ficheiros CSV
    fake_df = spark.read.csv(fake_path, header=True, inferSchema=True)
    true_df = spark.read.csv(true_path, header=True, inferSchema=True)
    
    # Adicionar etiquetas (0 para falsas, 1 para verdadeiras)
    fake_df = fake_df.withColumn("label", lit(0))
    true_df = true_df.withColumn("label", lit(1))
    
    # Mostrar informações sobre os DataFrames
    print(f"Notícias falsas carregadas: {fake_df.count()} registos")
    print(f"Notícias verdadeiras carregadas: {true_df.count()} registos")
    
    return fake_df, true_df

# %%
def create_hive_tables(fake_df, true_df, fake_table_name="fake", true_table_name="real"):
    """
    Cria tabelas Hive para os DataFrames de notícias falsas e verdadeiras.
    
    Args:
        fake_df: DataFrame com notícias falsas
        true_df: DataFrame com notícias verdadeiras
        fake_table_name (str): Nome da tabela Hive para notícias falsas
        true_table_name (str): Nome da tabela Hive para notícias verdadeiras
    """
    print(f"A criar tabelas Hive '{fake_table_name}' e '{true_table_name}'...")
    
    # Criar tabela para notícias falsas
    spark.sql(f"DROP TABLE IF EXISTS {fake_table_name}")
    fake_df.write.mode("overwrite").saveAsTable(fake_table_name)
    print(f"Tabela '{fake_table_name}' criada com sucesso")
    
    # Criar tabela para notícias verdadeiras
    spark.sql(f"DROP TABLE IF EXISTS {true_table_name}")
    true_df.write.mode("overwrite").saveAsTable(true_table_name)
    print(f"Tabela '{true_table_name}' criada com sucesso")
    
    # Verificar se as tabelas foram criadas corretamente
    print("\nTabelas disponíveis no catálogo:")
    spark.sql("SHOW TABLES").show()

# %%
def load_data_from_hive(fake_table_name="fake", true_table_name="real"):
    """
    Carrega dados das tabelas Hive.
    
    Args:
        fake_table_name (str): Nome da tabela Hive com notícias falsas
        true_table_name (str): Nome da tabela Hive com notícias verdadeiras
        
    Returns:
        tuple: (true_df, fake_df) DataFrames com os dados carregados
    """
    print(f"A carregar dados das tabelas Hive '{true_table_name}' e '{fake_table_name}'...")
    
    # Verificar se as tabelas existem
    tables = [row.tableName for row in spark.sql("SHOW TABLES").collect()]
    
    if true_table_name not in tables or fake_table_name not in tables:
        raise ValueError(f"As tabelas Hive '{true_table_name}' e/ou '{fake_table_name}' não existem")
    
    # Carregar dados das tabelas Hive
    true_df = spark.table(true_table_name)
    fake_df = spark.table(fake_table_name)
    
    # Registar como vistas temporárias para consultas SQL
    true_df.createOrReplaceTempView("true_news")
    fake_df.createOrReplaceTempView("fake_news")
    
    # Mostrar informações sobre os DataFrames
    print(f"Notícias verdadeiras carregadas: {true_df.count()} registos")
    print(f"Notícias falsas carregadas: {fake_df.count()} registos")
    
    return true_df, fake_df

# %% [markdown]
# ### Funções de Processamento de Dados

# %%
def combine_datasets(true_df, fake_df):
    """
    Combina os DataFrames de notícias verdadeiras e falsas.
    
    Args:
        true_df: DataFrame com notícias verdadeiras
        fake_df: DataFrame com notícias falsas
        
    Returns:
        DataFrame: DataFrame combinado
    """
    print("A combinar datasets de notícias verdadeiras e falsas...")
    
    # Verificar colunas disponíveis
    true_cols = set(true_df.columns)
    fake_cols = set(fake_df.columns)
    common_cols = true_cols.intersection(fake_cols)
    
    print(f"Colunas comuns: {common_cols}")
    
    # Selecionar colunas comuns para garantir compatibilidade
    if "title" in common_cols and "text" in common_cols:
        # Se tiver título e texto, combinar para melhor contexto
        true_df = true_df.select("title", "text", "label")
        fake_df = fake_df.select("title", "text", "label")
        
        # Combinar título e texto para melhor contexto
        true_df = true_df.withColumn("full_text", 
                                    concat(col("title"), lit(". "), col("text")))
        fake_df = fake_df.withColumn("full_text", 
                                    concat(col("title"), lit(". "), col("text")))
        
        # Selecionar colunas finais
        true_df = true_df.select("full_text", "label")
        fake_df = fake_df.select("full_text", "label")
        
        # Renomear coluna
        true_df = true_df.withColumnRenamed("full_text", "text")
        fake_df = fake_df.withColumnRenamed("full_text", "text")
    else:
        # Caso contrário, usar apenas texto e etiqueta
        true_df = true_df.select("text", "label")
        fake_df = fake_df.select("text", "label")
    
    # Combinar datasets
    combined_df = true_df.unionByName(fake_df)
    
    # Mostrar informações sobre o DataFrame combinado
    print(f"Dataset combinado: {combined_df.count()} registos")
    print(f"Distribuição de etiquetas:")
    combined_df.groupBy("label").count().show()
    
    return combined_df

# %%
def preprocess_text(df):
    """
    Pré-processa o texto, convertendo para minúsculas e removendo caracteres especiais.
    
    Args:
        df: DataFrame com coluna de texto
        
    Returns:
        DataFrame: DataFrame com texto pré-processado
    """
    print("A pré-processar texto...")
    
    # Converter para minúsculas
    df = df.withColumn("text", lower(col("text")))
    
    # Remover caracteres especiais
    df = df.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z0-9\\s]", " "))
    
    # Remover espaços múltiplos
    df = df.withColumn("text", regexp_replace(col("text"), "\\s+", " "))
    
    # Verificar se há colunas problemáticas que podem causar data leakage
    if "subject" in df.columns:
        print("\nAVISO: A remover coluna 'subject' para evitar data leakage")
        print("A coluna 'subject' discrimina perfeitamente entre notícias verdadeiras e falsas")
        print("Notícias verdadeiras: subject='politicsNews', Notícias falsas: subject='News'")
        df = df.drop("subject")
        print("Coluna 'subject' removida com sucesso")
    
    return df

# %%
def create_balanced_sample(df, sample_size=1000, seed=42):
    """
    Cria uma amostra balanceada do dataset.
    
    Args:
        df: DataFrame com dados
        sample_size (int): Tamanho da amostra para cada classe
        seed (int): Semente para reprodutibilidade
        
    Returns:
        DataFrame: Amostra balanceada
    """
    print(f"A criar amostra balanceada com {sample_size} registos por classe...")
    
    # Amostra de notícias verdadeiras (label=1)
    real_sample = df.filter(col("label") == 1) \
                    .orderBy(rand(seed=seed)) \
                    .limit(sample_size)
    
    # Amostra de notícias falsas (label=0)
    fake_sample = df.filter(col("label") == 0) \
                    .orderBy(rand(seed=seed)) \
                    .limit(sample_size)
    
    # Combinar as amostras
    sample_df = real_sample.unionByName(fake_sample)
    
    # Registar o DataFrame de amostra como uma vista temporária
    sample_df.createOrReplaceTempView("sample_news")
    
    # Mostrar estatísticas da amostra
    print("\nEstatísticas da amostra:")
    spark.sql("""
        SELECT 
            label, 
            COUNT(*) as count
        FROM sample_news
        GROUP BY label
        ORDER BY label DESC
    """).show()
    
    return sample_df

# %% [markdown]
# ### Funções de Armazenamento de Dados

# %%
def save_to_parquet(df, path, partition_by=None):
    """
    Guarda um DataFrame em formato Parquet.
    
    Args:
        df: DataFrame a guardar
        path (str): Caminho onde guardar o DataFrame
        partition_by (str): Coluna para particionar (opcional)
    """
    print(f"A guardar DataFrame em {path}...")
    
    writer = df.write.mode("overwrite")
    
    if partition_by:
        writer = writer.partitionBy(partition_by)
    
    writer.parquet(path)
    print(f"DataFrame guardado em {path}")

# %%
def save_to_hive_table(df, table_name, partition_by=None):
    """
    Guarda um DataFrame numa tabela Hive.
    
    Args:
        df: DataFrame a guardar
        table_name (str): Nome da tabela Hive a criar ou substituir
        partition_by (str): Coluna para particionar (opcional)
    """
    print(f"A guardar DataFrame na tabela Hive {table_name}...")
    
    writer = df.write.mode("overwrite").format("parquet")
    
    if partition_by:
        writer = writer.partitionBy(partition_by)
    
    writer.saveAsTable(table_name)
    print(f"DataFrame guardado na tabela Hive: {table_name}")

# %% [markdown]
# ### Funções de Análise de Dados

# %%
def analyze_dataset_characteristics(df):
    """
    Analisa características do dataset para identificar potenciais problemas.
    
    Args:
        df: DataFrame com colunas de texto e etiqueta
        
    Returns:
        dict: Dicionário com resultados da análise
    """
    print("A analisar características do dataset...")
    
    # Converter para pandas para análise mais fácil
    pandas_df = df.toPandas()
    
    # Calcular estatísticas básicas
    total_samples = len(pandas_df)
    class_distribution = pandas_df['label'].value_counts().to_dict()
    class_balance = min(class_distribution.values()) / max(class_distribution.values())
    
    # Calcular estatísticas de comprimento de texto
    pandas_df['text_length'] = pandas_df['text'].apply(len)
    avg_text_length = pandas_df['text_length'].mean()
    min_text_length = pandas_df['text_length'].min()
    max_text_length = pandas_df['text_length'].max()
    
    # Verificar textos vazios ou muito curtos
    short_texts = (pandas_df['text_length'] < 10).sum()
    
    # Verificar textos duplicados
    duplicate_texts = pandas_df['text'].duplicated().sum()
    
    # Compilar resultados
    results = {
        'total_samples': total_samples,
        'class_distribution': class_distribution,
        'class_balance': class_balance,
        'avg_text_length': avg_text_length,
        'min_text_length': min_text_length,
        'max_text_length': max_text_length,
        'short_texts': short_texts,
        'duplicate_texts': duplicate_texts
    }
    
    # Imprimir resumo
    print("Características do Dataset:")
    print(f"Total de amostras: {total_samples}")
    print(f"Distribuição de classes: {class_distribution}")
    print(f"Rácio de equilíbrio de classes: {class_balance:.2f}")
    print(f"Comprimento médio de texto: {avg_text_length:.2f} caracteres")
    print(f"Intervalo de comprimento de texto: {min_text_length} a {max_text_length} caracteres")
    print(f"Número de textos muito curtos (<10 chars): {short_texts}")
    print(f"Número de textos duplicados: {duplicate_texts}")
    
    # Criar gráficos
    plt.figure(figsize=(12, 5))
    
    # Gráfico de distribuição de classes
    plt.subplot(1, 2, 1)
    sns.countplot(x='label', data=pandas_df)
    plt.title('Distribuição de Classes')
    plt.xlabel('Classe (0=Falsa, 1=Verdadeira)')
    plt.ylabel('Contagem')
    
    # Gráfico de distribuição de comprimento de texto
    plt.subplot(1, 2, 2)
    sns.histplot(pandas_df['text_length'], bins=30)
    plt.title('Distribuição de Comprimento de Texto')
    plt.xlabel('Comprimento (caracteres)')
    plt.ylabel('Contagem')
    
    plt.tight_layout()
    plt.show()
    
    return results

# %% [markdown]
# ## Pipeline Completo de Ingestão de Dados

# %%
def process_and_save_data(fake_path="/FileStore/tables/fake.csv", 
                         true_path="/FileStore/tables/real.csv",
                         output_dir="dbfs:/FileStore/fake_news_detection/data",
                         create_tables=True):
    """
    Processa e guarda dados de notícias falsas e verdadeiras.
    
    Este pipeline completo carrega os dados CSV, combina datasets, cria amostras,
    e guarda os resultados em formato Parquet e como tabelas Hive.
    
    Args:
        fake_path (str): Caminho para o ficheiro CSV de notícias falsas
        true_path (str): Caminho para o ficheiro CSV de notícias verdadeiras
        output_dir (str): Diretório para guardar dados processados
        create_tables (bool): Se deve criar tabelas Hive
        
    Returns:
        dict: Dicionário com referências aos DataFrames processados
    """
    print("A iniciar pipeline de processamento de dados...")
    
    # 1. Carregar ficheiros CSV
    fake_df, true_df = load_csv_files(fake_path, true_path)
    
    # 2. Criar tabelas Hive (opcional)
    if create_tables:
        create_hive_tables(fake_df, true_df)
    
    # 3. Combinar datasets
    combined_df = combine_datasets(true_df, fake_df)
    
    # 4. Pré-processar texto
    combined_df = preprocess_text(combined_df)
    
    # 5. Criar amostra balanceada
    sample_df = create_balanced_sample(combined_df)
    
    # 6. Analisar características do dataset
    analyze_dataset_characteristics(combined_df)
    
    # 7. Guardar dataset combinado em DBFS
    combined_path = f"{output_dir}/combined_data/combined_news.parquet"
    save_to_parquet(combined_df, combined_path, partition_by="label")
    
    # 8. Guardar amostra em DBFS
    sample_path = f"{output_dir}/sample_data/sample_news.parquet"
    save_to_parquet(sample_df, sample_path)
    
    # 9. Guardar em tabelas Hive para acesso mais fácil
    save_to_hive_table(combined_df, "combined_news", partition_by="label")
    save_to_hive_table(sample_df, "sample_news")
    
    print("\nPipeline de processamento de dados concluído com sucesso!")
    
    return {
        "true_df": true_df,
        "fake_df": fake_df,
        "combined_df": combined_df,
        "sample_df": sample_df
    }

# %% [markdown]
# ## Tutorial Passo a Passo

# %% [markdown]
# ### 1. Carregar Dados CSV

# %%
# Definir caminhos dos ficheiros CSV
# Nota: Ajuste os caminhos conforme necessário para o seu ambiente
fake_path = "/FileStore/tables/fake.csv"
true_path = "/FileStore/tables/real.csv"

# Carregar os ficheiros CSV
fake_df, true_df = load_csv_files(fake_path, true_path)

# %% [markdown]
# ### 2. Criar Tabelas Hive

# %%
# Criar tabelas Hive para os dados
create_hive_tables(fake_df, true_df)

# %% [markdown]
# ### 3. Carregar Dados das Tabelas Hive

# %%
# Carregar dados das tabelas Hive
true_df, fake_df = load_data_from_hive()

# %% [markdown]
# ### 4. Combinar e Pré-processar Dados

# %%
# Combinar datasets
combined_df = combine_datasets(true_df, fake_df)

# Pré-processar texto
combined_df = preprocess_text(combined_df)

# %% [markdown]
# ### 5. Criar Amostra Balanceada

# %%
# Criar amostra balanceada
sample_df = create_balanced_sample(combined_df)

# %% [markdown]
# ### 6. Analisar Características do Dataset

# %%
# Analisar características do dataset
results = analyze_dataset_characteristics(combined_df)

# %% [markdown]
# ### 7. Guardar Dados Processados

# %%
# Definir diretório de saída
output_dir = "dbfs:/FileStore/fake_news_detection/data"

# Guardar dataset combinado em DBFS
combined_path = f"{output_dir}/combined_data/combined_news.parquet"
save_to_parquet(combined_df, combined_path, partition_by="label")

# Guardar amostra em DBFS
sample_path = f"{output_dir}/sample_data/sample_news.parquet"
save_to_parquet(sample_df, sample_path)

# %% [markdown]
# ### 8. Guardar em Tabelas Hive

# %%
# Guardar em tabelas Hive para acesso mais fácil
save_to_hive_table(combined_df, "combined_news", partition_by="label")
save_to_hive_table(sample_df, "sample_news")

# %% [markdown]
# ### 9. Pipeline Completo (Alternativa)

# %%
# Executar o pipeline completo de uma só vez
# Descomente a linha abaixo para executar
# result = process_and_save_data()

# %% [markdown]
# ## Notas Importantes
# 
# 1. **Caminhos de Ficheiros**: Ajuste os caminhos dos ficheiros CSV conforme necessário para o seu ambiente Databricks.
# 
# 2. **Memória e Recursos**: O código está otimizado para o Databricks Community Edition, mas pode ser necessário ajustar as configurações de memória e partições dependendo do tamanho dos seus dados.
# 
# 3. **Data Leakage**: O código remove automaticamente a coluna "subject" para evitar data leakage, pois esta coluna discrimina perfeitamente entre notícias verdadeiras e falsas.
# 
# 4. **Armazenamento**: Os dados processados são guardados em formato Parquet e como tabelas Hive para facilitar o acesso em notebooks subsequentes.
# 
# 5. **Conversão para .ipynb**: Este ficheiro .py pode ser facilmente convertido para um notebook .ipynb usando jupytext:
#    ```
#    jupytext --to notebook 01_data_ingestion_standalone.py
#    ```
