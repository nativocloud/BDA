"""
GraphX utility functions for fake news detection project.
This module contains functions for creating and analyzing graph-based features.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, collect_list, struct, count
import os

def create_word_cooccurrence_graph(spark, df, text_col="text", window_size=2):
    """
    Create a word co-occurrence graph from text data.
    
    Args:
        spark (SparkSession): Spark session
        df: DataFrame with text data
        text_col (str): Column containing text data
        window_size (int): Window size for co-occurrence
        
    Returns:
        DataFrame: Graph edges with weights
    """
    # Tokenize text
    words_df = df.select(
        col("label"),
        explode(split(col(text_col), " ")).alias("word")
    ).filter(col("word") != "")
    
    # Create window for co-occurrence
    window_df = words_df.withColumn("word_id", monotonically_increasing_id())
    
    # Self-join to find co-occurring words within window
    joined_df = window_df.alias("a").join(
        window_df.alias("b"),
        (col("a.word_id") != col("b.word_id")) & 
        (col("a.word_id") - col("b.word_id") <= window_size) & 
        (col("a.word_id") - col("b.word_id") > 0)
    )
    
    # Create edges with weights
    edges_df = joined_df.select(
        col("a.word").alias("src"),
        col("b.word").alias("dst")
    ).groupBy("src", "dst").count().withColumnRenamed("count", "weight")
    
    return edges_df

def extract_graph_features(spark, text_df, graph_df):
    """
    Extract graph-based features from text using the word co-occurrence graph.
    
    Args:
        spark (SparkSession): Spark session
        text_df: DataFrame with text data
        graph_df: Graph edges DataFrame
        
    Returns:
        DataFrame: Text data with graph-based features
    """
    # Register graph as a temporary view
    graph_df.createOrReplaceTempView("word_graph")
    
    # Calculate node centrality (degree centrality)
    centrality_df = spark.sql("""
        SELECT 
            src as word, 
            SUM(weight) as centrality
        FROM word_graph
        GROUP BY src
        UNION ALL
        SELECT 
            dst as word, 
            SUM(weight) as centrality
        FROM word_graph
        GROUP BY dst
    """).groupBy("word").sum("centrality").withColumnRenamed("sum(centrality)", "centrality")
    
    # Register centrality as a temporary view
    centrality_df.createOrReplaceTempView("word_centrality")
    
    # Tokenize text
    words_df = text_df.select(
        col("label"),
        col("text"),
        explode(split(col("text"), " ")).alias("word")
    ).filter(col("word") != "")
    
    # Join with centrality
    words_with_centrality = words_df.join(
        centrality_df,
        words_df.word == centrality_df.word,
        "left"
    ).na.fill(0)
    
    # Aggregate centrality features by text
    text_with_features = words_with_centrality.groupBy("text", "label").agg(
        collect_list(struct("word", "centrality")).alias("word_centrality"),
        count("word").alias("word_count"),
        sum("centrality").alias("total_centrality"),
        avg("centrality").alias("avg_centrality"),
        max("centrality").alias("max_centrality")
    )
    
    return text_with_features

def create_graphx_features(spark, df, output_col="graphx_features"):
    """
    Create GraphX-based features for text classification.
    
    Args:
        spark (SparkSession): Spark session
        df: DataFrame with text data
        output_col (str): Output column name for graph features
        
    Returns:
        DataFrame: DataFrame with graph features
    """
    from pyspark.ml.feature import VectorAssembler
    
    # Create word co-occurrence graph
    graph_df = create_word_cooccurrence_graph(spark, df)
    
    # Extract graph features
    features_df = extract_graph_features(spark, df, graph_df)
    
    # Assemble features into a vector
    assembler = VectorAssembler(
        inputCols=["total_centrality", "avg_centrality", "max_centrality", "word_count"],
        outputCol=output_col
    )
    
    result_df = assembler.transform(features_df)
    
    return result_df
