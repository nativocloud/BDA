# Data Preprocessing and Feature Engineering for Fake News Detection

## Table of Contents

1. [Introduction](#introduction)
2. [Data Understanding](#data-understanding)
3. [Text Preprocessing](#text-preprocessing)
4. [Entity Extraction](#entity-extraction)
5. [Feature Engineering](#feature-engineering)
6. [PySpark Implementation](#pyspark-implementation)
7. [Cross-Validation Strategy](#cross-validation-strategy)
8. [References](#references)

## Introduction

Data preprocessing and feature engineering are critical steps in building an effective fake news detection system. Raw news articles contain unstructured text, metadata, and implicit relationships that must be transformed into structured features suitable for machine learning algorithms. This book explores the techniques and implementations used in our fake news detection system to prepare data for analysis.

The preprocessing and feature engineering pipeline is implemented using PySpark to leverage distributed computing capabilities, enabling scalable processing of large news datasets. We follow data science best practices to ensure the quality and integrity of the features while preventing data leakage that could lead to overly optimistic model performance.

## Data Understanding

### Dataset Overview

Our fake news detection system uses two primary datasets:

1. **Fake.csv**: Contains articles labeled as fake news
2. **True.csv**: Contains articles labeled as genuine news

Each dataset includes the following fields:
- **title**: The headline of the article
- **text**: The main content of the article
- **subject**: The subject or category of the article
- **date**: The publication date

Understanding the characteristics of these datasets is essential for effective preprocessing and feature engineering.

### Exploratory Data Analysis

Before diving into preprocessing, we conducted exploratory data analysis to understand the data distribution, identify patterns, and detect potential issues. Key findings include:

- Distribution of article lengths
- Common topics and subjects
- Temporal patterns in publication dates
- Vocabulary differences between fake and genuine news
- Entity frequency and co-occurrence patterns

This analysis informed our preprocessing strategy and feature engineering approach.

## Text Preprocessing

Text preprocessing transforms raw text into a clean, normalized format suitable for feature extraction. Our preprocessing pipeline includes the following steps:

### 1. Text Cleaning

- Removing HTML tags and special characters
- Converting text to lowercase
- Handling missing values
- Removing URLs and non-textual content

### 2. Tokenization

Tokenization splits text into individual tokens (words, phrases, or symbols). We use PySpark's built-in tokenization functions for efficient distributed processing:

```python
from pyspark.ml.feature import Tokenizer, RegexTokenizer

# Create tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# Apply tokenization
tokenized_df = tokenizer.transform(df)
```

### 3. Stop Word Removal

Stop words are common words (e.g., "the", "is", "at") that typically don't carry significant meaning for classification tasks. We remove these words to reduce noise and dimensionality:

```python
from pyspark.ml.feature import StopWordsRemover

# Create stop words remover
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Apply stop words removal
filtered_df = remover.transform(tokenized_df)
```

### 4. Stemming and Lemmatization

Stemming and lemmatization reduce words to their root forms, helping to normalize variations of the same word. We implement these techniques using NLTK within a PySpark UDF (User Defined Function):

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define UDFs for stemming and lemmatization
stem_udf = udf(lambda words: [stemmer.stem(word) for word in words], ArrayType(StringType()))
lemmatize_udf = udf(lambda words: [lemmatizer.lemmatize(word) for word in words], ArrayType(StringType()))

# Apply stemming or lemmatization
stemmed_df = filtered_df.withColumn("stemmed_words", stem_udf("filtered_words"))
```

## Entity Extraction

Entity extraction identifies named entities such as people, places, organizations, and events mentioned in the text. This information provides valuable context for fake news detection.

### Named Entity Recognition (NER)

We use spaCy for named entity recognition, implemented as a PySpark UDF for distributed processing:

```python
import spacy
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, StructType, StructField

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define schema for entity extraction
entity_schema = ArrayType(
    StructType([
        StructField("text", StringType(), True),
        StructField("label", StringType(), True)
    ])
)

# Define UDF for entity extraction
def extract_entities(text):
    if not text:
        return []
    
    doc = nlp(text[:100000])  # Limit text length to avoid memory issues
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities

extract_entities_udf = udf(extract_entities, entity_schema)

# Apply entity extraction
entities_df = df.withColumn("entities", extract_entities_udf("text"))
```

### Entity Categorization

After extraction, we categorize entities into different types:

```python
from pyspark.sql.functions import col, explode

# Extract specific entity types
people_df = entities_df.withColumn("entity", explode("entities")) \
    .filter(col("entity.label") == "PERSON") \
    .select(col("id"), col("entity.text").alias("person"))

places_df = entities_df.withColumn("entity", explode("entities")) \
    .filter(col("entity.label").isin(["GPE", "LOC"])) \
    .select(col("id"), col("entity.text").alias("place"))

organizations_df = entities_df.withColumn("entity", explode("entities")) \
    .filter(col("entity.label") == "ORG") \
    .select(col("id"), col("entity.text").alias("organization"))
```

## Feature Engineering

Feature engineering transforms preprocessed text and extracted entities into numerical features suitable for machine learning algorithms.

### Text-Based Features

#### TF-IDF Vectorization

Term Frequency-Inverse Document Frequency (TF-IDF) is a statistical measure that evaluates the importance of a word in a document relative to a collection of documents:

```python
from pyspark.ml.feature import HashingTF, IDF

# Create TF (Term Frequency) vectors
hashingTF = HashingTF(inputCol="filtered_words", outputCol="tf_features", numFeatures=10000)
tf_df = hashingTF.transform(filtered_df)

# Create IDF (Inverse Document Frequency) model
idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)
```

#### Word Embeddings

Word embeddings capture semantic relationships between words by representing them as dense vectors:

```python
from pyspark.ml.feature import Word2Vec

# Create Word2Vec model
word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words", outputCol="word2vec_features")
model = word2vec.fit(filtered_df)
w2v_df = model.transform(filtered_df)
```

### Entity-Based Features

We create features based on extracted entities to capture the network structure of news articles:

```python
from pyspark.sql.functions import size, array_distinct

# Entity count features
entity_features_df = entities_df \
    .withColumn("person_count", size(col("people"))) \
    .withColumn("place_count", size(col("places"))) \
    .withColumn("org_count", size(col("organizations"))) \
    .withColumn("unique_person_count", size(array_distinct(col("people")))) \
    .withColumn("unique_place_count", size(array_distinct(col("places")))) \
    .withColumn("unique_org_count", size(array_distinct(col("organizations"))))
```

### Metadata Features

We extract features from article metadata such as publication date, source, and subject:

```python
from pyspark.sql.functions import to_timestamp, hour, dayofweek, month, year

# Date-time features
date_features_df = df \
    .withColumn("timestamp", to_timestamp(col("date"))) \
    .withColumn("hour", hour(col("timestamp"))) \
    .withColumn("day_of_week", dayofweek(col("timestamp"))) \
    .withColumn("month", month(col("timestamp"))) \
    .withColumn("year", year(col("timestamp")))
```

### Graph-Based Features

Using GraphX, we create features based on entity relationships and network structure:

```python
from graphframes import GraphFrame
from pyspark.sql.functions import monotonically_increasing_id

# Create vertices
vertices = entities_df.select(
    monotonically_increasing_id().alias("id"),
    col("entity.text").alias("name"),
    col("entity.label").alias("type")
).distinct()

# Create edges
edges = entities_df.select(
    col("id").alias("src"),
    col("entity.text").alias("dst"),
    lit("MENTIONS").alias("relationship")
)

# Create graph
g = GraphFrame(vertices, edges)

# Calculate PageRank
pagerank = g.pageRank(resetProbability=0.15, tol=0.01)
```

## PySpark Implementation

Our complete preprocessing and feature engineering pipeline is implemented using PySpark's ML Pipeline API, which provides a unified interface for chaining multiple transformations:

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler

# Define pipeline stages
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="tf_features", numFeatures=10000)
idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
assembler = VectorAssembler(
    inputCols=["tfidf_features", "person_count", "place_count", "org_count"],
    outputCol="features"
)

# Create and run pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, assembler])
model = pipeline.fit(train_data)
processed_data = model.transform(train_data)
```

## Cross-Validation Strategy

To ensure robust evaluation and prevent data leakage, we implement a proper cross-validation strategy:

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Define parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [5000, 10000]) \
    .addGrid(classifier.regParam, [0.1, 0.01]) \
    .build()

# Define evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# Create cross-validator
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5
)

# Run cross-validation
cv_model = cv.fit(train_data)
```

### Preventing Data Leakage

Data leakage occurs when information from outside the training dataset is used to create the model. To prevent this, we:

1. Split data before any preprocessing or feature engineering
2. Apply the same preprocessing steps to training and test data
3. Fit feature extraction models (e.g., IDF, Word2Vec) only on training data
4. Use pipeline API to ensure consistent application of transformations

```python
# Split data first
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# Create and fit pipeline on training data only
pipeline = Pipeline(stages=[...])
model = pipeline.fit(train_data)

# Apply the same transformations to test data
test_processed = model.transform(test_data)
```

## References

1. Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., ... & Stoica, I. (2016). Apache spark: a unified engine for big data processing. Communications of the ACM, 59(11), 56-65.

2. Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.

3. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26.

4. Dave, A., Jindal, A., Li, L. E., Xin, R., Gonzalez, J., & Zaharia, M. (2016). GraphFrames: an integrated API for mixing graph and relational queries. In Proceedings of the Fourth International Workshop on Graph Data Management Experiences and Systems (pp. 1-8).

5. Khan, J. Y., Khondaker, M. T. I., Afroz, S., Uddin, G., & Iqbal, A. (2021). A benchmark study of machine learning models for online fake news detection. Machine Learning with Applications, 4, 100032.

---

In the next book, we will explore the machine learning models used in our fake news detection system, including traditional algorithms, deep learning approaches, and graph-based methods.

# Last modified: May 29, 2025
