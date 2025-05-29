# Literature Review: Fake News Detection Methodologies

## Overview of Key Approaches

Based on the reviewed paper "Fake News Detection using Machine Learning Algorithms" by Uma Sharma, Sidarth Saran, and Shankar M. Patil, several important methodologies and approaches for fake news detection have been identified. This document summarizes these approaches and their relevance to our implementation.

## Key Methodologies

### 1. Naive Bayes Classification

Mykhaylo Granik et al. implemented a simple approach using naive Bayes classifier for Facebook news posts, achieving approximately 74% classification accuracy. This approach was tested on posts from both politically biased and mainstream sources.

**Relevance to our project**: We have already implemented a Naive Bayes model as part of our baseline approach, which aligns with this methodology. The paper confirms that Naive Bayes is a valid approach for fake news detection.

### 2. Combined Content and Social Context Features

Marco L. Della Vedova et al. proposed a novel ML approach that combines news content with social context features, achieving 78.8% accuracy. They further implemented this method on Facebook Messenger Chatbot with 81.7% accuracy.

**Relevance to our project**: This supports our approach of extracting both content-based features (text analysis) and metadata features (sources, entities). We should ensure our feature engineering incorporates both content and social context.

### 3. Bag-of-Words with Feature Selection

Himank Gupta et al. developed a framework using machine learning with lightweight features derived from Bag-of-Words model. They achieved 91.65% accuracy by selecting features with the highest information gain.

**Relevance to our project**: We should incorporate feature selection techniques to identify the most informative features from our TF-IDF vectors, potentially improving model performance.

### 4. Logistic Regression for Classification

The paper mentions using Logistic Regression specifically for fake news detection, tested on a manually labeled news dataset.

**Relevance to our project**: We should ensure Logistic Regression is included in our model comparison, as it appears to be effective for this specific task.

### 5. Twitter Credibility Assessment

Cody Buntain et al. developed a method for automating fake news detection on Twitter by learning to predict credibility assessments.

**Relevance to our project**: While our current dataset doesn't include Twitter data, the credibility assessment approach could be adapted to our news articles by extracting credibility signals.

## Characteristics of Fake News

The paper identifies several key characteristics of fake news:

1. Often contains grammatical mistakes
2. Usually emotionally colored
3. Uses attention-seeking words and clickbait
4. Sources are often not genuine
5. Content is not always factually accurate

**Relevance to our project**: These characteristics should inform our feature engineering. We should extract features related to:
- Grammatical quality (spelling errors, sentence structure)
- Emotional tone (sentiment analysis)
- Clickbait detection (question marks, sensational words)
- Source credibility (based on our source metadata extraction)

## Dataset Considerations

The paper mentions several datasets used in fake news research:
- Facebook posts from politically biased and mainstream sources
- HSPam14 dataset (400,000 tweets)
- Dataset of 15,500 posts from 32 pages (14 conspiracy, 18 scientific)

**Relevance to our project**: While we're using our own dataset, these references provide context on typical dataset compositions and potential skewness issues (the paper notes only 4.9% of one dataset was fake news).

## Implementation Approaches

The paper discusses both standalone systems and chatbot implementations for fake news detection.

**Relevance to our project**: Our streaming pipeline implementation aligns with the real-time detection systems mentioned in the paper. We should ensure our implementation can handle streaming data effectively.

## Conclusion

The methodologies identified in this literature review support our current approach of using multiple models (Naive Bayes, Random Forest) and extracting rich metadata features. We should incorporate the following insights:

1. Combine content-based and social context features
2. Implement feature selection to identify most informative features
3. Include credibility signals in our feature set
4. Extract features related to the identified characteristics of fake news
5. Ensure our models can handle class imbalance

These insights will be integrated into our feature engineering, model selection, and evaluation processes.
