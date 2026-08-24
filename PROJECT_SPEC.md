# InsightOps AI

## Product Vision

InsightOps AI is a decision-oriented sentiment intelligence platform.

The application analyzes customer feedback and transforms raw text into actionable business insights using AI.

Instead of simply classifying sentiment, the system identifies emotions, product aspects, recurring themes, and provides executive summaries with AI-generated recommendations.

---

# Architecture

Presentation Layer
- dashboard/
- visualization/

Business Layer
- services/
- reports/

Data Layer
- preprocessing/
- schemas/

Infrastructure Layer
- utils/
- config.py

---

# Pipeline

CSV Upload

↓

Validation

↓

Cleaning

↓

Normalization

↓

Sentiment Analysis

↓

Emotion Detection

↓

Aspect Extraction

↓

Embedding Generation

↓

Theme Clustering

↓

Business Insight Generation

↓

Executive Summary

↓

Recommendations

↓

AnalysisResult

↓

Dashboard / Reports / Export

---

# DataFrame Contract

Every analysis module:

Input:
- pandas.DataFrame

Output:
- pandas.DataFrame

Rule:
- Never remove columns.
- Never overwrite original columns.
- Only append new columns.

---

# Master DataFrame Schema

## Upload

- row_id
- source_text
- rating
- source
- ingested_at

## Validation

- is_valid
- validation_error

## Cleaning

- clean_text

## Normalization

- normalized_text

## Sentiment

- sentiment
- sentiment_confidence
- sentiment_score

## Emotion

- emotion
- emotion_confidence

## Aspect

- aspect
- aspect_sentiment

## Embeddings

- embedding

## Clustering

- cluster_id
- cluster_name

---

# AnalysisResult

Contains:

- processed_df
- business_insights
- executive_summary
- recommendations
- dashboard_metrics
- metadata
- processing_stats

---

# Module Contracts

## validator.py

Input:
DataFrame

Output:
DataFrame

Adds:
- is_valid
- validation_error

---

## cleaner.py

Input:
DataFrame

Output:
DataFrame

Adds:
- clean_text

---

## normalizer.py

Input:
DataFrame

Output:
DataFrame

Adds:
- normalized_text

---

## sentiment.py

Input:
DataFrame

Output:
DataFrame

Adds:
- sentiment
- sentiment_confidence
- sentiment_score

---

## emotion.py

Input:
DataFrame

Output:
DataFrame

Adds:
- emotion
- emotion_confidence

---

## aspects.py

Input:
DataFrame

Output:
DataFrame

Adds:
- aspect
- aspect_sentiment

---

## embeddings.py

Input:
DataFrame

Output:
DataFrame

Adds:
- embedding

---

## clustering.py

Input:
DataFrame

Output:
DataFrame

Adds:
- cluster_id
- cluster_name