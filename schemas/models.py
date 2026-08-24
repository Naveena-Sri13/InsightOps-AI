"""Core data models for InsightOps AI.

This module defines the dataclasses used to represent business
insights, dashboard metrics, processing statistics, metadata, and the
overall analysis result produced by the application.

No business logic or methods are defined here.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class BusinessInsights:
    """High-level business insights derived from analyzed reviews.

    Attributes:
        overall_sentiment: Aggregate sentiment across all reviews.
        top_positive_aspect: Aspect with the most positive sentiment.
        top_negative_aspect: Aspect with the most negative sentiment.
        dominant_emotion: Most frequently detected emotion.
        top_theme: Most prevalent theme or cluster.
        key_findings: List of notable findings extracted from the data.
    """

    overall_sentiment: str
    top_positive_aspect: str
    top_negative_aspect: str
    dominant_emotion: str
    top_theme: str
    critical_issues: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)


@dataclass
class DashboardMetrics:
    """Summary metrics displayed on the dashboard.

    Attributes:
        total_reviews: Total number of reviews analyzed.
        positive_count: Number of reviews classified as positive.
        neutral_count: Number of reviews classified as neutral.
        negative_count: Number of reviews classified as negative.
        average_rating: Average numeric rating across all reviews.
        top_aspect: Most frequently mentioned aspect.
        top_emotion: Most frequently detected emotion.
    """

    total_reviews: int
    positive_count: int
    neutral_count: int
    negative_count: int
    positive_percentage: float
    neutral_percentage: float
    negative_percentage: float
    average_rating: float
    top_aspect: str
    top_emotion: str


@dataclass
class ProcessingStats:
    """Statistics describing a data processing run.

    Attributes:
        rows_uploaded: Number of rows present in the uploaded input.
        rows_processed: Number of rows successfully processed.
        rows_rejected: Number of rows rejected during processing.
        processing_time_seconds: Total time taken to process the data,
            in seconds.
    """

    rows_uploaded: int
    rows_processed: int
    rows_rejected: int
    processing_time_seconds: float
    average_review_length: float
    languages_detected: int


@dataclass
class Metadata:
    """Metadata describing the application and generation context.

    Attributes:
        app_name: Name of the application.
        app_version: Version of the application.
        generated_at: Timestamp at which the associated output was
            generated.
    """

    app_name: str
    app_version: str
    pipeline_version: str
    generated_at: datetime
    models_used: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """AI-generated recommendation."""

    priority: str
    category: str
    title: str
    description: str
    affected_reviews: int
    confidence: float



@dataclass
class AnalysisResult:
    """Complete result of an end-to-end analysis run.

    Attributes:
        processed_df: DataFrame containing the fully processed review
            data.
        business_insights: Derived business insights.
        executive_summary: Narrative summary of the analysis.
        recommendations: List of recommended actions.
        dashboard_metrics: Summary metrics for the dashboard.
        metadata: Application and generation metadata.
        processing_stats: Statistics from the data processing run.
    """

    processed_df: pd.DataFrame
    business_insights: BusinessInsights
    executive_summary: str
    recommendations: list[Recommendation]
    dashboard_metrics: DashboardMetrics
    metadata: Metadata
    processing_stats: ProcessingStats
