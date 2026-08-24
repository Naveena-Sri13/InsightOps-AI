"""Configuration values for InsightOps AI.

This module contains only static configuration constants: application
metadata, directory paths, dataframe column names, supported label
sets, default thresholds, model name references, and export formats.

No business logic, AI inference, or Streamlit code is present here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

APP_NAME: Final[str] = "InsightOps AI"
APP_VERSION: Final[str] = "1.0.0"


# ---------------------------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------------------------

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
ASSETS_DIR: Final[Path] = BASE_DIR / "assets"

# ---------------------------------------------------------------------------
# General Configuration
# ---------------------------------------------------------------------------

DEFAULT_TEXT_COLUMN: Final[str] = "source_text"
DEFAULT_ENCODING: Final[str] = "utf-8"
RANDOM_SEED: Final[int] = 42
DEFAULT_BATCH_SIZE: Final[int] = 32

# ---------------------------------------------------------------------------
# DataFrame Column Names
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Columns:
    """Canonical dataframe column name constants.

    Attributes:
        ROW_ID: Unique identifier for a row.
        SOURCE_TEXT: Raw text as ingested from the source.
        RATING: Numeric or star rating associated with the record.
        SOURCE: Origin of the record (e.g., platform or channel).
        INGESTED_AT: Timestamp at which the record was ingested.
        IS_VALID: Whether the record passed validation.
        VALIDATION_ERROR: Description of a validation failure, if any.
        CLEAN_TEXT: Text after cleaning/preprocessing.
        NORMALIZED_TEXT: Text after normalization.
        SENTIMENT: Predicted sentiment label.
        SENTIMENT_CONFIDENCE: Confidence score for the sentiment label.
        SENTIMENT_SCORE: Numeric sentiment score.
        EMOTION: Predicted emotion label.
        EMOTION_CONFIDENCE: Confidence score for the emotion label.
        ASPECT: Extracted aspect/topic term.
        ASPECT_SENTIMENT: Sentiment associated with the aspect.
        EMBEDDING: Vector embedding representation of the text.
        CLUSTER_ID: Identifier of the assigned cluster.
        CLUSTER_NAME: Human-readable name of the assigned cluster.
    """

    ROW_ID: str = "row_id"
    SOURCE_TEXT: str = "source_text"
    RATING: str = "rating"
    SOURCE: str = "source"
    INGESTED_AT: str = "ingested_at"
    IS_VALID: str = "is_valid"
    VALIDATION_ERROR: str = "validation_error"
    CLEAN_TEXT: str = "clean_text"
    NORMALIZED_TEXT: str = "normalized_text"
    SENTIMENT: str = "sentiment"
    SENTIMENT_CONFIDENCE: str = "sentiment_confidence"
    SENTIMENT_SCORE: str = "sentiment_score"
    EMOTION: str = "emotion"
    EMOTION_CONFIDENCE: str = "emotion_confidence"
    ASPECT: str = "aspect"
    ASPECT_SENTIMENT: str = "aspect_sentiment"
    ASPECT_CONFIDENCE: str = "aspect_confidence"
    EMBEDDING: str = "embedding"
    CLUSTER_ID: str = "cluster_id"
    CLUSTER_NAME: str = "cluster_name"


COLUMNS: Final[Columns] = Columns()


# ---------------------------------------------------------------------------
# Supported Sentiments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sentiments:
    """Supported sentiment label constants.

    Attributes:
        POSITIVE: Positive sentiment label.
        NEUTRAL: Neutral sentiment label.
        NEGATIVE: Negative sentiment label.
    """

    POSITIVE: str = "Positive"
    NEUTRAL: str = "Neutral"
    NEGATIVE: str = "Negative"


SENTIMENTS: Final[Sentiments] = Sentiments()

SUPPORTED_SENTIMENTS: Final[tuple[str, ...]] = (
    SENTIMENTS.POSITIVE,
    SENTIMENTS.NEUTRAL,
    SENTIMENTS.NEGATIVE,
)


# ---------------------------------------------------------------------------
# Supported Emotions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Emotions:
    """Supported emotion label constants.

    Attributes:
        JOY: Joy emotion label.
        SADNESS: Sadness emotion label.
        ANGER: Anger emotion label.
        FEAR: Fear emotion label.
        SURPRISE: Surprise emotion label.
        DISGUST: Disgust emotion label.
    """

    JOY: str = "Joy"
    SADNESS: str = "Sadness"
    ANGER: str = "Anger"
    FEAR: str = "Fear"
    SURPRISE: str = "Surprise"
    DISGUST: str = "Disgust"


EMOTIONS: Final[Emotions] = Emotions()

SUPPORTED_EMOTIONS: Final[tuple[str, ...]] = (
    EMOTIONS.JOY,
    EMOTIONS.SADNESS,
    EMOTIONS.ANGER,
    EMOTIONS.FEAR,
    EMOTIONS.SURPRISE,
    EMOTIONS.DISGUST,
)


# ---------------------------------------------------------------------------
# Default Thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Thresholds:
    """Default threshold values used across the application.

    Attributes:
        CONFIDENCE_THRESHOLD: Minimum confidence required to accept a
            model prediction.
        MAXIMUM_UPLOAD_SIZE_MB: Maximum allowed upload size, in
            megabytes.
        MINIMUM_REVIEW_LENGTH: Minimum character length for a review
            to be considered valid.
    """

    CONFIDENCE_THRESHOLD: float = 0.60
    MAXIMUM_UPLOAD_SIZE_MB: int = 20
    MINIMUM_REVIEW_LENGTH: int = 5


THRESHOLDS: Final[Thresholds] = Thresholds()


# ---------------------------------------------------------------------------
# Model Names
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelNames:
    """Model name string constants.

    These are reference identifiers only. No models are downloaded or
    initialized in this module.

    Attributes:
        SENTIMENT_MODEL: Name of the sentiment classification model.
        EMOTION_MODEL: Name of the emotion classification model.
        EMBEDDING_MODEL: Name of the text embedding model.
    """

    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


MODEL_NAMES: Final[ModelNames] = ModelNames()


# ---------------------------------------------------------------------------
# Export Formats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExportFormats:
    """Supported export format constants.

    Attributes:
        CSV: CSV export format identifier.
        PDF: PDF export format identifier.
        JSON: JSON export format identifier.
    """

    CSV: str = "CSV"
    PDF: str = "PDF"
    JSON: str = "JSON"


EXPORT_FORMATS: Final[ExportFormats] = ExportFormats()

SUPPORTED_EXPORT_FORMATS: Final[tuple[str, ...]] = (
    EXPORT_FORMATS.CSV,
    EXPORT_FORMATS.PDF,
    EXPORT_FORMATS.JSON,
)
