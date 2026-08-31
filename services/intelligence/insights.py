"""Business insight derivation service.

This module interprets analytical columns already computed by the
analysis layer and derives structured, evidence-based business
insights. It performs no preprocessing, model inference, or
embedding/clustering computation of its own.
"""

import logging

import pandas as pd

from config import COLUMNS, SENTIMENTS
from schemas.models import BusinessInsights


logger = logging.getLogger(__name__)

_UNKNOWN = "Unknown"


class InsightsGenerator:
    """Derives structured business insights from processed review data."""

    def generate(self, df: pd.DataFrame) -> BusinessInsights:
        """Generate structured business insights from a processed DataFrame."""
        if df.empty:
            logger.info(
                "Empty DataFrame received; returning default insights."
            )
            return self._empty_insights()

        valid_df = self._filter_valid(df)

        if valid_df.empty:
            logger.info(
                "No valid rows found; returning default insights."
            )
            return self._empty_insights()

        overall_sentiment = self._compute_mode(
            valid_df,
            COLUMNS.SENTIMENT,
        )

        dominant_emotion = self._compute_mode(
            valid_df,
            COLUMNS.EMOTION,
        )

        top_positive_aspect = self._compute_top_aspect(
            valid_df,
            SENTIMENTS.POSITIVE,
        )

        top_negative_aspect = self._compute_top_aspect(
            valid_df,
            SENTIMENTS.NEGATIVE,
        )

        top_theme = self._compute_top_theme(valid_df)

        key_findings = self._compute_key_findings(
            valid_df,
            overall_sentiment=overall_sentiment,
            dominant_emotion=dominant_emotion,
            top_positive_aspect=top_positive_aspect,
            top_negative_aspect=top_negative_aspect,
            top_theme=top_theme,
        )

        logger.info(
            "Business insights generated from %d valid rows.",
            len(valid_df),
        )

        return BusinessInsights(
            overall_sentiment=overall_sentiment,
            top_positive_aspect=top_positive_aspect,
            top_negative_aspect=top_negative_aspect,
            dominant_emotion=dominant_emotion,
            top_theme=top_theme,
            key_findings=key_findings,
        )

    def _empty_insights(self) -> BusinessInsights:
        """Build a BusinessInsights instance with default values."""
        return BusinessInsights(
            overall_sentiment=_UNKNOWN,
            top_positive_aspect=_UNKNOWN,
            top_negative_aspect=_UNKNOWN,
            dominant_emotion=_UNKNOWN,
            top_theme=_UNKNOWN,
            key_findings=[],
        )

    def _filter_valid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict analysis to valid rows when possible."""
        if COLUMNS.IS_VALID not in df.columns:
            logger.warning(
                "Column '%s' not found; treating all rows as valid.",
                COLUMNS.IS_VALID,
            )
            return df

        return df[df[COLUMNS.IS_VALID] == True]  # noqa: E712

    def _compute_mode(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> str:
        """Compute the most frequent non-null value in a column."""
        if column not in df.columns:
            logger.warning(
                "Column '%s' not found; skipping.",
                column,
            )
            return _UNKNOWN

        values = df[column].dropna()
        values = values[
            values.astype(str).str.strip() != ""
        ]

        if values.empty:
            return _UNKNOWN

        return str(values.mode().iloc[0])

    def _compute_top_aspect(
        self,
        df: pd.DataFrame,
        sentiment_label: str,
    ) -> str:
        """Compute the most frequently mentioned aspect for a sentiment."""
        if COLUMNS.ASPECT not in df.columns:
            logger.warning(
                "Column '%s' not found; skipping.",
                COLUMNS.ASPECT,
            )
            return _UNKNOWN

        if COLUMNS.ASPECT_SENTIMENT in df.columns:
            filtered = df[
                df[COLUMNS.ASPECT_SENTIMENT] == sentiment_label
            ]
        elif COLUMNS.SENTIMENT in df.columns:
            logger.warning(
                "Column '%s' not found; falling back to '%s'.",
                COLUMNS.ASPECT_SENTIMENT,
                COLUMNS.SENTIMENT,
            )
            filtered = df[
                df[COLUMNS.SENTIMENT] == sentiment_label
            ]
        else:
            return _UNKNOWN

        aspects = filtered[COLUMNS.ASPECT].dropna()
        aspects = aspects[
            aspects.astype(str).str.strip() != ""
        ]

        if aspects.empty:
            return _UNKNOWN

        return str(aspects.mode().iloc[0])

    def _compute_top_theme(
        self,
        df: pd.DataFrame,
    ) -> str:
        """Compute the most prevalent theme or cluster."""
        has_cluster_id = COLUMNS.CLUSTER_ID in df.columns
        has_cluster_name = COLUMNS.CLUSTER_NAME in df.columns

        if not has_cluster_id and not has_cluster_name:
            logger.warning(
                "Columns '%s'/'%s' not found; skipping theme detection.",
                COLUMNS.CLUSTER_ID,
                COLUMNS.CLUSTER_NAME,
            )
            return _UNKNOWN

        clustered = df

        if has_cluster_id:
            clustered = clustered[
                clustered[COLUMNS.CLUSTER_ID].notna()
            ]

            clustered = clustered[
                clustered[COLUMNS.CLUSTER_ID] != -1
            ]

        if clustered.empty:
            return _UNKNOWN

        if has_cluster_name:
            names = clustered[COLUMNS.CLUSTER_NAME].dropna()
            names = names[
                names.astype(str).str.strip() != ""
            ]

            if not names.empty:
                return str(names.mode().iloc[0])

        if has_cluster_id:
            ids = clustered[COLUMNS.CLUSTER_ID].dropna()

            if not ids.empty:
                return str(ids.mode().iloc[0])

        return _UNKNOWN

    def _compute_key_findings(
        self,
        df: pd.DataFrame,
        overall_sentiment: str,
        dominant_emotion: str,
        top_positive_aspect: str,
        top_negative_aspect: str,
        top_theme: str,
    ) -> list[str]:
        """Assemble evidence-based key findings."""
        findings: list[str] = []
        total_rows = len(df)

        if COLUMNS.SENTIMENT in df.columns:
            sentiment_counts = (
                df[COLUMNS.SENTIMENT]
                .dropna()
                .value_counts()
            )

            if not sentiment_counts.empty:
                distribution = ", ".join(
                    f"{label}: {round((count / total_rows) * 100)}%"
                    for label, count in sentiment_counts.items()
                )

                findings.append(
                    f"Sentiment distribution — {distribution}."
                )

        if overall_sentiment != _UNKNOWN:
            findings.append(
                f"Overall sentiment is {overall_sentiment}."
            )

        if dominant_emotion != _UNKNOWN:
            findings.append(
                f"Dominant emotion detected is {dominant_emotion}."
            )

        if top_positive_aspect != _UNKNOWN:
            findings.append(
                "Most frequently mentioned positive aspect: "
                f"{top_positive_aspect}."
            )

        if top_negative_aspect != _UNKNOWN:
            findings.append(
                "Most frequently mentioned negative aspect: "
                f"{top_negative_aspect}."
            )

        if top_theme != _UNKNOWN:
            findings.append(
                f"Most prevalent theme/cluster: {top_theme}."
            )

        if COLUMNS.RATING in df.columns:
            ratings = pd.to_numeric(
                df[COLUMNS.RATING],
                errors="coerce",
            ).dropna()

            if not ratings.empty:
                findings.append(
                    "Average rating across valid reviews is "
                    f"{round(ratings.mean(), 2)}."
                )

        return findings