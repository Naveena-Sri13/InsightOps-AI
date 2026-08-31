"""Review summarization service.

This module defines a Summarizer class that creates a concise,
evidence-based summary from already processed review data.

It does not perform preprocessing, sentiment analysis, emotion
analysis, aspect extraction, clustering, or recommendation generation.
"""

import logging

import pandas as pd

from config import COLUMNS

logger = logging.getLogger(__name__)

_UNKNOWN = "Unknown"


class Summarizer:
    """Generates a concise summary from processed review data."""

    def summarize(self, df: pd.DataFrame) -> str:
        """Generate a concise, evidence-based summary.

        Args:
            df: DataFrame containing previously computed analytical
                columns such as sentiment, emotion, aspect, and
                cluster information.

        Returns:
            A concise summary string based only on available data.
        """
        if df.empty:
            logger.info("Empty DataFrame received; returning empty summary.")
            return "No review data is available for summarization."

        valid_df = self._filter_valid(df)

        if valid_df.empty:
            logger.info("No valid rows found; returning empty summary.")
            return "No valid review data is available for summarization."

        total_reviews = len(valid_df)

        sentiment = self._mode(valid_df, COLUMNS.SENTIMENT)
        emotion = self._mode(valid_df, COLUMNS.EMOTION)
        aspect = self._mode(valid_df, COLUMNS.ASPECT)

        parts: list[str] = []

        parts.append(
            f"Analysis of {total_reviews} valid review"
            f"{'s' if total_reviews != 1 else ''} indicates "
            f"an overall {sentiment.lower()} sentiment."
        )

        if emotion != _UNKNOWN:
            parts.append(f"The dominant emotion is {emotion}.")

        if aspect != _UNKNOWN:
            parts.append(
                f"The most frequently mentioned aspect is {aspect}."
            )

        if COLUMNS.RATING in valid_df.columns:
            ratings = pd.to_numeric(
                valid_df[COLUMNS.RATING], errors="coerce"
            ).dropna()

            if not ratings.empty:
                average_rating = round(ratings.mean(), 2)
                parts.append(
                    f"The average rating is {average_rating}."
                )

        summary = " ".join(parts)

        logger.info(
            "Review summary generated successfully from %d valid rows.",
            total_reviews,
        )

        return summary

    def _filter_valid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only valid rows when the validation column exists."""
        if COLUMNS.IS_VALID not in df.columns:
            logger.warning(
                "Column '%s' not found; treating all rows as valid.",
                COLUMNS.IS_VALID,
            )
            return df

        return df[df[COLUMNS.IS_VALID] == True]  # noqa: E712

    def _mode(self, df: pd.DataFrame, column: str) -> str:
        """Return the most frequent usable value from a column."""
        if column not in df.columns:
            logger.warning(
                "Column '%s' not found while generating summary.",
                column,
            )
            return _UNKNOWN

        values = df[column].dropna()
        values = values[values.astype(str).str.strip() != ""]

        if values.empty:
            return _UNKNOWN

        return str(values.mode().iloc[0])