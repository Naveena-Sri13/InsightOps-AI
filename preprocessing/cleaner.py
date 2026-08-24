"""Text cleaning for uploaded review data.

This module defines a Cleaner class that removes HTML tags, URLs,
email addresses, and HTML entities from review text, and normalizes
whitespace, without altering the semantic content of the text (no
lowercasing, stemming, lemmatization, punctuation removal, emoji
removal, or stopword removal).
"""

import html
import logging
import re

import pandas as pd

from config import COLUMNS

logger = logging.getLogger(__name__)

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


class Cleaner:
    """Cleans review text while preserving its original meaning.

    The cleaner strips HTML tags, URLs, email addresses, and decodes
    HTML entities, then collapses repeated whitespace and trims the
    result. It does not perform any semantic-altering transformations
    such as lowercasing or stopword removal.
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the source text column of a DataFrame.

        Adds or updates the ``clean_text`` column on the given
        DataFrame. Rows where ``source_text`` is missing or not a
        string are assigned an empty string in ``clean_text``.

        Args:
            df: DataFrame containing uploaded review data.

        Returns:
            The same DataFrame, with the ``clean_text`` column added
            or updated. No rows are added or removed.
        """
        if COLUMNS.SOURCE_TEXT not in df.columns:
            logger.error(
                "Missing required column '%s'; cannot clean text.",
                COLUMNS.SOURCE_TEXT,
            )
            df[COLUMNS.CLEAN_TEXT] = ""
            return df

        df[COLUMNS.CLEAN_TEXT] = df[COLUMNS.SOURCE_TEXT].apply(self._clean_text)

        logger.info("Cleaned text for %d rows.", len(df))

        return df

    def _clean_text(self, value: object) -> str:
        """Clean a single text value.

        Args:
            value: The raw value found in the source_text column.

        Returns:
            The cleaned text, or an empty string if the input is not
            a valid string.
        """
        if not isinstance(value, str):
            return ""

        text = html.unescape(value)
        text = _HTML_TAG_PATTERN.sub(" ", text)
        text = _URL_PATTERN.sub(" ", text)
        text = _EMAIL_PATTERN.sub(" ", text)
        text = _WHITESPACE_PATTERN.sub(" ", text)
        text = text.strip()

        return text
