"""Text normalization for cleaned review data.

This module defines a Normalizer class that converts cleaned review
text into a consistent representation: Unicode NFC normalization,
standardized quotation marks and apostrophes, and collapsed
whitespace. It does not alter the semantic content of the text (no
lowercasing, stemming, lemmatization, punctuation removal, emoji
removal, stopword removal, spell correction, or translation).
"""

import logging
import re
import unicodedata

import pandas as pd

from config import COLUMNS

logger = logging.getLogger(__name__)

_QUOTE_PATTERN = re.compile(r"[\u201c\u201d\u201e\u201f\u2033\u2036]")
_APOSTROPHE_PATTERN = re.compile(r"[\u2018\u2019\u201a\u201b\u2032\u2035]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


class Normalizer:
    """Normalizes cleaned review text into a consistent representation.

    The normalizer applies Unicode NFC normalization, standardizes
    quotation marks and apostrophes, and collapses repeated
    whitespace. It does not perform any semantic-altering
    transformations.
    """

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the cleaned text column of a DataFrame.

        Adds or updates the ``normalized_text`` column on the given
        DataFrame. Rows where ``clean_text`` is missing or not a
        string are assigned an empty string in ``normalized_text``.

        Args:
            df: DataFrame containing cleaned review data.

        Returns:
            The same DataFrame, with the ``normalized_text`` column
            added or updated. No rows are added or removed.
        """
        if COLUMNS.CLEAN_TEXT not in df.columns:
            logger.error(
                "Missing required column '%s'; cannot normalize text.",
                COLUMNS.CLEAN_TEXT,
            )
            df[COLUMNS.NORMALIZED_TEXT] = ""
            return df

        df[COLUMNS.NORMALIZED_TEXT] = df[COLUMNS.CLEAN_TEXT].apply(
            self._normalize_text
        )

        logger.info("Normalized text for %d rows.", len(df))

        return df

    def _normalize_text(self, value: object) -> str:
        """Normalize a single text value.

        Args:
            value: The raw value found in the clean_text column.

        Returns:
            The normalized text, or an empty string if the input is
            not a valid string.
        """
        if not isinstance(value, str):
            return ""

        text = unicodedata.normalize("NFC", value)
        text = _QUOTE_PATTERN.sub('"', text)
        text = _APOSTROPHE_PATTERN.sub("'", text)
        text = _WHITESPACE_PATTERN.sub(" ", text)
        text = text.strip()

        return text
