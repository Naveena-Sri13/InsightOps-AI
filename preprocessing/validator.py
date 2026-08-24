"""Validation for uploaded review data.

This module defines a Validator class that checks each row of an
uploaded DataFrame against a fixed set of validation rules and
annotates the DataFrame with the results. No rows are removed, and no
text cleaning, normalization, or language detection is performed here.
"""

import logging

import pandas as pd

from config import COLUMNS, THRESHOLDS

logger = logging.getLogger(__name__)


class Validator:
    """Validates uploaded review data prior to preprocessing.

    The validator checks for the presence, non-nullness, and minimum
    length of the source text column, and records the outcome for
    each row without modifying or removing any rows.
    """

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate a DataFrame of uploaded reviews.

        Adds or updates the ``is_valid`` and ``validation_error``
        columns on the given DataFrame. The ``source_text`` column is
        required; if it is missing, every row is marked invalid with
        an appropriate error message. Rating is optional and is not
        validated.

        Args:
            df: DataFrame containing uploaded review data.

        Returns:
            The same DataFrame, with ``is_valid`` and
            ``validation_error`` columns added or updated. No rows are
            added or removed.
        """
        if COLUMNS.SOURCE_TEXT not in df.columns:
            logger.error(
                "Missing required column '%s' in uploaded data.",
                COLUMNS.SOURCE_TEXT,
            )
            df[COLUMNS.IS_VALID] = False
            df[COLUMNS.VALIDATION_ERROR] = (
                f"Missing required column: {COLUMNS.SOURCE_TEXT}"
            )
            return df

        is_valid_flags: list[bool] = []
        validation_errors: list[str | None] = []

        for value in df[COLUMNS.SOURCE_TEXT]:
            is_valid, error = self._validate_source_text(value)
            is_valid_flags.append(is_valid)
            validation_errors.append(error)

        df[COLUMNS.IS_VALID] = is_valid_flags
        df[COLUMNS.VALIDATION_ERROR] = validation_errors

        invalid_count = sum(1 for flag in is_valid_flags if not flag)
        logger.info(
            "Validation complete: %d valid, %d invalid rows out of %d.",
            len(df) - invalid_count,
            invalid_count,
            len(df),
        )

        return df

    def _validate_source_text(self, value: object) -> tuple[bool, str | None]:
        """Validate a single source_text value.

        Args:
            value: The raw value found in the source_text column.

        Returns:
            A tuple of (is_valid, validation_error). validation_error
            is None when is_valid is True.
        """
        if pd.isna(value):
            return False, "source_text is null."

        if not isinstance(value, str):
            return False, "source_text is not a string."

        if value == "":
            return False, "source_text is empty."

        if value.strip() == "":
            return False, "source_text contains only whitespace."

        if len(value.strip()) < THRESHOLDS.MINIMUM_REVIEW_LENGTH:
            return False, (
                "source_text is shorter than the minimum review length "
                f"of {THRESHOLDS.MINIMUM_REVIEW_LENGTH} characters."
            )

        return True, None
