"""Embedding generation service.

This module defines an EmbeddingGenerator class that loads the
pretrained sentence-transformers embedding model configured in
config.py and generates semantic embeddings for normalized review
text, for use by the downstream theme/cluster discovery layer.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer

from config import COLUMNS, DEFAULT_BATCH_SIZE, MODEL_NAMES
from utils.logger import logger


class EmbeddingGenerator:
    """Generates semantic embeddings for review text.

    The embedding model is loaded once at construction time and
    reused for all subsequent calls to ``generate``.
    """

    def __init__(self) -> None:
        """Initialize the generator by loading the embedding model."""
        logger.info(
            "Loading embedding model '%s'.", MODEL_NAMES.EMBEDDING_MODEL
        )
        try:
            self._model = SentenceTransformer(MODEL_NAMES.EMBEDDING_MODEL)
        except Exception:
            logger.exception(
                "Failed to load embedding model '%s'.",
                MODEL_NAMES.EMBEDDING_MODEL,
            )
            raise

        logger.info("Embedding model loaded successfully.")

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for each valid row in the DataFrame.

        Adds or updates the ``embedding`` column on a copy of the
        given DataFrame. Rows where ``is_valid`` is False, or where
        ``normalized_text`` is null, non-string, or empty/whitespace-
        only, are skipped and left unset. No rows are added or
        removed, and row alignment is preserved.

        Args:
            df: DataFrame containing a ``normalized_text`` column and
                optionally an ``is_valid`` column.

        Returns:
            A copy of the given DataFrame, with the ``embedding``
            column added or updated for eligible rows.
        """
        df = df.copy()

        if COLUMNS.NORMALIZED_TEXT not in df.columns:
            logger.error(
                "Missing required column '%s'; cannot generate embeddings.",
                COLUMNS.NORMALIZED_TEXT,
            )
            df[COLUMNS.EMBEDDING] = None
            return df

        if COLUMNS.EMBEDDING not in df.columns:
            df[COLUMNS.EMBEDDING] = None

        if COLUMNS.IS_VALID in df.columns:
            eligible_index = df.index[df[COLUMNS.IS_VALID] == True]  # noqa: E712
        else:
            logger.warning(
                "Column '%s' not found; treating all rows as eligible.",
                COLUMNS.IS_VALID,
            )
            eligible_index = df.index

        eligible_texts = df.loc[eligible_index, COLUMNS.NORMALIZED_TEXT]
        non_empty_mask = eligible_texts.apply(
            lambda value: isinstance(value, str) and value.strip() != ""
        )
        inference_index = eligible_index[non_empty_mask]
        texts = df.loc[inference_index, COLUMNS.NORMALIZED_TEXT].tolist()

        if not texts:
            logger.info("No eligible rows found for embedding generation.")
            return df

        embeddings = self._model.encode(
            texts,
            batch_size=DEFAULT_BATCH_SIZE,
            convert_to_numpy=True,
        )

        for row_index, embedding in zip(inference_index, embeddings):
            df.at[row_index, COLUMNS.EMBEDDING] = embedding

        logger.info("Embedding generation complete for %d rows.", len(texts))

        return df