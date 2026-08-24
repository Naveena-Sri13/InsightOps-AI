"""Aspect extraction service.

This module defines an AspectExtractor class that uses KeyBERT with a
sentence-transformers embedding model to extract the primary semantic
aspect discussed in each review, and pairs it with the review's
existing sentiment.
"""

import logging

import pandas as pd
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from config import COLUMNS, DEFAULT_BATCH_SIZE, MODEL_NAMES

logger = logging.getLogger(__name__)

_KEYPHRASE_NGRAM_RANGE = (1, 2)
_STOP_WORDS = "english"
_TOP_N = 1


class AspectExtractor:
    """Extracts the primary aspect discussed in each review using KeyBERT.

    The embedding model is loaded once at construction time and reused
    for all subsequent calls to ``extract``.
    """

    def __init__(self) -> None:
        """Initialize the extractor by loading the embedding model."""
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Loading aspect embedding model '%s' on device '%s'.",
            MODEL_NAMES.EMBEDDING_MODEL,
            self._device,
        )
        try:
            sentence_model = SentenceTransformer(
                MODEL_NAMES.EMBEDDING_MODEL, device=str(self._device)
            )
            self._model = KeyBERT(model=sentence_model)
        except Exception:
            logger.exception(
                "Failed to load aspect embedding model '%s'.",
                MODEL_NAMES.EMBEDDING_MODEL,
            )
            raise

        logger.info("Aspect embedding model loaded successfully.")

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the primary aspect for each valid row in the DataFrame.

        Adds or updates the ``aspect``, ``aspect_sentiment``, and
        ``aspect_confidence`` columns on the given DataFrame. Rows
        where ``is_valid`` is False, or where ``normalized_text`` is
        empty, are assigned ``aspect=None``, ``aspect_sentiment=None``,
        and ``aspect_confidence=0.0``. The original DataFrame is
        preserved; no rows are added or removed.

        Args:
            df: DataFrame containing ``normalized_text``, ``sentiment``,
                ``sentiment_confidence``, and ``is_valid`` columns.

        Returns:
            The same DataFrame, with aspect columns added or updated.
        """
        df = df.copy()

        required_columns = (COLUMNS.NORMALIZED_TEXT, COLUMNS.SENTIMENT)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(
                "Missing required column(s) %s; cannot extract aspects.",
                missing_columns,
            )
            df[COLUMNS.ASPECT] = None
            df[COLUMNS.ASPECT_SENTIMENT] = None
            df[COLUMNS.ASPECT_CONFIDENCE] = 0.0
            return df

        df[COLUMNS.ASPECT] = None
        df[COLUMNS.ASPECT_SENTIMENT] = None
        df[COLUMNS.ASPECT_CONFIDENCE] = 0.0

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

        aspects: list[str | None] = []
        confidences: list[float] = []

        total_batches = (
            (len(texts) + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE
            if texts
            else 0
        )

        for batch_number, start in enumerate(
            range(0, len(texts), DEFAULT_BATCH_SIZE), start=1
        ):
            batch_texts = texts[start:start + DEFAULT_BATCH_SIZE]

            logger.info(
                "Processing aspect extraction batch %d/%d (%d rows).",
                batch_number,
                total_batches,
                len(batch_texts),
            )

            batch_results = self._extract_batch(batch_texts)
            for aspect, confidence in batch_results:
                aspects.append(aspect)
                confidences.append(confidence)

        df.loc[inference_index, COLUMNS.ASPECT] = aspects
        df.loc[inference_index,COLUMNS.ASPECT_CONFIDENCE ] = confidences
        df.loc[inference_index, COLUMNS.ASPECT_SENTIMENT] = df.loc[
            inference_index, COLUMNS.SENTIMENT
        ]

        logger.info("Aspect extraction complete for %d rows.", len(texts))

        return df

    def _extract_batch(
        self, texts: list[str]
    ) -> list[tuple[str | None, float]]:
        """Run keyword/aspect extraction on a batch of texts.

        Args:
            texts: List of normalized text strings.

        Returns:
            A list of (aspect, aspect_confidence) tuples, one per
            input text. If no keyword can be extracted for a text,
            (None, 0.0) is returned for that text.
        """
        results: list[tuple[str | None, float]] = []

        for text in texts:
            try:
                keywords = self._model.extract_keywords(
                    text,
                    keyphrase_ngram_range=_KEYPHRASE_NGRAM_RANGE,
                    stop_words=_STOP_WORDS,
                    top_n=_TOP_N,
                )
            except Exception:
                logger.exception("Failed to extract aspect for a row.")
                results.append((None, 0.0))
                continue

            if not keywords:
                results.append((None, 0.0))
                continue

            aspect, score = keywords[0]
            results.append((aspect, float(score)))

        return results
