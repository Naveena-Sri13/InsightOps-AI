"""Sentiment prediction service.

This module defines a SentimentAnalyzer class that loads the
pretrained Hugging Face sentiment model configured in config.py and
predicts sentiment labels, confidences, and scores for normalized
review text using batch inference.
"""

import logging

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import COLUMNS, DEFAULT_BATCH_SIZE, MODEL_NAMES, SENTIMENTS

logger = logging.getLogger(__name__)

_MAX_SEQUENCE_LENGTH = 512

_LABEL_MAP = {
    "positive": SENTIMENTS.POSITIVE,
    "neutral": SENTIMENTS.NEUTRAL,
    "negative": SENTIMENTS.NEGATIVE,
    "label_0": SENTIMENTS.NEGATIVE,
    "label_1": SENTIMENTS.NEUTRAL,
    "label_2": SENTIMENTS.POSITIVE,
}


class SentimentAnalyzer:
    """Predicts sentiment for review text using a pretrained model.

    The tokenizer and model are loaded once at construction time and
    reused for all subsequent calls to ``analyze``.
    """

    def __init__(self) -> None:
        """Initialize the analyzer by loading the tokenizer and model."""
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Loading sentiment model '%s' on device '%s'.",
            MODEL_NAMES.SENTIMENT_MODEL,
            self._device,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAMES.SENTIMENT_MODEL
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAMES.SENTIMENT_MODEL
            )
        except Exception:
            logger.exception(
                "Failed to load sentiment model '%s'.",
                MODEL_NAMES.SENTIMENT_MODEL,
            )
            raise

        self._model.to(self._device)
        self._model.eval()
        self._id2label = {
            idx: str(label).lower()
            for idx, label in self._model.config.id2label.items()
        }
        logger.info("Sentiment model loaded successfully.")

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict sentiment for each valid row in the DataFrame.

        Adds or updates the ``sentiment``, ``sentiment_confidence``,
        and ``sentiment_score`` columns on the given DataFrame. Rows
        where ``is_valid`` is False are skipped and left unset. The
        original DataFrame is preserved and returned; no rows are
        added or removed.

        Args:
            df: DataFrame containing a ``normalized_text`` column and
                optionally an ``is_valid`` column.

        Returns:
            The same DataFrame, with sentiment columns added or
            updated for eligible rows.
        """
        df = df.copy()

        if COLUMNS.NORMALIZED_TEXT not in df.columns:
            logger.error(
                "Missing required column '%s'; cannot analyze sentiment.",
                COLUMNS.NORMALIZED_TEXT,
            )
            df[COLUMNS.SENTIMENT] = None
            df[COLUMNS.SENTIMENT_CONFIDENCE] = None
            df[COLUMNS.SENTIMENT_SCORE] = None
            return df

        df[COLUMNS.SENTIMENT] = SENTIMENTS.NEUTRAL
        df[COLUMNS.SENTIMENT_CONFIDENCE] = 0.0
        df[COLUMNS.SENTIMENT_SCORE] = 0.0

        if COLUMNS.IS_VALID in df.columns:
            eligible_index = df.index[df[COLUMNS.IS_VALID]]  # noqa: E712
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

        sentiments: list[str] = []
        confidences: list[float] = []
        scores: list[float] = []

        total_batches = (
    len(texts) + DEFAULT_BATCH_SIZE - 1
) // DEFAULT_BATCH_SIZE

        for batch_number, start in enumerate(
    range(0, len(texts), DEFAULT_BATCH_SIZE),
    start=1,
):
            logger.info(
        "Processing sentiment batch %d/%d",
        batch_number,
        total_batches,
    )
            batch_texts = texts[start:start + DEFAULT_BATCH_SIZE]
            batch_texts = [
        text if isinstance(text, str) else ""
        for text in batch_texts
    ]

            batch_results = self._predict_batch(batch_texts)

            for label, confidence, score in batch_results:
                sentiments.append(label)
                confidences.append(confidence)
                scores.append(score)

        df.loc[inference_index, COLUMNS.SENTIMENT] = sentiments
        df.loc[inference_index, COLUMNS.SENTIMENT_CONFIDENCE] = confidences
        df.loc[inference_index, COLUMNS.SENTIMENT_SCORE] = scores

        logger.info("Sentiment analysis complete for %d rows.", len(texts))

        return df

    def _predict_batch(
        self, texts: list[str]
    ) -> list[tuple[str, float, float]]:
        """Run batch inference on a list of texts.

        Args:
            texts: List of normalized text strings.

        Returns:
            A list of (sentiment_label, sentiment_confidence,
            sentiment_score) tuples, one per input text.
        """
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=_MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: tensor.to(self._device) for key, tensor in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)

        results: list[tuple[str, float, float]] = []
        for row in probabilities:
            label_scores = {
                self._id2label[idx]: float(prob)
                for idx, prob in enumerate(row.tolist())
            }

            mapped_scores = {
                SENTIMENTS.POSITIVE: 0.0,
                SENTIMENTS.NEUTRAL: 0.0,
                SENTIMENTS.NEGATIVE: 0.0,
            }
            for raw_label, prob in label_scores.items():
                mapped_label = _LABEL_MAP.get(raw_label)
                if mapped_label is not None:
                    mapped_scores[mapped_label] = prob

            predicted_label = max(mapped_scores, key=mapped_scores.get)
            confidence = mapped_scores[predicted_label]

            if predicted_label == SENTIMENTS.POSITIVE:
                sentiment_score = mapped_scores[SENTIMENTS.POSITIVE]
            elif predicted_label == SENTIMENTS.NEGATIVE:
                sentiment_score = -mapped_scores[SENTIMENTS.NEGATIVE]
            else:
                sentiment_score = 0.0

            results.append((predicted_label, confidence, sentiment_score))

        return results
