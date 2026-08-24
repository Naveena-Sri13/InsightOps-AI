"""Emotion prediction service.

This module defines an EmotionAnalyzer class that loads the pretrained
Hugging Face emotion model configured in config.py and predicts the
dominant emotion and its confidence for normalized review text using
batch inference.
"""

import logging

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import COLUMNS, DEFAULT_BATCH_SIZE, EMOTIONS, MODEL_NAMES

logger = logging.getLogger(__name__)

_MAX_SEQUENCE_LENGTH = 512

_LABEL_MAP = {
    "joy": EMOTIONS.JOY,
    "sadness": EMOTIONS.SADNESS,
    "anger": EMOTIONS.ANGER,
    "fear": EMOTIONS.FEAR,
    "surprise": EMOTIONS.SURPRISE,
    "disgust": EMOTIONS.DISGUST,
}


class EmotionAnalyzer:
    """Predicts the dominant emotion for review text using a pretrained model.

    The tokenizer and model are loaded once at construction time and
    reused for all subsequent calls to ``analyze``. Only the emotions
    defined in config.py are considered when selecting the dominant
    emotion.
    """

    def __init__(self) -> None:
        """Initialize the analyzer by loading the tokenizer and model."""
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(
            "Loading emotion model '%s' on device '%s'.",
            MODEL_NAMES.EMOTION_MODEL,
            self._device,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAMES.EMOTION_MODEL
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAMES.EMOTION_MODEL
            )
        except Exception:
            logger.exception(
                "Failed to load emotion model '%s'.",
                MODEL_NAMES.EMOTION_MODEL,
            )
            raise

        self._model.to(self._device)
        self._model.eval()
        self._id2label = {
            idx: str(label).lower()
            for idx, label in self._model.config.id2label.items()
        }
        logger.info("Emotion model loaded successfully.")

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the dominant emotion for each valid row in the DataFrame.

        Adds or updates the ``emotion`` and ``emotion_confidence``
        columns on the given DataFrame. Rows where ``is_valid`` is
        False, or where ``normalized_text`` is empty, are skipped and
        left unset. The original DataFrame is preserved and returned;
        no rows are added or removed.

        Args:
            df: DataFrame containing a ``normalized_text`` column and
                optionally an ``is_valid`` column.

        Returns:
            The same DataFrame, with emotion columns added or updated
            for eligible rows.
        """
        df = df.copy()

        if COLUMNS.NORMALIZED_TEXT not in df.columns:
            logger.error(
                "Missing required column '%s'; cannot analyze emotion.",
                COLUMNS.NORMALIZED_TEXT,
            )
            df[COLUMNS.EMOTION] = None
            df[COLUMNS.EMOTION_CONFIDENCE] = 0.0
            return df

        if COLUMNS.EMOTION not in df.columns:
            df[COLUMNS.EMOTION] = None
        if COLUMNS.EMOTION_CONFIDENCE not in df.columns:
            df[COLUMNS.EMOTION_CONFIDENCE] = 0.0

        if COLUMNS.IS_VALID in df.columns:
            eligible_index = df.index[df[COLUMNS.IS_VALID]]  
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
            logger.info("No valid rows available for emotion analysis.")
            return df

        emotions: list[str | None] = []
        confidences: list[float] = []

        total_batches = (
            len(texts) + DEFAULT_BATCH_SIZE - 1
        ) // DEFAULT_BATCH_SIZE

        for batch_number, start in enumerate(
            range(0, len(texts), DEFAULT_BATCH_SIZE),
            start=1,
        ):
            logger.info(
                "Processing emotion batch %d/%d...",
                batch_number,
                total_batches,
            )
            batch_texts = texts[start:start + DEFAULT_BATCH_SIZE]
            batch_results = self._predict_batch(batch_texts)

            for label, confidence in batch_results:
                emotions.append(label)
                confidences.append(confidence)

        df.loc[inference_index, COLUMNS.EMOTION] = emotions
        df.loc[inference_index, COLUMNS.EMOTION_CONFIDENCE] = confidences

        logger.info("Emotion analysis complete for %d rows.", len(texts))

        return df

    def _predict_batch(self, texts: list[str]) -> list[tuple[str | None, float]]:
        """Run batch inference on a list of texts.

        Args:
            texts: List of normalized text strings.

        Returns:
            A list of (emotion_label, emotion_confidence) tuples, one
            per input text.
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

        results: list[tuple[str | None, float]] = []
        for row in probabilities:
            label_scores = {
                self._id2label[idx]: float(prob)
                for idx, prob in enumerate(row.tolist())
            }

            mapped_scores = {
                mapped_label: label_scores[raw_label]
                for raw_label, mapped_label in _LABEL_MAP.items()
                if raw_label in label_scores
            }

            if not mapped_scores:
                results.append((None, 0.0))
                continue

            predicted_label = max(mapped_scores, key=mapped_scores.get)
            confidence = mapped_scores[predicted_label]

            results.append((predicted_label, confidence))

        return results
