"""Semantic cluster discovery service.

This module defines a Clusterer class that groups reviews with
semantically similar embeddings using DBSCAN, producing a cluster_id
per row for consumption by a downstream theme/intelligence layer.
This module performs only cluster discovery: it does not name
clusters, generate insights, or make business interpretations.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from config import COLUMNS

logger = logging.getLogger(__name__)

# DBSCAN parameters. Kept explicit and easy to tune.
_DBSCAN_EPS = 0.3
_DBSCAN_MIN_SAMPLES = 3
_DBSCAN_METRIC = "cosine"

_NOISE_LABEL = -1


class Clusterer:
    """Discovers semantic clusters among review embeddings using DBSCAN.

    DBSCAN is used because it does not require a fixed number of
    clusters to be specified in advance and can identify noise points
    (outliers) using the standard label -1.
    """

    def __init__(
        self,
        eps: float = _DBSCAN_EPS,
        min_samples: int = _DBSCAN_MIN_SAMPLES,
        metric: str = _DBSCAN_METRIC,
    ) -> None:
        """Initialize the clusterer with explicit DBSCAN parameters.

        Args:
            eps: Maximum cosine distance between two samples for one
                to be considered in the neighborhood of the other.
            min_samples: Number of samples required in a neighborhood
                for a point to be considered a core point.
            metric: Distance metric passed to DBSCAN. Cosine distance
                is preferred since embeddings represent semantic
                similarity.
        """
        self._eps = eps
        self._min_samples = min_samples
        self._metric = metric

    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign a cluster_id to rows with usable embeddings.

        Adds or updates the ``cluster_id`` column on a copy of the
        given DataFrame. Rows that are invalid, missing an embedding,
        or have a malformed or inconsistently-shaped embedding are
        left with ``cluster_id=None``. Noise points identified by
        DBSCAN are assigned ``cluster_id=-1``. Row alignment and the
        original row order are preserved.

        Args:
            df: DataFrame containing an ``embedding`` column and
                optionally an ``is_valid`` column.

        Returns:
            A copy of the given DataFrame, with the ``cluster_id``
            column added or updated.
        """
        df = df.copy()

        if COLUMNS.EMBEDDING not in df.columns:
            logger.error(
                "Missing required column '%s'; cannot perform clustering.",
                COLUMNS.EMBEDDING,
            )
            df[COLUMNS.CLUSTER_ID] = None
            return df

        df[COLUMNS.CLUSTER_ID] = None

        if df.empty:
            logger.info("Empty DataFrame received; skipping clustering.")
            return df

        if COLUMNS.IS_VALID in df.columns:
            eligible_index = df.index[df[COLUMNS.IS_VALID] == True]  # noqa: E712
        else:
            logger.warning(
                "Column '%s' not found; treating all rows as eligible.",
                COLUMNS.IS_VALID,
            )
            eligible_index = df.index

        row_indices, embedding_vectors = self._collect_usable_embeddings(
            df, eligible_index
        )

        if not row_indices:
            logger.info("No usable embeddings found; skipping clustering.")
            return df

        embedding_matrix = np.vstack(embedding_vectors)

        logger.info(
            "Running DBSCAN clustering on %d usable embeddings "
            "(eps=%s, min_samples=%s, metric=%s).",
            len(row_indices),
            self._eps,
            self._min_samples,
            self._metric,
        )

        try:
            labels = DBSCAN(
                eps=self._eps,
                min_samples=self._min_samples,
                metric=self._metric,
            ).fit_predict(embedding_matrix)
        except Exception:
            logger.exception("DBSCAN clustering failed.")
            return df

        noise_count = int(np.sum(labels == _NOISE_LABEL))
        cluster_count = len(set(labels.tolist()) - {_NOISE_LABEL})
        logger.info(
            "Clustering complete: %d clusters discovered, %d noise points.",
            cluster_count,
            noise_count,
        )

        for row_index, label in zip(row_indices, labels):
            df.at[row_index, COLUMNS.CLUSTER_ID] = int(label)

        return df

    def _collect_usable_embeddings(
        self, df: pd.DataFrame, eligible_index: pd.Index
    ) -> tuple[list, list[np.ndarray]]:
        """Collect valid, consistently-shaped embeddings for clustering.

        Rows with missing, null, malformed, empty, or non-numeric
        embeddings are excluded. Rows whose embedding dimensionality
        does not match the most common dimensionality among eligible
        rows are also excluded, since DBSCAN requires a consistent
        vector shape.

        Args:
            df: DataFrame containing the embedding column.
            eligible_index: Index of rows eligible for clustering
                (e.g., valid rows).

        Returns:
            A tuple of (row_indices, embedding_vectors), where
            row_indices is a list of DataFrame index labels and
            embedding_vectors is a parallel list of 1D numpy arrays,
            all of the same length.
        """
        candidates: list[tuple[object, np.ndarray]] = []

        for row_index in eligible_index:
            value = df.at[row_index, COLUMNS.EMBEDDING]
            vector = self._to_numeric_vector(value)
            if vector is not None:
                candidates.append((row_index, vector))

        if not candidates:
            return [], []

        lengths = [vector.shape[0] for _, vector in candidates]
        expected_length = max(set(lengths), key=lengths.count)

        row_indices: list = []
        embedding_vectors: list[np.ndarray] = []
        skipped = 0

        for row_index, vector in candidates:
            if vector.shape[0] != expected_length:
                skipped += 1
                continue
            row_indices.append(row_index)
            embedding_vectors.append(vector)

        if skipped:
            logger.warning(
                "Skipped %d row(s) with inconsistent embedding dimensions.",
                skipped,
            )

        return row_indices, embedding_vectors

    def _to_numeric_vector(self, value: object) -> np.ndarray | None:
        """Attempt to convert an embedding cell to a 1D numeric array.

        Args:
            value: The raw value found in the embedding column.

        Returns:
            A 1D numpy array of floats if the value is a valid, non-
            empty, finite numeric vector; otherwise None.
        """
        if value is None:
            return None

        if isinstance(value, float) and np.isnan(value):
            return None

        if not isinstance(value, (np.ndarray, list, tuple)):
            return None

        try:
            vector = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None

        if vector.size == 0:
            return None

        if not np.all(np.isfinite(vector)):
            return None

        return vector