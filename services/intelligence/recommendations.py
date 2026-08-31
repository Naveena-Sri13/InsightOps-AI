"""Business recommendation derivation service.

This module interprets analytical columns already computed by the
analysis layer, and optional ``BusinessInsights``, to derive structured
evidence-based recommendations. It performs no preprocessing, model
inference, embedding generation, or clustering of its own.
"""

from __future__ import annotations

import logging

import pandas as pd

from config import COLUMNS, EMOTIONS, SENTIMENTS
from schemas.models import BusinessInsights, Recommendation


logger = logging.getLogger(__name__)

_UNKNOWN = "Unknown"

_PRIORITY_HIGH = "High"
_PRIORITY_MEDIUM = "Medium"
_PRIORITY_LOW = "Low"

_CATEGORY_PAIN_POINT = "Pain Point"
_CATEGORY_SENTIMENT = "Sentiment"
_CATEGORY_EMOTION = "Emotion"
_CATEGORY_THEME = "Theme"
_CATEGORY_RATING = "Rating"
_CATEGORY_STRENGTH = "Strength"

_PRIORITY_ORDER = {
    _PRIORITY_HIGH: 0,
    _PRIORITY_MEDIUM: 1,
    _PRIORITY_LOW: 2,
}

# Share thresholds used only to assign priority, not to invent facts.
_NEGATIVE_SHARE_HIGH = 0.40
_NEGATIVE_SHARE_MEDIUM = 0.25
_NEGATIVE_SHARE_LOW = 0.15
_DISTRESS_SHARE_HIGH = 0.20
_DISTRESS_SHARE_MEDIUM = 0.10
_ASPECT_NEGATIVE_SHARE_HIGH = 0.50
_LOW_RATING_HIGH = 3.0
_LOW_RATING_MEDIUM = 3.5
_MAX_ASPECT_RECS = 3
_MAX_THEME_RECS = 2

_DISTRESS_EMOTIONS = (
    EMOTIONS.ANGER,
    EMOTIONS.DISGUST,
    EMOTIONS.FEAR,
    EMOTIONS.SADNESS,
)

_DISTRESS_PRIORITY = {
    EMOTIONS.ANGER: _PRIORITY_HIGH,
    EMOTIONS.DISGUST: _PRIORITY_HIGH,
    EMOTIONS.FEAR: _PRIORITY_MEDIUM,
    EMOTIONS.SADNESS: _PRIORITY_MEDIUM,
}


class RecommendationsGenerator:
    """Derives structured business recommendations from processed reviews."""

    def generate(
        self,
        df: pd.DataFrame,
        insights: BusinessInsights | None = None,
    ) -> list[Recommendation]:
        """Generate evidence-based recommendations from a processed DataFrame.

        The input DataFrame is never mutated. When ``is_valid`` is present,
        only rows where it is ``True`` are considered. Missing analytical
        columns cause the related recommendation rules to be skipped.

        Args:
            df: DataFrame containing analysis-layer columns.
            insights: Optional insights already produced by
                ``InsightsGenerator``. Used as hints and always
                re-checked against the DataFrame.

        Returns:
            A deterministic list of ``Recommendation`` objects. Empty
            when there is no usable evidence.
        """
        if df.empty:
            logger.info(
                "Empty DataFrame received; returning no recommendations."
            )
            return []

        valid_df = self._filter_valid(df)

        if valid_df.empty:
            logger.info(
                "No valid rows found; returning no recommendations."
            )
            return []

        total_rows = len(valid_df)
        recommendations: list[Recommendation] = []

        recommendations.extend(
            self._recommend_negative_aspects(valid_df, total_rows, insights)
        )
        sentiment_rec = self._recommend_negative_sentiment(
            valid_df, total_rows, insights
        )
        if sentiment_rec is not None:
            recommendations.append(sentiment_rec)

        emotion_rec = self._recommend_distress_emotion(
            valid_df, total_rows, insights
        )
        if emotion_rec is not None:
            recommendations.append(emotion_rec)

        recommendations.extend(
            self._recommend_negative_themes(valid_df, total_rows, insights)
        )

        rating_rec = self._recommend_low_rating(valid_df, total_rows)
        if rating_rec is not None:
            recommendations.append(rating_rec)

        strength_rec = self._recommend_positive_strength(
            valid_df, total_rows, insights
        )
        if strength_rec is not None:
            recommendations.append(strength_rec)

        recommendations = self._sort_recommendations(recommendations)

        logger.info(
            "Generated %d recommendation(s) from %d valid rows.",
            len(recommendations),
            total_rows,
        )
        return recommendations

    def _filter_valid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict analysis to valid rows when possible."""
        if COLUMNS.IS_VALID not in df.columns:
            logger.warning(
                "Column '%s' not found; treating all rows as valid.",
                COLUMNS.IS_VALID,
            )
            return df

        return df[df[COLUMNS.IS_VALID] == True]  # noqa: E712

    def _recommend_negative_aspects(
        self,
        df: pd.DataFrame,
        total_rows: int,
        insights: BusinessInsights | None,
    ) -> list[Recommendation]:
        """Recommend action on aspects with evidenced negative sentiment."""
        if COLUMNS.ASPECT not in df.columns:
            logger.warning(
                "Column '%s' not found; skipping aspect recommendations.",
                COLUMNS.ASPECT,
            )
            return []

        negative_df = self._rows_with_sentiment(df, SENTIMENTS.NEGATIVE)
        if negative_df.empty:
            return []

        aspects = self._non_empty_series(negative_df, COLUMNS.ASPECT)
        if aspects.empty:
            return []

        counts = aspects.astype(str).value_counts()
        preferred = self._insight_value(
            insights, "top_negative_aspect"
        )

        ordered_aspects: list[str] = []
        if preferred is not None and preferred in counts.index:
            ordered_aspects.append(preferred)
        for aspect in counts.index.astype(str):
            if aspect not in ordered_aspects:
                ordered_aspects.append(aspect)

        recommendations: list[Recommendation] = []
        for aspect in ordered_aspects[:_MAX_ASPECT_RECS]:
            affected = int(counts[aspect])
            aspect_mentions = self._non_empty_series(
                df, COLUMNS.ASPECT
            ).astype(str)
            mention_count = int((aspect_mentions == aspect).sum())
            negative_share = (
                affected / mention_count if mention_count else 0.0
            )
            priority = (
                _PRIORITY_HIGH
                if (
                    negative_share >= _ASPECT_NEGATIVE_SHARE_HIGH
                    or (affected / total_rows) >= _NEGATIVE_SHARE_MEDIUM
                )
                else _PRIORITY_MEDIUM
            )
            affected_rows = negative_df.loc[
                aspects[aspects.astype(str) == aspect].index
            ]
            confidence = self._confidence(
                affected=affected,
                total=total_rows,
                confidence_series=self._confidence_for_rows(affected_rows),
            )
            recommendations.append(
                Recommendation(
                    priority=priority,
                    category=_CATEGORY_PAIN_POINT,
                    title=f"Address negative feedback on {aspect}",
                    description=(
                        f"{affected} of {total_rows} valid reviews mention "
                        f"'{aspect}' with negative sentiment"
                        f"{self._mention_clause(mention_count, affected)}."
                    ),
                    affected_reviews=affected,
                    confidence=confidence,
                )
            )

        return recommendations

    def _recommend_negative_sentiment(
        self,
        df: pd.DataFrame,
        total_rows: int,
        insights: BusinessInsights | None,
    ) -> Recommendation | None:
        """Recommend action when negative sentiment is materially present."""
        if COLUMNS.SENTIMENT not in df.columns:
            logger.warning(
                "Column '%s' not found; skipping sentiment recommendation.",
                COLUMNS.SENTIMENT,
            )
            return None

        negative_count = int(
            (df[COLUMNS.SENTIMENT] == SENTIMENTS.NEGATIVE).sum()
        )
        if negative_count == 0:
            return None

        share = negative_count / total_rows
        if share < _NEGATIVE_SHARE_LOW:
            overall = self._insight_value(insights, "overall_sentiment")
            if overall != SENTIMENTS.NEGATIVE:
                return None

        if share >= _NEGATIVE_SHARE_HIGH:
            priority = _PRIORITY_HIGH
        elif share >= _NEGATIVE_SHARE_MEDIUM:
            priority = _PRIORITY_MEDIUM
        else:
            priority = _PRIORITY_LOW

        percent = round(share * 100)
        return Recommendation(
            priority=priority,
            category=_CATEGORY_SENTIMENT,
            title="Investigate drivers of negative sentiment",
            description=(
                f"{negative_count} of {total_rows} valid reviews "
                f"({percent}%) are classified as {SENTIMENTS.NEGATIVE}."
            ),
            affected_reviews=negative_count,
            confidence=self._confidence(
                affected=negative_count,
                total=total_rows,
                confidence_series=self._confidence_for_rows(
                    df[df[COLUMNS.SENTIMENT] == SENTIMENTS.NEGATIVE]
                ),
            ),
        )

    def _recommend_distress_emotion(
        self,
        df: pd.DataFrame,
        total_rows: int,
        insights: BusinessInsights | None,
    ) -> Recommendation | None:
        """Recommend action on the strongest evidenced distress emotion."""
        if COLUMNS.EMOTION not in df.columns:
            logger.warning(
                "Column '%s' not found; skipping emotion recommendation.",
                COLUMNS.EMOTION,
            )
            return None

        emotions = self._non_empty_series(df, COLUMNS.EMOTION)
        if emotions.empty:
            return None

        distress_counts = {
            emotion: int((emotions == emotion).sum())
            for emotion in _DISTRESS_EMOTIONS
            if int((emotions == emotion).sum()) > 0
        }
        if not distress_counts:
            return None

        preferred = self._insight_value(insights, "dominant_emotion")
        if preferred in distress_counts:
            selected = preferred
        else:
            selected = max(
                distress_counts,
                key=lambda emotion: (
                    distress_counts[emotion],
                    -_PRIORITY_ORDER.get(
                        _DISTRESS_PRIORITY[emotion], 99
                    ),
                    emotion,
                ),
            )

        affected = distress_counts[selected]
        share = affected / total_rows
        base_priority = _DISTRESS_PRIORITY[selected]
        if share >= _DISTRESS_SHARE_HIGH:
            priority = _PRIORITY_HIGH
        elif (
            share >= _DISTRESS_SHARE_MEDIUM
            and _PRIORITY_ORDER[base_priority]
            <= _PRIORITY_ORDER[_PRIORITY_MEDIUM]
        ):
            priority = base_priority
        else:
            priority = _PRIORITY_LOW
            if base_priority == _PRIORITY_HIGH:
                priority = _PRIORITY_MEDIUM

        percent = round(share * 100)
        return Recommendation(
            priority=priority,
            category=_CATEGORY_EMOTION,
            title=f"Respond to customer {selected.lower()} signals",
            description=(
                f"{affected} of {total_rows} valid reviews ({percent}%) "
                f"express {selected}."
            ),
            affected_reviews=affected,
            confidence=self._confidence(
                affected=affected,
                total=total_rows,
                confidence_series=self._series_if_present(
                    df.loc[emotions[emotions == selected].index],
                    COLUMNS.EMOTION_CONFIDENCE,
                ),
            ),
        )

    def _recommend_negative_themes(
        self,
        df: pd.DataFrame,
        total_rows: int,
        insights: BusinessInsights | None,
    ) -> list[Recommendation]:
        """Recommend review of themes with concentrated negative sentiment."""
        has_cluster_id = COLUMNS.CLUSTER_ID in df.columns
        has_cluster_name = COLUMNS.CLUSTER_NAME in df.columns
        if not has_cluster_id and not has_cluster_name:
            logger.warning(
                "Columns '%s'/'%s' not found; skipping theme recommendations.",
                COLUMNS.CLUSTER_ID,
                COLUMNS.CLUSTER_NAME,
            )
            return []

        if COLUMNS.SENTIMENT not in df.columns:
            return []

        clustered = df
        if has_cluster_id:
            clustered = clustered[clustered[COLUMNS.CLUSTER_ID].notna()]
            clustered = clustered[clustered[COLUMNS.CLUSTER_ID] != -1]

        if clustered.empty:
            return []

        negative = clustered[
            clustered[COLUMNS.SENTIMENT] == SENTIMENTS.NEGATIVE
        ]
        if negative.empty:
            return []

        label_column = (
            COLUMNS.CLUSTER_NAME
            if has_cluster_name
            and self._non_empty_series(negative, COLUMNS.CLUSTER_NAME).shape[0]
            else COLUMNS.CLUSTER_ID
            if has_cluster_id
            else COLUMNS.CLUSTER_NAME
        )
        labels = self._non_empty_series(negative, label_column)
        if labels.empty:
            return []

        counts = labels.astype(str).value_counts()
        preferred = self._insight_value(insights, "top_theme")

        ordered_labels: list[str] = []
        if preferred is not None and preferred in counts.index:
            ordered_labels.append(preferred)
        for label in counts.index.astype(str):
            if label not in ordered_labels:
                ordered_labels.append(label)

        recommendations: list[Recommendation] = []
        for label in ordered_labels[:_MAX_THEME_RECS]:
            affected = int(counts[label])
            recommendations.append(
                Recommendation(
                    priority=(
                        _PRIORITY_HIGH
                        if (affected / total_rows) >= _NEGATIVE_SHARE_MEDIUM
                        else _PRIORITY_MEDIUM
                    ),
                    category=_CATEGORY_THEME,
                    title=f"Review negative feedback in theme {label}",
                    description=(
                        f"{affected} valid reviews in theme '{label}' are "
                        f"classified as {SENTIMENTS.NEGATIVE}."
                    ),
                    affected_reviews=affected,
                    confidence=self._confidence(
                        affected=affected,
                        total=total_rows,
                    ),
                )
            )

        return recommendations

    def _recommend_low_rating(
        self,
        df: pd.DataFrame,
        total_rows: int,
    ) -> Recommendation | None:
        """Recommend action when numeric ratings are evidenced as low."""
        if COLUMNS.RATING not in df.columns:
            logger.warning(
                "Column '%s' not found; skipping rating recommendation.",
                COLUMNS.RATING,
            )
            return None

        ratings = pd.to_numeric(df[COLUMNS.RATING], errors="coerce").dropna()
        if ratings.empty:
            return None

        average_rating = float(ratings.mean())
        if average_rating >= _LOW_RATING_MEDIUM:
            return None

        low_mask = ratings < _LOW_RATING_MEDIUM
        affected = int(low_mask.sum())
        if affected == 0:
            return None

        priority = (
            _PRIORITY_HIGH
            if average_rating < _LOW_RATING_HIGH
            else _PRIORITY_MEDIUM
        )
        return Recommendation(
            priority=priority,
            category=_CATEGORY_RATING,
            title="Improve experience where ratings are below target",
            description=(
                f"Average rating across {len(ratings)} valid reviews is "
                f"{round(average_rating, 2)}; {affected} review(s) are "
                f"below {_LOW_RATING_MEDIUM}."
            ),
            affected_reviews=affected,
            confidence=self._confidence(affected=affected, total=total_rows),
        )

    def _recommend_positive_strength(
        self,
        df: pd.DataFrame,
        total_rows: int,
        insights: BusinessInsights | None,
    ) -> Recommendation | None:
        """Recommend protecting an evidenced positive aspect."""
        if COLUMNS.ASPECT not in df.columns:
            return None

        positive_df = self._rows_with_sentiment(df, SENTIMENTS.POSITIVE)
        if positive_df.empty:
            return None

        aspects = self._non_empty_series(positive_df, COLUMNS.ASPECT)
        if aspects.empty:
            return None

        counts = aspects.astype(str).value_counts()
        preferred = self._insight_value(insights, "top_positive_aspect")
        if preferred is not None and preferred in counts.index:
            aspect = preferred
        else:
            aspect = str(counts.index[0])

        affected = int(counts[aspect])
        if affected == 0:
            return None

        return Recommendation(
            priority=_PRIORITY_LOW,
            category=_CATEGORY_STRENGTH,
            title=f"Protect strengths in {aspect}",
            description=(
                f"{affected} of {total_rows} valid reviews mention "
                f"'{aspect}' with positive sentiment. Preserve this "
                "experience while addressing higher-priority issues."
            ),
            affected_reviews=affected,
            confidence=self._confidence(
                affected=affected,
                total=total_rows,
                confidence_series=self._confidence_for_rows(
                    positive_df.loc[
                        aspects[aspects.astype(str) == aspect].index
                    ]
                ),
            ),
        )

    def _rows_with_sentiment(
        self,
        df: pd.DataFrame,
        sentiment_label: str,
    ) -> pd.DataFrame:
        """Filter rows by aspect sentiment, falling back to review sentiment."""
        if COLUMNS.ASPECT_SENTIMENT in df.columns:
            return df[df[COLUMNS.ASPECT_SENTIMENT] == sentiment_label]

        if COLUMNS.SENTIMENT in df.columns:
            logger.warning(
                "Column '%s' not found; falling back to '%s'.",
                COLUMNS.ASPECT_SENTIMENT,
                COLUMNS.SENTIMENT,
            )
            return df[df[COLUMNS.SENTIMENT] == sentiment_label]

        return df.iloc[0:0]

    def _non_empty_series(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        """Return non-null, non-blank values from a column."""
        if column not in df.columns:
            return pd.Series(dtype="object")

        values = df[column].dropna()
        values = values[values.astype(str).str.strip() != ""]
        values = values[values.astype(str) != _UNKNOWN]
        return values

    def _series_if_present(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> pd.Series | None:
        """Return a numeric confidence series when the column exists."""
        if df.empty or column not in df.columns:
            return None

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            return None
        return values

    def _confidence_for_rows(self, df: pd.DataFrame) -> pd.Series | None:
        """Prefer aspect confidence, then sentiment confidence."""
        aspect_conf = self._series_if_present(df, COLUMNS.ASPECT_CONFIDENCE)
        if aspect_conf is not None:
            return aspect_conf
        return self._series_if_present(df, COLUMNS.SENTIMENT_CONFIDENCE)

    def _confidence(
        self,
        affected: int,
        total: int,
        confidence_series: pd.Series | None = None,
    ) -> float:
        """Combine evidence coverage with available model confidence."""
        if total <= 0 or affected <= 0:
            return 0.0

        coverage = affected / total
        if confidence_series is not None and not confidence_series.empty:
            model_conf = float(confidence_series.mean())
            value = (0.5 * coverage) + (0.5 * model_conf)
        else:
            value = coverage

        return round(min(1.0, max(0.0, value)), 4)

    def _insight_value(
        self,
        insights: BusinessInsights | None,
        field_name: str,
    ) -> str | None:
        """Return a usable insight field, or None when missing/unknown."""
        if insights is None:
            return None

        value = getattr(insights, field_name, None)
        if value is None:
            return None

        text = str(value).strip()
        if text == "" or text == _UNKNOWN:
            return None
        return text

    def _mention_clause(self, mention_count: int, affected: int) -> str:
        """Add mention context when aspect volume differs from negatives."""
        if mention_count <= 0 or mention_count == affected:
            return ""
        return f" ({affected} of {mention_count} mentions of this aspect)"

    def _sort_recommendations(
        self,
        recommendations: list[Recommendation],
    ) -> list[Recommendation]:
        """Return recommendations in a stable, deterministic order."""
        return sorted(
            recommendations,
            key=lambda rec: (
                _PRIORITY_ORDER.get(rec.priority, 99),
                rec.category,
                rec.title,
            ),
        )
