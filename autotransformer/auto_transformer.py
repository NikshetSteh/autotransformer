from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .text import TextFeatureExtractor
from .utils import detect_column_types


class MLPreprocessor:
    """
    Flexible preprocessor for machine learning competitions.

    Supports automatic or explicit typing of columns into:
        - Numerical (scaled)
        - Categorical (encoded)
        - Text (TF-IDF, BERT, or length-based)
        - Custom (user-provided transformers per column)
        - Targets (with optional per-column transformers)

    Behavior:
        - fit_transform(df) → (X_df, y_df) if target columns are present
        - transform(df) → X_df if no targets in df; (X_df, y_df) if targets present
        - y_df preserves original index and column names (after optional transformation)
    """

    DEFAULT_TEXT_EXTRACTOR_KWARGS = {"strategy": "tfidf"}
    DEFAULT_NUMERIC_SCALER = MinMaxScaler()
    DEFAULT_CAT_ENCODER = OneHotEncoder()

    def __init__(
        self,
        target_cols: Optional[list[str]] = None,
        target_transformers: Optional[dict[str, BaseEstimator]] = None,
        default_target_transformer: Optional[BaseEstimator] = None,
        drop_cols: Optional[list[str]] = None,
        num_cols: Optional[list[str]] = None,
        cat_cols: Optional[list[str]] = None,
        text_cols: Optional[list[str]] = None,
        custom_cols: Optional[dict[str, BaseEstimator]] = None,
        num_scaler: Optional[BaseEstimator] = None,
        cat_encoder: Optional[BaseEstimator] = None,
        text_extractor_kwargs: Optional[dict[str, Any]] = None,
        per_column_transformers: Optional[dict[str, BaseEstimator]] = None,
        auto_detect: bool = True,
        text_heuristic_min_unique_ratio: float = 0.9,
        cat_heuristic_max_unique_ratio: float = 0.01,
        cat_heuristic_max_unique_count: int = 20,
    ):
        """
        Initialize the MLPreprocessor with configurable column typing and transformation rules.

        Args:
            target_cols (Optional[list[str]]):
                List of column names to treat as targets. These are excluded from feature
                preprocessing and optionally transformed.
                Example: `["price", "is_fraud"]`.

            target_transformers (Optional[dict[str, BaseEstimator]]):
                Per-target transformers. Keys are target column names, values are fitted or
                unfitted sklearn-style transformers.
                Example: `{"price": PowerTransformer(), "is_fraud": LabelBinarizer()}`.

            default_target_transformer (Optional[BaseEstimator]):
                Fallback transformer applied to all targets not in `target_transformers`.
                Example: `StandardScaler()` for regression targets.

            drop_cols (Optional[list[str]]):
                Columns to completely ignore (neither features nor targets).
                Example: `["user_id", "timestamp"]`.

            num_cols (Optional[list[str]]):
                Explicitly declare these columns as numerical (overrides auto-detection).
                Example: `["age", "salary"]`.

            cat_cols (Optional[list[str]]):
                Explicitly declare these columns as categorical.
                Example: `["gender", "city"]`.

            text_cols (Optional[list[str]]):
                Explicitly declare these columns as text (for NLP feature extraction).
                Example: `["product_description", "review"]`.

            custom_cols (Optional[dict[str, BaseEstimator]]):
                Columns with fully custom transformers not covered by num/cat/text logic.
                Each transformer must accept a single column and implement `fit`/`transform`.
                Example: `{"url": URLLengthExtractor(), "email": EmailDomainEncoder()}`.

            num_scaler (Optional[BaseEstimator]):
                Transformer for numerical features (applies to auto-detected or forced num cols).
                Default: `MinMaxScaler()`.
                Example: `StandardScaler()` or `RobustScaler()`.

            cat_encoder (Optional[BaseEstimator]):
                Transformer for categorical features.
                Default: `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
                Example: `OrdinalEncoder()` or a custom category embedder.

            text_extractor_kwargs (Optional[dict[str, Any]]):
                Configuration for text feature extraction. Passed to `TextFeatureExtractor`.
                Supported keys: `"strategy"`, `"tfidf_params"`, `"bert_model_name"`, etc.
                Examples:
                    - `{"strategy": "len", "length_type": "word"}`
                    - `{"strategy": "bert", "bert_model_name": "all-mpnet-base-v2"}`
                    - `{"strategy": "tfidf", "max_features": 5000}`

            per_column_transformers (Optional[dict[str, BaseEstimator]]):
                Alternative interface for custom per-column transformers (same as `custom_cols`).
                If both are provided, they are merged (with `per_column_transformers` taking
                precedence in case of key collision during internal resolution).
                Example: `{"description": TfidfVectorizer(max_features=1000)}`.

            auto_detect (bool, default=True):
                Whether to automatically infer column types for columns not explicitly assigned.
                If `False`, only columns in `num_cols`, `cat_cols`, `text_cols`, or custom dicts
                are processed; others are dropped.

            text_heuristic_min_unique_ratio (float, default=0.9):
                Threshold for auto-detecting text: if a non-numeric column has
                `(n_unique / n_total) >= this value`, it is treated as text.
                Higher values → more conservative (fewer text cols).
                Example: `0.95` for stricter text detection.

            cat_heuristic_max_unique_ratio (float, default=0.01):
                For numeric columns: if `(n_unique / n_total) <= this`, treat as categorical.
                Example: A column with 10 unique values in 10,000 rows → 0.001 → categorical.

            cat_heuristic_max_unique_count (int, default=20):
                Numeric or string columns with `n_unique <= this` are treated as categorical,
                regardless of ratio.
                Example: `50` to allow more categorical levels for small datasets.

        Note:
            - `custom_cols` and `per_column_transformers` serve the same purpose;
              the latter is kept for API flexibility.
            - Target transformation is optional and independent of feature preprocessing.
        """
        self.target_cols = list(target_cols) if target_cols else []
        self.target_transformers = target_transformers or {}
        self.default_target_transformer = default_target_transformer
        self.drop_cols = set(drop_cols) if drop_cols else set()
        self.forced_num_cols = set(num_cols) if num_cols else set()
        self.forced_cat_cols = set(cat_cols) if cat_cols else set()
        self.forced_text_cols = set(text_cols) if text_cols else set()
        self.custom_cols = custom_cols or {}
        self.per_column_transformers = per_column_transformers or {}

        self.num_scaler = (
            num_scaler if num_scaler is not None else self.DEFAULT_NUMERIC_SCALER
        )
        self.cat_encoder = (
            cat_encoder if cat_encoder is not None else self.DEFAULT_CAT_ENCODER 
        )
        self.text_extractor_kwargs = (
            text_extractor_kwargs
            if text_extractor_kwargs is not None
            else self.DEFAULT_TEXT_EXTRACTOR_KWARGS.copy()
        )

        self.auto_detect = auto_detect
        self.text_heuristic_min_unique_ratio = text_heuristic_min_unique_ratio
        self.cat_heuristic_max_unique_ratio = cat_heuristic_max_unique_ratio
        self.cat_heuristic_max_unique_count = cat_heuristic_max_unique_count

        self._fitted = False
        self._fitted_target_transformers = {}
        self._detected_num_cols = []
        self._detected_cat_cols = []
        self._detected_text_cols = []

    def _detect_column_types(self, df: pd.DataFrame):
        """Delegate to utility function for column type detection."""
        (
            self._detected_num_cols,
            self._detected_cat_cols,
            self._detected_text_cols,
        ) = detect_column_types(
            df=df,
            drop_cols=self.drop_cols,
            target_cols=set(self.target_cols),
            forced_num_cols=self.forced_num_cols,
            forced_cat_cols=self.forced_cat_cols,
            forced_text_cols=self.forced_text_cols,
            custom_cols=set(self.custom_cols),
            per_column_transformers=set(self.per_column_transformers),
            text_heuristic_min_unique_ratio=self.text_heuristic_min_unique_ratio,
            cat_heuristic_max_unique_ratio=self.cat_heuristic_max_unique_ratio,
            cat_heuristic_max_unique_count=self.cat_heuristic_max_unique_count,
        )

    def fit(self, df: pd.DataFrame, y=None):
        """Fit the preprocessor on input DataFrame."""
        if self.auto_detect:
            self._detect_column_types(df)

        missing_targets = [col for col in self.target_cols if col not in df.columns]
        if missing_targets:
            raise ValueError(f"Target columns not found in DataFrame: {missing_targets}")

        self._fitted_target_transformers = {}
        for target in self.target_cols:
            y_series = df[target]
            transformer = (
                self.target_transformers.get(target) or self.default_target_transformer
            )
            if transformer is not None:
                self._fitted_target_transformers[target] = transformer.fit(
                    y_series.values.reshape(-1, 1)
                )
            else:
                self._fitted_target_transformers[target] = None

        transformers = []
        all_custom = set(self.custom_cols) | set(self.per_column_transformers)

        for col in all_custom:
            if col in df.columns and col not in self.target_cols and col not in self.drop_cols:
                trans = self.per_column_transformers.get(col) or self.custom_cols.get(col)
                transformers.append((f"{col}", trans, [col]))

        for col in self._detected_num_cols:
            if col not in all_custom and col not in self.target_cols and col not in self.drop_cols:
                transformers.append((f"{col}", self.num_scaler, [col]))

        for col in self._detected_cat_cols:
            if col not in all_custom and col not in self.target_cols and col not in self.drop_cols:
                transformers.append((f"{col}", self.cat_encoder, [col]))

        for col in self._detected_text_cols:
            if col not in all_custom and col not in self.target_cols and col not in self.drop_cols:
                kwargs = self.text_extractor_kwargs.copy()
                kwargs["feature_names_prefix"] = col
                extractor = TextFeatureExtractor(**kwargs)
                transformers.append((f"{col}", extractor, [col]))

        self._feature_preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            n_jobs=-1,
            verbose_feature_names_out=False,
        )
        self._feature_preprocessor.fit(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        """Transform the input DataFrame using fitted components."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        X_array = self._feature_preprocessor.transform(df)
        if hasattr(X_array, "toarray"):  # sparse matrix
            X_array = X_array.toarray()
        feature_names = self._feature_preprocessor.get_feature_names_out()
        X_df = pd.DataFrame(X_array, columns=feature_names, index=df.index)

        present_targets = [col for col in self.target_cols if col in df.columns]
        if not present_targets:
            return X_df

        y_data = {}
        for target in present_targets:
            y_raw = df[target]
            transformer = self._fitted_target_transformers[target]
            if transformer is not None:
                y_trans = transformer.transform(y_raw.values.reshape(-1, 1))
                if y_trans.ndim == 1:
                    y_trans = y_trans.reshape(-1, 1)
                y_data[target] = y_trans
            else:
                y_data[target] = y_raw.values.reshape(-1, 1)

        y_arrays = [y_data[col] for col in present_targets]
        y_combined = np.hstack(y_arrays)
        y_df = pd.DataFrame(y_combined, columns=present_targets, index=df.index)

        return X_df, y_df

    def fit_transform(self, df: pd.DataFrame):
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_names_out(self):
        """Return output feature names after transformation."""
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        return self._feature_preprocessor.get_feature_names_out().tolist()

    def save(self, path: str):
        """Save the fitted preprocessor to disk."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """Load a preprocessor from disk."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj
