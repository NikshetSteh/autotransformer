from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class MLPreprocessor:
    """
    Preprocessor for ML competitions.

    Returns:
      - fit_transform(df) → (X_df, y_df) if target_cols are specified and present
      - transform(df) → X_df (if no targets in df), or (X_df, y_df) if targets present
      - y_df preserves original index and column names (after optional transformation)
    """

    DEFAULT_NUM_SCALER = MinMaxScaler()
    DEFAULT_CAT_ENCODER = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    DEFAULT_TEXT_EXTRACTOR_KWARGS = {"strategy": "tfidf"}

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
            num_scaler if num_scaler is not None else self.DEFAULT_NUM_SCALER
        )
        self.cat_encoder = (
            cat_encoder if cat_encoder is not None else self.DEFAULT_CAT_ENCODER
        )
        self.text_extractor_kwargs = (
            text_extractor_kwargs
            if text_extractor_kwargs is not None
            else self.DEFAULT_TEXT_EXTRACTOR_KWARGS
        )

        self.auto_detect = auto_detect
        self.text_heuristic_min_unique_ratio = text_heuristic_min_unique_ratio
        self.cat_heuristic_max_unique_ratio = cat_heuristic_max_unique_ratio
        self.cat_heuristic_max_unique_count = cat_heuristic_max_unique_count

        self._fitted = False
        self._fitted_target_transformers = {}

    def _detect_column_types(self, df: pd.DataFrame):
        exclude = self.drop_cols | set(self.target_cols)
        cols_to_consider = [col for col in df.columns if col not in exclude]

        num_cols, cat_cols, text_cols = [], [], []

        for col in cols_to_consider:
            if col in self.forced_num_cols:
                num_cols.append(col)
                continue
            if col in self.forced_cat_cols:
                cat_cols.append(col)
                continue
            if col in self.forced_text_cols:
                text_cols.append(col)
                continue
            if col in self.custom_cols or col in self.per_column_transformers:
                continue

            series = df[col]
            n_total = len(series)
            if n_total == 0:
                continue
            n_unique = series.nunique(dropna=True)

            if pd.api.types.is_numeric_dtype(series):
                if n_unique <= 1:
                    num_cols.append(col)
                elif (n_unique / n_total <= self.cat_heuristic_max_unique_ratio) or (
                    n_unique <= self.cat_heuristic_max_unique_count
                ):
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
            else:
                if (n_unique / n_total) >= self.text_heuristic_min_unique_ratio:
                    text_cols.append(col)
                else:
                    cat_cols.append(col)

        self._detected_num_cols = num_cols
        self._detected_cat_cols = cat_cols
        self._detected_text_cols = text_cols

    def fit(self, df: pd.DataFrame, y=None):
        if self.auto_detect:
            self._detect_column_types(df)

        # Validate all target cols exist in df
        missing_targets = [col for col in self.target_cols if col not in df.columns]
        if missing_targets:
            raise ValueError(
                f"Target columns not found in DataFrame: {missing_targets}"
            )

        # Fit target transformers
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

        # Build feature transformers
        transformers = []
        all_custom = set(self.custom_cols) | set(self.per_column_transformers)

        for col in all_custom:
            if (
                col in df.columns
                and col not in self.target_cols
                and col not in self.drop_cols
            ):
                trans = self.per_column_transformers.get(col) or self.custom_cols.get(
                    col
                )
                transformers.append((f"{col}", trans, [col]))

        for col in self._detected_num_cols:
            if (
                col not in all_custom
                and col not in self.target_cols
                and col not in self.drop_cols
            ):
                transformers.append((f"{col}", self.num_scaler, [col]))

        for col in self._detected_cat_cols:
            if (
                col not in all_custom
                and col not in self.target_cols
                and col not in self.drop_cols
            ):
                transformers.append((f"{col}", self.cat_encoder, [col]))

        for col in self._detected_text_cols:
            if (
                col not in all_custom
                and col not in self.target_cols
                and col not in self.drop_cols
            ):
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
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Transform features
        X_array = self._feature_preprocessor.transform(df)
        feature_names = self._feature_preprocessor.get_feature_names_out()
        X_df = pd.DataFrame(X_array, columns=feature_names, index=df.index)

        # Transform targets (if any specified AND present in df)
        present_targets = [col for col in self.target_cols if col in df.columns]
        if not present_targets:
            return X_df

        y_data = {}
        for target in present_targets:
            y_raw = df[target]
            transformer = self._fitted_target_transformers[target]
            if transformer is not None:
                y_trans = transformer.transform(y_raw.values.reshape(-1, 1))
                # Ensure 2D for DataFrame construction
                if y_trans.ndim == 1:
                    y_trans = y_trans.reshape(-1, 1)
                y_data[target] = y_trans
            else:
                y_data[target] = y_raw.values.reshape(-1, 1)

        # Stack into DataFrame
        y_arrays = [y_data[col] for col in present_targets]
        y_combined = np.hstack(y_arrays)
        y_df = pd.DataFrame(y_combined, columns=present_targets, index=df.index)

        return X_df, y_df

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def get_feature_names_out(self):
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        return self._feature_preprocessor.get_feature_names_out().tolist()

    def save(self, path: str):
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj
