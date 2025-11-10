from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Text transformer compatible with sklearn's get_feature_names_out.
    Always receives 2D input (DataFrame with one column).
    """

    def __init__(
        self,
        strategy: str = "tfidf",
        tfidf_params: Optional[Dict] = None,
        bert_model_name: str = "all-MiniLM-L6-v2",
        length_type: str = "char",
        max_features: Optional[int] = None,
        feature_names_prefix: str = "text",
    ):
        self.strategy = strategy
        self.tfidf_params = tfidf_params or {}
        self.bert_model_name = bert_model_name
        self.length_type = length_type
        self.max_features = max_features
        self.feature_names_prefix = feature_names_prefix

        self._vectorizer = None
        self._bert_model = None
        self._feature_names = None

    def fit(self, X, y=None):
        # X is 2D (DataFrame or array with shape (n, 1))
        if hasattr(X, "iloc"):
            series = X.iloc[:, 0].fillna("").astype(str)
        else:
            series = pd.Series(X.ravel()).fillna("").astype(str)

        if self.strategy == "tfidf":
            params = {"max_features": self.max_features, **self.tfidf_params}
            self._vectorizer = TfidfVectorizer(**params).fit(series)
            self._feature_names = [
                f"{self.feature_names_prefix}_tfidf_{term}"
                for term in sorted(
                    self._vectorizer.vocabulary_, key=self._vectorizer.vocabulary_.get
                )
            ]
        elif self.strategy == "bert":
            self._bert_model = SentenceTransformer(self.bert_model_name)
            dim = self._bert_model.encode(["dummy"]).shape[1]
            self._feature_names = [
                f"{self.feature_names_prefix}_bert_{i}" for i in range(dim)
            ]
        elif self.strategy == "len":
            suffix = "char_len" if self.length_type == "char" else "word_len"
            self._feature_names = [f"{self.feature_names_prefix}_{suffix}"]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            series = X.iloc[:, 0].fillna("").astype(str)
        else:
            series = pd.Series(X.ravel()).fillna("").astype(str)

        if self.strategy == "len":
            values = (
                series.str.len()
                if self.length_type == "char"
                else series.str.split().str.len()
            )
            return values.values.reshape(-1, 1).astype(np.float32)
        elif self.strategy == "tfidf":
            return self._vectorizer.transform(series).toarray()
        elif self.strategy == "bert":
            return self._bert_model.encode(series.tolist(), show_progress_bar=False)
        else:
            raise RuntimeError("Call fit() first")

    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names)
