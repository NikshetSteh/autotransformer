from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract features from text columns using configurable strategies.

    Supported strategies:
        - 'tfidf': TF-IDF vectorization.
        - 'bert': Sentence-BERT embeddings.
        - 'len': Length-based features (character or word count).

    Always accepts 2D input (e.g., single-column DataFrame) and outputs a 2D array
    compatible with scikit-learn pipelines.
    """

    def __init__(
        self,
        strategy: str = "tfidf",
        tfidf_params: Optional[Dict[str, Any]] = None,
        bert_model_name: str = "all-MiniLM-L6-v2",
        length_type: str = "char",
        max_features: Optional[int] = None,
        feature_names_prefix: str = "text",
    ):
        """
        Initialize the text feature extractor with the specified strategy and parameters.

        Args:
            strategy (str, default="tfidf"):
                Feature extraction strategy. Must be one of:
                    - `"tfidf"`: Term Frequency–Inverse Document Frequency.
                    - `"bert"`: Dense sentence embeddings using Sentence-BERT.
                    - `"len"`: Scalar feature representing text length.
                Example: `strategy="bert"` for semantic embeddings.

            tfidf_params (Optional[Dict[str, Any]], default=None):
                Additional keyword arguments passed directly to `sklearn.feature_extraction.text.TfidfVectorizer`.
                These override the default behavior (e.g., `ngram_range`, `stop_words`, `lowercase`).
                Note: `max_features` from the top-level argument takes precedence if also provided here.
                Example: `tfidf_params={"ngram_range": (1, 2), "stop_words": "english"}`.

            bert_model_name (str, default="all-MiniLM-L6-v2"):
                Name of the pre-trained Sentence-BERT model from Hugging Face (used only if `strategy="bert"`).
                Must be compatible with `sentence-transformers`. Common choices include:
                    - `"all-MiniLM-L6-v2"` (fast, good general-purpose)
                    - `"all-mpnet-base-v2"` (higher quality, slower)
                    - `"paraphrase-multilingual-MiniLM-L12-v2"` (multilingual)
                Example: `bert_model_name="all-mpnet-base-v2"`.

            length_type (str, default="char"):
                Type of length to compute when `strategy="len"`. Must be:
                    - `"char"`: Number of characters.
                    - `"word"`: Number of words (split by whitespace).
                Example: `length_type="word"` to count words instead of characters.

            max_features (Optional[int], default=None):
                Maximum number of features to keep for `"tfidf"` strategy (top-k by frequency).
                Passed to `TfidfVectorizer(max_features=...)`.
                Ignored for `"bert"` and `"len"` strategies.
                Example: `max_features=1000` to limit vocabulary size and reduce dimensionality.

            feature_names_prefix (str, default="text"):
                Prefix used in generated feature names to maintain traceability.
                Final feature names will be like:
                    - `"text_tfidf_price"` (for TF-IDF term "price")
                    - `"text_bert_0"`, `"text_bert_1"`, ... (for BERT dimensions)
                    - `"text_word_len"` (for word-length feature)
                Example: `feature_names_prefix="review"` → `"review_tfidf_excellent"`.

        Raises:
            ValueError: If an unsupported `strategy` or `length_type` is provided (checked at `fit` time).
        """
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
        """Fit the text transformer based on the selected strategy."""
        if hasattr(X, "iloc"):
            series = X.iloc[:, 0].fillna("").astype(str)
        else:
            series = pd.Series(X.ravel()).fillna("").astype(str)

        if self.strategy == "tfidf":
            params = {"max_features": self.max_features, **self.tfidf_params}
            self._vectorizer = TfidfVectorizer(**params).fit(series)
            vocab = self._vectorizer.vocabulary_
            self._feature_names = [
                f"{self.feature_names_prefix}_tfidf_{term}"
                for term in sorted(vocab, key=vocab.get)
            ]
        elif self.strategy == "bert":
            self._bert_model = SentenceTransformer(self.bert_model_name)
            dummy_emb = self._bert_model.encode(["dummy"])
            dim = dummy_emb.shape[1]
            self._feature_names = [
                f"{self.feature_names_prefix}_bert_{i}" for i in range(dim)
            ]
        elif self.strategy == "len":
            suffix = "char_len" if self.length_type == "char" else "word_len"
            self._feature_names = [f"{self.feature_names_prefix}_{suffix}"]
        else:
            raise ValueError(f"Unknown text extraction strategy: {self.strategy}")
        return self

    def transform(self, X):
        """Transform input text data using the fitted strategy."""
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
            return self._bert_model.encode(
                series.tolist(), show_progress_bar=False
            )
        else:
            raise RuntimeError("Text transformer must be fitted before transform.")

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        return np.array(self._feature_names)
