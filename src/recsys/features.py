from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


class ContentEncoder:
    def __init__(
        self,
        text_columns: Optional[List[str]] = None,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:
        self.text_columns = text_columns or ["name", "brand", "category", "tags", "description"]
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False

    def _build_corpus(self, catalog: pd.DataFrame) -> List[str]:
        corpus = (
            catalog[self.text_columns]
            .astype(str)
            .apply(lambda row: " ".join(row.values), axis=1)
            .tolist()
        )
        return corpus

    def fit(self, catalog: pd.DataFrame) -> "ContentEncoder":
        corpus = self._build_corpus(catalog)
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        _ = self._vectorizer.fit(corpus)

        self._scaler = StandardScaler(with_mean=False)
        numeric = catalog[["avg_rating", "num_reviews"]].fillna(0.0).astype(float).values
        _ = self._scaler.fit(numeric)

        self._fitted = True
        return self

    def transform(self, catalog: pd.DataFrame) -> sparse.csr_matrix:
        assert self._fitted and self._vectorizer is not None and self._scaler is not None, "Fit first"
        tfidf = self._vectorizer.transform(self._build_corpus(catalog))
        numeric = self._scaler.transform(catalog[["avg_rating", "num_reviews"]].fillna(0.0).astype(float).values)
        numeric_sparse = sparse.csr_matrix(numeric)
        features = sparse.hstack([tfidf, numeric_sparse]).tocsr()
        return features

    def fit_transform(self, catalog: pd.DataFrame) -> sparse.csr_matrix:
        self.fit(catalog)
        return self.transform(catalog)

    @property
    def vocabulary_size(self) -> int:
        if self._vectorizer is None:
            return 0
        return len(self._vectorizer.vocabulary_)  # type: ignore[return-value]