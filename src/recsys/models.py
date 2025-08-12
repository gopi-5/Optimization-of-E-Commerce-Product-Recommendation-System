from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


class ItemKNNRecommender:
    def __init__(self, n_neighbors: int = 50, metric: str = "cosine") -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._nn: Optional[NearestNeighbors] = None
        self._item_features: Optional[sparse.csr_matrix] = None
        self._catalog: Optional[pd.DataFrame] = None

    def fit(self, item_features: sparse.csr_matrix, catalog: pd.DataFrame) -> "ItemKNNRecommender":
        self._item_features = item_features
        self._catalog = catalog.reset_index(drop=True)
        self._nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, item_features.shape[0]), metric=self.metric)
        self._nn.fit(item_features)
        return self

    def recommend_similar(self, item_name: str, top_k: int = 10) -> pd.DataFrame:
        assert self._nn is not None and self._catalog is not None and self._item_features is not None, "Fit first"
        mask = self._catalog['name'] == item_name
        if not mask.any():
            return pd.DataFrame()
        idx = int(np.flatnonzero(mask)[0])
        distances, indices = self._nn.kneighbors(self._item_features[idx], n_neighbors=min(top_k + 1, self._item_features.shape[0]))
        indices = indices.flatten().tolist()
        distances = distances.flatten().tolist()
        # Exclude the item itself at position 0
        pairs = [(i, d) for i, d in zip(indices, distances) if i != idx]
        top_indices = [i for i, _ in pairs][:top_k]
        results = self._catalog.iloc[top_indices][['item_key', 'item_id', 'name', 'brand', 'category', 'image_url']].copy()
        results['score_knn'] = 1.0 - np.array([d for _, d in pairs][:top_k])
        return results.reset_index(drop=True)


class PopularityScorer:
    def __init__(self, rating_weight: float = 1.0, review_weight: float = 0.5) -> None:
        self.rating_weight = rating_weight
        self.review_weight = review_weight

    def score(self, catalog: pd.DataFrame) -> pd.Series:
        # Bayesian-like smoothing of avg_rating by num_reviews
        reviews = catalog['num_reviews'].fillna(0.0)
        ratings = catalog['avg_rating'].fillna(0.0)
        global_mean = ratings.mean() if len(ratings) > 0 else 0.0
        m = reviews.quantile(0.8) if reviews.notna().any() else 0.0
        weighted = (reviews / (reviews + m)).fillna(0.0)
        bayes = weighted * ratings + (1 - weighted) * global_mean
        return self.rating_weight * bayes + self.review_weight * np.log1p(reviews)


class HybridRecommender:
    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = alpha
        self._pop = PopularityScorer()
        self._knn: Optional[ItemKNNRecommender] = None
        self._catalog: Optional[pd.DataFrame] = None

    def fit(self, knn: ItemKNNRecommender, catalog: pd.DataFrame) -> "HybridRecommender":
        self._knn = knn
        self._catalog = catalog.reset_index(drop=True)
        return self

    def recommend(self, item_name: str, top_k: int = 10) -> pd.DataFrame:
        assert self._knn is not None and self._catalog is not None, "Fit first"
        similar = self._knn.recommend_similar(item_name=item_name, top_k=top_k * 3)
        if similar.empty:
            # cold start: return popular items
            pop = self._catalog.copy()
            pop['score_pop'] = self._pop.score(pop)
            return pop.sort_values('score_pop', ascending=False).head(top_k)[['item_key','item_id','name','brand','category','image_url','score_pop']]

        # Blend with popularity
        merged = similar.merge(self._catalog[['name','num_reviews','avg_rating','item_key']], on=['name','item_key'], how='left')
        merged['score_pop'] = self._pop.score(merged.rename(columns={'avg_rating':'avg_rating','num_reviews':'num_reviews'}))
        merged['score'] = self.alpha * merged['score_knn'] + (1 - self.alpha) * merged['score_pop']
        merged = merged.sort_values('score', ascending=False)
        return merged.head(top_k)[['item_key','item_id','name','brand','category','image_url','score']].reset_index(drop=True)