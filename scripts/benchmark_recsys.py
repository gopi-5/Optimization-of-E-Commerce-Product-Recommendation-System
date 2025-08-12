import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from recsys.data import load_walmart_reviews_dataset, prepare_item_catalog
from recsys.features import ContentEncoder
from recsys.models import ItemKNNRecommender, HybridRecommender


def build_baseline(catalog: pd.DataFrame):
    tags = catalog["tags"].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
    X = tfidf.fit_transform(tags)
    nn = NearestNeighbors(n_neighbors=50, metric="cosine").fit(X)
    return tfidf, X, nn


def baseline_recommend(nn: NearestNeighbors, X: sparse.csr_matrix, catalog: pd.DataFrame, item_name: str, top_k: int) -> pd.DataFrame:
    mask = catalog["name"] == item_name
    if not mask.any():
        return pd.DataFrame()
    idx = int(np.flatnonzero(mask)[0])
    distances, indices = nn.kneighbors(X[idx], n_neighbors=min(top_k + 1, X.shape[0]))
    indices = [i for i in indices.flatten().tolist() if i != idx][:top_k]
    out = catalog.iloc[indices][["item_key","item_id","name","brand","category","image_url"]].copy()
    return out.reset_index(drop=True)


def compute_quality_proxies(catalog: pd.DataFrame, seed_idx: int, recs: pd.DataFrame, tfidf: TfidfVectorizer, X_tfidf: sparse.csr_matrix) -> Dict[str, float]:
    if recs.empty:
        return {"category_match@k": 0.0, "brand_match@k": 0.0, "avg_tag_cosine": 0.0, "distinct_brands@k": 0.0}
    seed_row = catalog.iloc[seed_idx]
    same_category = (recs["category"] == seed_row["category"]).mean()
    same_brand = (recs["brand"] == seed_row["brand"]).mean()

    # Tag cosine using tags-only TF-IDF for apples-to-apples
    rec_indices = catalog.reset_index().merge(recs[["item_key"]], on="item_key", how="inner")["index"].tolist()
    if len(rec_indices) == 0:
        avg_cos = 0.0
    else:
        sims = 1.0 - sparse.csr_matrix.nnz  # dummy to satisfy type checker
        seed_vec = X_tfidf[seed_idx]
        rec_mat = X_tfidf[rec_indices]
        cos = 1 - (seed_vec @ rec_mat.T).toarray()[0]
        # but above yields distances for cosine metric; compute properly using norms
        seed_norm = np.sqrt(seed_vec.multiply(seed_vec).sum())
        rec_norms = np.sqrt(rec_mat.multiply(rec_mat).sum(axis=1)).A1
        dot = (seed_vec @ rec_mat.T).A1
        denom = (seed_norm * rec_norms + 1e-12)
        cos_sim = (dot / denom)
        avg_cos = float(np.mean(cos_sim))

    distinct_brands = recs["brand"].nunique() / max(len(recs), 1)

    return {
        "category_match@k": float(same_category),
        "brand_match@k": float(same_brand),
        "avg_tag_cosine": avg_cos,
        "distinct_brands@k": float(distinct_brands),
    }


def measure_latency(fn, n_runs: int = 5) -> Tuple[float, float]:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return float(np.median(times)), float(np.percentile(times, 95))


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline vs optimized recommender")
    parser.add_argument("--data", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--out", default="/workspace/benchmark_results.json")
    args = parser.parse_args()

    # Load and prep
    df = load_walmart_reviews_dataset(args.data)
    catalog = prepare_item_catalog(df)

    # Baseline build
    t0 = time.perf_counter()
    bl_tfidf, bl_X, bl_nn = build_baseline(catalog)
    baseline_build_s = time.perf_counter() - t0

    # Optimized build
    t1 = time.perf_counter()
    encoder = ContentEncoder(max_features=30000)
    X_opt = encoder.fit_transform(catalog)
    knn = ItemKNNRecommender(n_neighbors=50).fit(X_opt, catalog)
    hybrid = HybridRecommender(alpha=0.7).fit(knn, catalog)
    optimized_build_s = time.perf_counter() - t1

    # Artifact sizes (approx)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        p_bl = Path(td) / "bl.npz"
        p_opt = Path(td) / "opt.npz"
        sparse.save_npz(p_bl, bl_X)
        sparse.save_npz(p_opt, X_opt)
        size_bl_mb = p_bl.stat().st_size / (1024 * 1024)
        size_opt_mb = p_opt.stat().st_size / (1024 * 1024)

    # Sample items
    rng = np.random.default_rng(0)
    valid_indices = catalog.index.tolist()
    sample_indices = rng.choice(valid_indices, size=min(args.sample, len(valid_indices)), replace=False).tolist()

    # Quality proxies and latency
    bl_metrics: List[Dict[str, float]] = []
    opt_metrics: List[Dict[str, float]] = []
    bl_latencies = []
    opt_latencies = []

    for idx in sample_indices:
        name = catalog.iloc[idx]["name"]
        # Baseline latency
        med, p95 = measure_latency(lambda: baseline_recommend(bl_nn, bl_X, catalog, name, args.k))
        bl_latencies.append((med, p95))
        bl_recs = baseline_recommend(bl_nn, bl_X, catalog, name, args.k)
        bl_metrics.append(compute_quality_proxies(catalog, idx, bl_recs, bl_tfidf, bl_X))

        # Optimized latency
        med2, p95_2 = measure_latency(lambda: hybrid.recommend(name, top_k=args.k))
        opt_latencies.append((med2, p95_2))
        opt_recs = hybrid.recommend(name, top_k=args.k)
        opt_metrics.append(compute_quality_proxies(catalog, idx, opt_recs, bl_tfidf, bl_X))

    def aggregate(metrics: List[Dict[str, float]]) -> Dict[str, float]:
        keys = metrics[0].keys() if metrics else []
        return {k: float(np.mean([m[k] for m in metrics])) for k in keys}

    results = {
        "build_time_s": {"baseline": baseline_build_s, "optimized": optimized_build_s},
        "artifact_size_mb": {"baseline": size_bl_mb, "optimized": size_opt_mb},
        "quality_proxies": {"baseline": aggregate(bl_metrics), "optimized": aggregate(opt_metrics)},
        "latency_ms": {
            "baseline": {
                "median": float(np.median([x[0] for x in bl_latencies])) * 1000,
                "p95": float(np.median([x[1] for x in bl_latencies])) * 1000,
            },
            "optimized": {
                "median": float(np.median([x[0] for x in opt_latencies])) * 1000,
                "p95": float(np.median([x[1] for x in opt_latencies])) * 1000,
            },
        },
        "k": args.k,
        "sample": len(sample_indices),
    }

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()