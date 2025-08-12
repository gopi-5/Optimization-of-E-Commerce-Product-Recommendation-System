### Optimization Report: E‑Commerce Product Recommendation System

#### What changed
- Hybrid content + popularity pipeline under `src/recsys/`
  - TF‑IDF over `name + brand + category + tags + description` plus scaled `[avg_rating, num_reviews]` features.
  - Item‑KNN (cosine) retrieval on sparse features.
  - Bayesian‑smoothed popularity blending for ranking and cold‑start fallback.
- CLI (`python -m recsys.cli`) to build artifacts and query recommendations.
- Reusable data utilities for the Walmart dataset and catalog preparation.

#### Why these changes
- Item‑KNN over TF‑IDF is simple, robust, and fast to serve; it scales linearly with catalog size and works well for text‑rich product data.
- Numeric popularity features counteract pure text similarity pitfalls (near‑duplicates with poor ratings, dead SKUs).
- Blending similarity with popularity generally improves engagement proxies when explicit user interaction data is sparse.
- Packaging the pipeline enables reuse from notebooks, scripts, and services.

#### Metrics to quantify optimization
- Performance
  - Build time (s): feature extraction + index fit.
  - Query latency (ms): median and p95 for Top‑K.
  - Artifact size (MB): on‑disk footprint of feature matrix.
- Quality proxies (no user logs available)
  - category_match@K: share of recommendations that match the seed item’s category.
  - brand_match@K: share matching brand (captures exact‑line extensions).
  - avg_tag_cosine: average TF‑IDF(tags) cosine similarity between seed and recommendations (method‑agnostic scoring).
  - distinct_brands@K: fraction of unique brands in Top‑K (diversity proxy).

These are neutral, data‑only proxies suitable for the provided catalog. If interaction logs are available later, prefer Recall@K, NDCG@K, MAP@K using held‑out user‑item interactions.

#### How to reproduce stats
1. Environment
   - Prefer Python 3.10 or 3.11 for binary wheels of scientific stack.
   - Create and activate venv, then install:
     ```bash
     python3 -m venv .venv && source .venv/bin/activate
     pip install -r requirements.txt
     pip install -e .
     ```
2. Run benchmark
   ```bash
   python scripts/benchmark_recsys.py \
     --data /workspace/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv \
     --k 10 \
     --sample 200 \
     --out /workspace/benchmark_results.json
   ```
3. Inspect results
   - `benchmark_results.json` contains a before/after summary for baseline vs optimized.

#### Example result format (your numbers will differ)
```json
{
  "build_time_s": {"baseline": 2.14, "optimized": 2.61},
  "artifact_size_mb": {"baseline": 12.3, "optimized": 14.8},
  "quality_proxies": {
    "baseline": {"category_match@k": 0.64, "brand_match@k": 0.21, "avg_tag_cosine": 0.38, "distinct_brands@k": 0.72},
    "optimized": {"category_match@k": 0.71, "brand_match@k": 0.27, "avg_tag_cosine": 0.42, "distinct_brands@k": 0.75}
  },
  "latency_ms": {"baseline": {"median": 3.2, "p95": 5.7}, "optimized": {"median": 2.4, "p95": 4.3}},
  "k": 10,
  "sample": 200
}
```

#### Interpretation guide
- If optimized median/p95 latency decreases while category_match@K, avg_tag_cosine increase, the new pipeline is both faster and more semantically aligned.
- A small increase in artifact size is expected from extra numeric features; serving cost typically remains dominated by TF‑IDF.
- If brand_match@K is too high and diversity drops, reduce `alpha` in `HybridRecommender` or increase `ngram_range` to encourage broader semantic variety.

#### Next steps
- Replace sklearn KNN with FAISS/Annoy for million‑item catalogs.
- Add user‑aware reranker (e.g., NMF/ALS embeddings) when interaction data is available.
- Add offline metrics (NDCG/Recall@K) once user‑item histories exist; integrate with MLflow.