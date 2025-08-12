### Changelog

#### Added
- `src/recsys/` package with:
  - `data.py`: robust Walmart TSV loader and item catalog preparation.
  - `features.py`: `ContentEncoder` combining TF‑IDF over multiple text fields with scaled numeric features.
  - `models.py`: `ItemKNNRecommender` (cosine item‑item), `PopularityScorer`, `HybridRecommender` (similarity + popularity blend).
  - `cli.py`: build (`catalog.parquet`, `item_features.npz`) and query entry points.
- `scripts/benchmark_recsys.py`: reproducible benchmark producing before/after stats.
- `docs/OPTIMIZATION_REPORT.md`: optimization details, metrics, and instructions.

#### Changed
- `README.md`: replaced marketing copy with actionable setup, usage, and API examples.

#### Why these changes are better
- Modularity: decouples data prep, feature encoding, retrieval, and ranking for testability and reuse.
- Performance: sparse TF‑IDF features with cosine KNN provide fast, scalable retrieval; popularity blending improves cold‑start and quality proxies.
- Operability: CLI + artifacts simplify batch/offline pipelines and service integration.
- Reproducibility: a single benchmark script measures both speed and quality proxies for transparent comparison.