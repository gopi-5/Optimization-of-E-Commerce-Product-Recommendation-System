# Optimization of E-Commerce Product Recommendation System

### What’s included
- Modular, production-lean hybrid recommender: TF-IDF content + numeric popularity blend
- Fast item-item KNN retrieval with cosine similarity over sparse features
- CLI to build artifacts and query recommendations
- Reusable Python package under `src/recsys`

### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Build artifacts
```bash
python -m recsys.cli build --data \
  /workspace/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv \
  --out /workspace/artifacts
```
Artifacts written to `artifacts/`:
- `catalog.parquet`: deduplicated item catalog with `avg_rating` and `num_reviews`
- `item_features.npz`: sparse TF-IDF+numeric feature matrix
- `config.json`: minimal config

### Query recommendations
```bash
python -m recsys.cli query --artifacts /workspace/artifacts \
  --item "OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath" \
  --k 10
```

### Core approach
- Content encoding: TF-IDF over `name + brand + category + tags + description`, plus scaled `[avg_rating, num_reviews]`.
- Retrieval: cosine Item-KNN on sparse features.
- Ranking: blend KNN similarity with Bayesian-smoothed popularity
  - final_score = 0.7 * similarity + 0.3 * popularity.
- Cold-start: popularity-only fallback if the seed item is unseen.

### Reusing with your notebook
- Existing `miniprojcode.ipynb` loads the same TSV and explores multiple methods.
- To switch to the optimized pipeline inside the notebook:
  - Import and use the package:
    ```python
    from recsys.data import load_walmart_reviews_dataset, prepare_item_catalog
    from recsys.features import ContentEncoder
    from recsys.models import ItemKNNRecommender, HybridRecommender

    df = load_walmart_reviews_dataset('/workspace/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv')
    catalog = prepare_item_catalog(df)

    encoder = ContentEncoder(max_features=30000)
    X = encoder.fit_transform(catalog)

    knn = ItemKNNRecommender(n_neighbors=50).fit(X, catalog)
    hybrid = HybridRecommender(alpha=0.7).fit(knn, catalog)

    recs = hybrid.recommend('OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath', top_k=10)
    recs.head(10)
    ```

### Notes and improvements
- Swap `ItemKNNRecommender` with FAISS or Annoy for larger catalogs.
- Add user-personalized reranking with NMF/ALS embeddings when implicit feedback is available.
- Log experiments and metrics in MLflow; add offline evaluation (NDCG/Recall@K) over held-out interactions.
