### Usage Guide

#### Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

#### Build Artifacts
```bash
python -m recsys.cli build --data \
  /workspace/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv \
  --out /workspace/artifacts
```
Artifacts:
- `catalog.parquet`: item catalog with `avg_rating` and `num_reviews`
- `item_features.npz`: sparse feature matrix
- `config.json`: minimal configuration

#### Query Recommendations
```bash
python -m recsys.cli query --artifacts /workspace/artifacts \
  --item "OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath" \
  --k 10
```

#### Benchmark (Baseline vs Optimized)
```bash
python scripts/benchmark_recsys.py \
  --data /workspace/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv \
  --k 10 \
  --sample 200 \
  --out /workspace/benchmark_results.json
```

#### Notebook Integration
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