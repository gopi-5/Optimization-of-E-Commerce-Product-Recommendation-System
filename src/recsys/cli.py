import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import sparse

from .data import load_walmart_reviews_dataset, prepare_item_catalog
from .features import ContentEncoder
from .models import ItemKNNRecommender, HybridRecommender


def build_index(data_path: str, artifacts_dir: str) -> None:
    df = load_walmart_reviews_dataset(data_path)
    catalog = prepare_item_catalog(df)
    encoder = ContentEncoder(max_features=30000)
    X = encoder.fit_transform(catalog)

    knn = ItemKNNRecommender(n_neighbors=50).fit(X, catalog)

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist minimal artifacts (parquet and npz)
    catalog.to_parquet(out_dir / "catalog.parquet", index=False)
    sparse.save_npz(out_dir / "item_features.npz", X)

    # Simple config for reload
    config = {
        "n_neighbors": 50,
        "encoder": {
            "max_features": encoder.max_features,
            "ngram_range": encoder.ngram_range,
            "text_columns": encoder.text_columns,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config))


def query(artifacts_dir: str, item_name: str, top_k: int = 10) -> pd.DataFrame:
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401 (for pickle safety if needed)

    out_dir = Path(artifacts_dir)
    catalog = pd.read_parquet(out_dir / "catalog.parquet")
    X = sparse.load_npz(out_dir / "item_features.npz")

    knn = ItemKNNRecommender(n_neighbors=50).fit(X, catalog)
    hybrid = HybridRecommender(alpha=0.7).fit(knn, catalog)
    return hybrid.recommend(item_name=item_name, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommender CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build item index and artifacts")
    b.add_argument("--data", required=True, help="Path to Walmart TSV dataset")
    b.add_argument("--out", required=True, help="Artifacts output directory")

    q = sub.add_parser("query", help="Query recommendations by item name")
    q.add_argument("--artifacts", required=True, help="Artifacts directory")
    q.add_argument("--item", required=True, help="Item name to search similar")
    q.add_argument("--k", type=int, default=10, help="Top K recommendations")

    args = parser.parse_args()
    if args.cmd == "build":
        build_index(args.data, args.out)
    elif args.cmd == "query":
        recs = query(args.artifacts, args.item, args.k)
        print(recs.to_string(index=False))


if __name__ == "__main__":
    main()