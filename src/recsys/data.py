import pandas as pd
from typing import Tuple

WALMART_COLS = [
    'Uniq Id', 'Product Id', 'Product Rating', 'Product Reviews Count',
    'Product Category', 'Product Brand', 'Product Name', 'Product Image Url',
    'Product Description', 'Product Tags'
]

RENAME_MAP = {
    'Uniq Id': 'user_id',
    'Product Id': 'item_id',
    'Product Rating': 'rating',
    'Product Reviews Count': 'num_reviews',
    'Product Category': 'category',
    'Product Brand': 'brand',
    'Product Name': 'name',
    'Product Image Url': 'image_url',
    'Product Description': 'description',
    'Product Tags': 'tags',
}


def load_walmart_reviews_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', usecols=WALMART_COLS)
    df = df.rename(columns=RENAME_MAP)

    df['rating'] = df['rating'].fillna(0.0).astype(float)
    df['num_reviews'] = df['num_reviews'].fillna(0.0).astype(float)
    for col in ['category', 'brand', 'description', 'tags', 'name', 'image_url']:
        df[col] = df[col].fillna('')

    # Normalize ids by extracting numeric substrings when present; fall back to categorical codes
    def extract_numeric(series: pd.Series) -> pd.Series:
        extracted = series.astype(str).str.extract(r'(\d+)')[0]
        if extracted.notna().any():
            return extracted.fillna('-1').astype(int)
        return series.astype('category').cat.codes.astype(int)

    df['user_key'] = extract_numeric(df['user_id'])
    df['item_key'] = extract_numeric(df['item_id'])

    return df


def prepare_item_catalog(df: pd.DataFrame) -> pd.DataFrame:
    item_cols = ['item_key', 'item_id', 'name', 'brand', 'category', 'description', 'tags', 'image_url', 'num_reviews']
    catalog = (
        df[item_cols + ['rating']]
        .groupby(item_cols, as_index=False)
        .agg(avg_rating=('rating', 'mean'))
    )
    return catalog