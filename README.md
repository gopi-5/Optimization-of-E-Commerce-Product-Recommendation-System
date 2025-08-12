# Optimization of E-Commerce Product Recommendation System

### Overview
This project builds an optimized product recommendation system for e-commerce catalogs. It aims to improve product discovery and user engagement by surfacing relevant items through a combination of semantic similarity and popularity awareness.

### Goals
- Deliver relevant and diverse product recommendations
- Remain robust with sparse or missing interaction data
- Keep retrieval scalable and fast for large catalogs

### Dataset
- Source: Walmart product review sample (TSV)
- Core fields used: user/product identifiers, product names, brands, categories, descriptions, tags, ratings, review counts, and image URLs

### Recommendation Strategies
- Popularity-aware ranking to highlight high-quality and well-reviewed items
- Content-based retrieval using product metadata (names, brands, categories, tags, descriptions)
- Hybrid ranking to blend semantic similarity with popularity for better overall relevance

### System Components
- Data preparation to normalize identifiers and build an item catalog
- Feature encoding to represent items from text and numeric signals
- Nearest-neighbor retrieval for efficient candidate generation
- Reranking to balance relevance, quality, and diversity

### Evaluation (catalog-only setting)
Without user-event logs, the project uses catalog-derived proxies:
- Category/brand matching rates within Top-K
- Average content similarity to the seed item
- Diversity metrics (e.g., distinct brands)
- Latency and artifact footprint for performance

### Repository Structure
- `src/recsys/`: modular Python package for data, features, models, and CLI
- `scripts/`: utilities for building and benchmarking
- `docs/`: detailed documentation and reports
- `miniprojcode.ipynb`: exploratory notebook using the Walmart sample

### Learn More
- Usage guide: see `docs/USAGE.md`
- Optimization details and metrics: see `docs/OPTIMIZATION_REPORT.md`
- Change history and rationale: see `docs/CHANGELOG.md`
