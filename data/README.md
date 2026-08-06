# Data

Raw and processed data live here. Both are gitignored — only this README is
tracked.

## How to get the raw data

### Option 1 — Kaggle API (recommended)

```bash
pip install kaggle
# place your API token at ~/.kaggle/kaggle.json
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

### Option 2 — Manual download

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it at `data/raw/creditcard.csv`

## What goes where

| Path | Contents | Gitignored |
|---|---|---|
| `data/raw/creditcard.csv` | Original ULB dataset (284,807 rows) | Yes |
| `data/processed/train.parquet` | Stratified train split (70%) | Yes |
| `data/processed/val.parquet` | Stratified validation split (15%) | Yes |
| `data/processed/test.parquet` | Stratified test split (15%) | Yes |
| `data/alerts.log` | Alert log from the simulator | Yes |
| `data/transactions.log` | Transaction log from the simulator | Yes |

## Dataset details

- Source: ULB, Kaggle
- 284,807 transactions, 31 columns
- 492 frauds (0.172%)
- Features V1–V28 are PCA-anonymized; Time + Amount are original
- License: ODbL
- See Obsidian: `01-data/Dataset Choice.md` and `ADR-001`
