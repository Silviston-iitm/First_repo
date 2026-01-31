# House Price Regression Trainer

Usage:

1. Install dependencies (preferably in a virtualenv):

```bash
python -m pip install -r requirements.txt
```

2. Run the trainer with your CSV:

```bash
python train_house_price_model.py --csv path/to/your.csv --out model.joblib
```

The script will try to auto-detect these features (using common synonyms):
- bedrooms
- area (sqft)
- bathrooms
- distance to city
and a target column like `price` or `value`.

It prints evaluation metrics for a `LinearRegression` baseline and a `RandomForestRegressor`, and saves the trained RandomForest model and preprocessor to the `--out` path.
