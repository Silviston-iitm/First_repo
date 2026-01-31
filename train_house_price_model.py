#!/usr/bin/env python3
"""
Train a regression model to predict house price from a CSV file.

Expected minimum features (try common synonyms):
- bedrooms (e.g. bedrooms, num_bedrooms)
- area (e.g. area, sqft, size)
- bathrooms (e.g. bathrooms, baths)
- distance to city (e.g. distance_to_city, distance)
Target column (e.g. price, value)

Example:
python train_house_price_model.py --csv data/houses.csv --out model.joblib
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COL_KEYWORDS = {
    "bedrooms": ["bedrooms", "num_bedrooms", "number_of_bedrooms", "beds"],
    "area": ["area", "sqft", "size", "area_sqft", "square_feet"],
    "bathrooms": ["bathrooms", "baths", "num_bathrooms", "number_of_bathrooms"],
    "distance": ["distance", "distance_to_city", "distance_to_center", "dist_city", "distance_km", "dist_km"],
    "price": ["price", "value", "sale_price", "house_price", "target"],
}


def find_column(df_cols, keywords):
    cols_lower = {c.lower(): c for c in df_cols}
    for kw in keywords:
        if kw.lower() in cols_lower:
            return cols_lower[kw.lower()]
    # try substring match
    for col_low, col_orig in cols_lower.items():
        for kw in keywords:
            if kw.lower() in col_low:
                return col_orig
    return None


def detect_columns(df):
    found = {}
    for name, keywords in COL_KEYWORDS.items():
        col = find_column(df.columns.tolist(), keywords)
        if col:
            found[name] = col
    return found


def to_numeric_df(df, cols):
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def print_metrics(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix} MAE: {mae:.3f}")
    print(f"{prefix} RMSE: {rmse:.3f}")
    print(f"{prefix} R2: {r2:.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train house price regression model from CSV")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--out", default="model.joblib", help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        sys.exit(2)

    df = pd.read_csv(csv_path)
    if df.shape[0] < 10:
        print("Warning: dataset has fewer than 10 rows — metrics may be unreliable")

    detected = detect_columns(df)
    missing = [k for k in ("bedrooms", "area", "bathrooms", "distance", "price") if k not in detected]
    if missing:
        print("Could not detect required columns for:", ", ".join(missing))
        print("Available columns:", ", ".join(df.columns.tolist()))
        sys.exit(2)

    feature_cols = [detected[k] for k in ("bedrooms", "area", "bathrooms", "distance")]
    target_col = detected["price"]

    X = to_numeric_df(df, feature_cols)
    y = pd.to_numeric(df[target_col], errors="coerce")

    # drop rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    if X.shape[1] < 4:
        print("Error: fewer than 4 numeric feature columns after conversion")
        sys.exit(2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Train a small baseline linear model and a RandomForest
    lr = LinearRegression()
    lr.fit(X_train_p, y_train)
    y_pred_lr = lr.predict(X_test_p)

    rf = RandomForestRegressor(n_estimators=100, random_state=args.random_state)
    rf.fit(X_train_p, y_train)
    y_pred_rf = rf.predict(X_test_p)

    print("Evaluation on test set:")
    print("LinearRegression:")
    print_metrics(y_test, y_pred_lr, prefix="  ")
    print("RandomForestRegressor:")
    print_metrics(y_test, y_pred_rf, prefix="  ")

    out_path = Path(args.out)
    joblib.dump({"model": rf, "preprocessor": preprocessor, "features": feature_cols}, out_path)
    print(f"Saved RandomForest model and preprocessor to {out_path}")


if __name__ == "__main__":
    main()
