"""
train_models.py
----------------
End-to-end pipeline for the Smart City Traffic Forecasting project, using the
real dataset (train_aWnotuB.csv / datasets_8494_11879_test_BdBKkAj.csv from
the Analytics Vidhya / Kaggle "Smart City Traffic Patterns" competition).

1. Load & clean train data
2. Feature engineering (time features, lag features, rolling mean)
3. Time-based train/validation split (for honest evaluation)
4. Train Linear Regression, Decision Tree, Random Forest
5. Evaluate with MAE / RMSE / R^2
6. Refit the best model on ALL of train, save it with joblib
7. Generate predictions for the real (unlabeled) test set -> submission.csv
"""

import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MODEL_PATH = "models/traffic_prediction_model.pkl"
SUBMISSION_PATH = "data/submission.csv"

FEATURE_COLS = [
    "Junction", "Year", "Month", "Day", "Hour", "DayOfWeek",
    "Weekend", "WeekNumber", "Quarter", "Season",
    "MonthStart", "MonthEnd", "QuarterStart", "QuarterEnd", "IsHoliday",
    "Lag1", "Lag24", "RollingMean",
]
TARGET_COL = "Vehicles"


# ---------------------------------------------------------------- #
# 1. Load & clean
# ---------------------------------------------------------------- #
def load_and_clean(path):
    df = pd.read_csv(path, parse_dates=["DateTime"])
    df = df.drop_duplicates()
    df = df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- #
# 2. Feature engineering
# ---------------------------------------------------------------- #
def get_holidays(years):
    holidays = []
    for year in years:
        holidays.extend([
            datetime.date(year, 1, 1),   # New Year's Day
            datetime.date(year, 1, 26),  # Republic Day
            datetime.date(year, 8, 15),  # Independence Day
            datetime.date(year, 10, 2),  # Gandhi Jayanti
            datetime.date(year, 12, 25), # Christmas
        ])
    return set(holidays)


def add_time_features(df):
    df = df.copy()
    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Day"] = df["DateTime"].dt.day
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["WeekNumber"] = df["DateTime"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["DateTime"].dt.quarter
    df["Season"] = ((df["Month"] % 12 + 3) // 3).astype(int)
    df["MonthStart"] = df["DateTime"].dt.is_month_start.astype(int)
    df["MonthEnd"] = df["DateTime"].dt.is_month_end.astype(int)
    df["QuarterStart"] = df["DateTime"].dt.is_quarter_start.astype(int)
    df["QuarterEnd"] = df["DateTime"].dt.is_quarter_end.astype(int)
    holidays = get_holidays(df["Year"].unique())
    df["IsHoliday"] = df["DateTime"].dt.date.isin(holidays).astype(int)
    return df


def add_lag_features(df, vehicles_col="Vehicles"):
    """Adds Lag1 / Lag24 / RollingMean computed per junction. `df` must
    already be sorted by Junction, DateTime and contain `vehicles_col`."""
    df = df.copy()
    df["Lag1"] = df.groupby("Junction")[vehicles_col].shift(1)
    df["Lag24"] = df.groupby("Junction")[vehicles_col].shift(24)
    df["RollingMean"] = (
        df.groupby("Junction")[vehicles_col]
        .shift(1)
        .rolling(window=24, min_periods=1)
        .mean()
    )
    return df


# ---------------------------------------------------------------- #
# 3. Split (time-based, per junction, to avoid look-ahead leakage)
# ---------------------------------------------------------------- #
def time_based_split(df, test_frac=0.2):
    train_parts, val_parts = [], []
    for _, group in df.groupby("Junction"):
        cutoff = int(len(group) * (1 - test_frac))
        train_parts.append(group.iloc[:cutoff])
        val_parts.append(group.iloc[cutoff:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(val_parts).reset_index(drop=True)


def evaluate(name, model, X_val, y_val):
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"{name:20s}  MAE={mae:6.2f}   RMSE={rmse:6.2f}   R2={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    print("Loading data...")
    train_raw = load_and_clean(TRAIN_PATH)
    test_raw = load_and_clean(TEST_PATH)

    print("Engineering features...")
    train_feat = add_time_features(train_raw)
    train_feat = add_lag_features(train_feat)
    train_feat = train_feat.dropna().reset_index(drop=True)

    # --- Hold-out validation split (last 20% of each junction's history) ---
    tr, val = time_based_split(train_feat)
    X_train, y_train = tr[FEATURE_COLS], tr[TARGET_COL]
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL]

    results = []
    trained_models = {}

    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    trained_models["Linear Regression"] = lr
    results.append(evaluate("Linear Regression", lr, X_val, y_val))

    print("Training Decision Tree Regressor...")
    dt = DecisionTreeRegressor(max_depth=12, random_state=42)
    dt.fit(X_train, y_train)
    trained_models["Decision Tree"] = dt
    results.append(evaluate("Decision Tree", dt, X_val, y_val))

    print("Training Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=3, n_jobs=1, random_state=42
    )
    rf.fit(X_train, y_train)
    trained_models["Random Forest"] = rf
    results.append(evaluate("Random Forest", rf, X_val, y_val))

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\n=== Model comparison (sorted by RMSE) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    # --- Refit best model's architecture on ALL available train data ------
    model_builders = {
        "Linear Regression": lambda: LinearRegression(),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_leaf=3, n_jobs=1, random_state=42
        ),
    }
    final_model = model_builders[best_name]()
    X_full, y_full = train_feat[FEATURE_COLS], train_feat[TARGET_COL]
    final_model.fit(X_full, y_full)

    joblib.dump(final_model, MODEL_PATH, compress=3)
    print(f"Saved final {best_name} (trained on full train set) to {MODEL_PATH}")

    if hasattr(final_model, "feature_importances_"):
        importances = pd.Series(
            final_model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)
        print("\nTop feature importances:")
        print(importances.head(8).to_string())

    # --- Predict on the real (unlabeled) test set --------------------------
    # The test set continues right after the train set for each junction, so
    # Lag1/Lag24/RollingMean are seeded from the tail of train and then rolled
    # forward autoregressively hour by hour.
    print("\nGenerating forecasts for the held-out test period...")
    from collections import deque

    test_feat = add_time_features(test_raw.sort_values(["Junction", "DateTime"]).reset_index(drop=True))
    all_preds = []

    for j, test_group in test_feat.groupby("Junction"):
        train_tail = (
            train_raw[train_raw["Junction"] == j]
            .sort_values("DateTime")["Vehicles"]
            .tail(24)
            .tolist()
        )
        history = deque(train_tail, maxlen=24)   # most-recent-last

        for _, row in test_group.sort_values("DateTime").iterrows():
            lag1 = history[-1]
            lag24 = history[0] if len(history) == 24 else history[-1]
            rolling_mean = float(np.mean(history))

            feat_row = pd.DataFrame([{
                "Junction": j,
                "Year": row["Year"], "Month": row["Month"], "Day": row["Day"],
                "Hour": row["Hour"], "DayOfWeek": row["DayOfWeek"],
                "Weekend": row["Weekend"], "WeekNumber": row["WeekNumber"],
                "Quarter": row["Quarter"], "Season": row["Season"],
                "MonthStart": row["MonthStart"], "MonthEnd": row["MonthEnd"],
                "QuarterStart": row["QuarterStart"], "QuarterEnd": row["QuarterEnd"],
                "IsHoliday": row["IsHoliday"],
                "Lag1": lag1, "Lag24": lag24, "RollingMean": rolling_mean,
            }])
            pred = max(round(final_model.predict(feat_row[FEATURE_COLS])[0]), 1)
            history.append(pred)
            all_preds.append({"ID": row["ID"], "Vehicles": int(pred)})

    submission = pd.DataFrame(all_preds).sort_values("ID").reset_index(drop=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {len(submission):,} test-set predictions to {SUBMISSION_PATH}")

    print("\n=== Generated output files ===")
    print(f"Model file: {MODEL_PATH}")
    print(f"Submission file: {SUBMISSION_PATH}")
    print("Text output: console metrics and top feature importances")
    print("Visualization/report files: run 'python scripts/generate_report.py' to create report files under reports/")

    return results_df, trained_models, final_model, submission


if __name__ == "__main__":
    main()
