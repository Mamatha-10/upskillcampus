# 🚦 Forecasting Smart City Traffic Patterns Using Machine Learning

Predicting hourly vehicle counts at four real city junctions to help a smart-city traffic management system
anticipate congestion, optimize signal timing, and support infrastructure planning.

**Dataset:** [Smart City Traffic Patterns](https://www.kaggle.com/utathya/smart-city-traffic-patterns)
(Analytics Vidhya / Kaggle) — hourly traffic counts, Nov 2015–Jun 2017 (train) and Jul–Oct 2017 (test, unlabeled).

## Project Structure

```
traffic_project/
├── data/
│   ├── train.csv                       # Labeled training data (48,120 rows)
│   ├── test.csv                        # Unlabeled competition test set (11,808 rows)
│   └── submission.csv                  # Generated forecasts for the test period
├── models/
│   └── traffic_prediction_model.pkl    # Trained & saved Random Forest model
├── notebooks/
│   └── Traffic_Forecasting.ipynb       # Full EDA + modeling notebook (with outputs)
├── train_models.py                     # End-to-end training/evaluation/forecasting script
├── requirements.txt
└── README.md
```

## The Data

| Feature  | Description                          |
|----------|---------------------------------------|
| DateTime | Date and time of observation (hourly) |
| Junction | Junction ID (1–4)                     |
| Vehicles | Number of vehicles (target, train only) |
| ID       | Unique record identifier              |

- **Train:** 48,120 rows. Junctions 1–3 span Nov 2015–Jun 2017 (14,592 rows each); Junction 4 only has data
  from Jan–Jun 2017 (4,344 rows) — it was added to monitoring later.
- **Test:** 11,808 rows (2,952 per junction), Jul–Oct 2017, **no `Vehicles` column** — this is the period the
  model needs to forecast (Analytics Vidhya competition style).
- No missing values, no duplicates, and no gaps in the hourly time index for any junction.
- Junction 1 shows a strong upward trend over time (traffic roughly triples from late 2015 to mid-2017) and
  carries by far the highest volume (up to 150+ vehicles/hour); Junctions 2–4 are much quieter.

## Workflow

1. **Data Cleaning** — duplicate handling, datetime parsing, gap check
2. **EDA** — traffic distribution, hourly/weekly/monthly trends, junction comparisons, correlation heatmap
3. **Feature Engineering** — time features (hour, day, month, weekend, week number, quarter) + lag features
   (`Lag1`, `Lag24`) + 24-hour rolling mean, computed per junction
4. **Modeling** — Linear Regression, Decision Tree Regressor, Random Forest Regressor, evaluated on a
   time-based validation split (last 20% of each junction's history)
5. **Deployment** — best model refit on all labeled data and saved with `joblib`
6. **Forecasting** — since the real test set has no target column, `Lag1`/`Lag24`/`RollingMean` are rolled
   forward **autoregressively**: each hour's prediction feeds into the next hour's lag features, seeded from
   the last known 24 hours of training data per junction

## Results (validation split)

| Model             | MAE  | RMSE | R² Score |
|-------------------|------|------|----------|
| Linear Regression | 3.73 | 5.62 | 0.957    |
| Decision Tree      | 3.97 | 6.60 | 0.941    |
| **Random Forest**  | **3.36** | **5.36** | **0.961** |

**Random Forest** was the best performer — lowest error, highest R², and better generalization than a single
Decision Tree.

Feature importance is dominated by `Lag1` (previous hour's count, ~97% importance), followed by `Hour`,
`Lag24`, and `RollingMean` — confirming that very recent history is by far the strongest signal for traffic
volume, with time-of-day playing a secondary role.

## How to Run

```bash
pip install -r requirements.txt
python train_models.py         # trains, evaluates, saves the best model, and forecasts the test period
jupyter notebook notebooks/Traffic_Forecasting.ipynb   # full walkthrough with plots
```

Running `train_models.py` produces:
- `models/traffic_prediction_model.pkl` — the trained model
- `data/submission.csv` — forecasted `Vehicles` for every row in `test.csv` (ID, Vehicles)

## Applications

- Smart traffic signal control
- Congestion prediction & route optimization
- Infrastructure & public transport planning
- Emergency vehicle routing

## Limitations

- Autoregressive forecasting means errors can compound over the 4-month test horizon — accuracy is best in
  the near term and degrades further out, since each hour's forecast depends on the previous hour's (predicted)
  value rather than ground truth.
- Weather, public holidays, and special events aren't in the data and aren't modeled.
- Junction 4 has a shorter history, which may limit how well the model learns its patterns.

## Future Scope

- LSTM/GRU networks for true sequential forecasting (would likely reduce compounding-error effects)
- Incorporate weather and holiday/festival calendars
- Live traffic feed integration (e.g., Google Maps)
- Real-time dashboard (Streamlit/Flask) or cloud-hosted API
