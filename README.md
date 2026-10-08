# NSE Stock Direction Predictor

This Streamlit application connects directly to the trained **universal GA-XGBoost + LSTM** models from the accompanying project.

## What it does

1. Select one of the 100 stocks used by the trained universal model.
2. Fetch recent NSE market data from Yahoo Finance, or upload an OHLCV CSV.
3. Recreate the project's feature-engineering pipeline.
4. Run the saved GA-XGBoost classifier on the latest feature row.
5. Run the saved universal LSTM on the latest 60-day feature sequence.
6. Display P(UP), the model thresholds, UP/DOWN direction, model agreement, and recent price history.

## Run locally on Windows

Open PowerShell in this folder:

```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Because TensorFlow is large, the first install can take some time.

## Files that connect the application to the trained model

- `model_artifacts/universal_ga_xgboost.json`
- `model_artifacts/universal_lstm.keras`
- `model_artifacts/universal_lstm_scaler.pkl`
- `model_artifacts/universal_features.json`
- `model_artifacts/universal_config.json`
- `model_artifacts/feature_engineering.py`
- `model_artifacts/supported_stocks.json`

## CSV upload format

Minimum columns:

`Date, Open, High, Low, Close, Volume`

A `Symbol` column is optional. When it is missing, the selected stock name is assigned automatically.

## Important model limitation

The saved model was trained on the project dataset through 2023-12-29. The app can fetch newer prices, but the model itself is **not retrained** on those newer observations.

The prediction is therefore an inference from the fixed trained model, not a guarantee of future market movement.
