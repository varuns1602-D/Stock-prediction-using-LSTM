"""Backend for the NSE AI Stock Analyzer.

IMPORTANT: The functions in this module are the original app.py backend,
moved here verbatim so the UI can be restructured without touching the
prediction pipeline. Feature engineering, model loading, thresholds and the
inference maths are unchanged. Do NOT duplicate this logic elsewhere.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import yfinance as yf
from tensorflow import keras

from feature_engineering import build_features_for_stock


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model_artifacts"


@st.cache_resource(show_spinner=False)
def load_models():
    """Load every trained artifact once per process; cached across reruns."""

    def _guard(artifact_name, fn):
        try:
            return fn()
        except Exception as exc:  # surfaced as a clean message, never a traceback
            raise RuntimeError(f"Model artifact '{artifact_name}' could not be loaded: {exc}") from exc

    xgb_model = _guard(
        "universal_ga_xgboost.json",
        lambda: _load_xgb(),
    )
    lstm_model = _guard(
        "universal_lstm.keras",
        lambda: keras.models.load_model(str(MODEL_DIR / "universal_lstm.keras"), compile=False),
    )
    scaler = _guard(
        "universal_lstm_scaler.pkl",
        lambda: joblib.load(MODEL_DIR / "universal_lstm_scaler.pkl"),
    )
    features = _guard(
        "universal_features.json",
        lambda: json.loads((MODEL_DIR / "universal_features.json").read_text()),
    )
    config = _guard(
        "universal_config.json",
        lambda: json.loads((MODEL_DIR / "universal_config.json").read_text()),
    )
    stocks = _guard(
        "supported_stocks.json",
        lambda: json.loads((MODEL_DIR / "supported_stocks.json").read_text()),
    )
    return xgb_model, lstm_model, scaler, features, config, stocks


def _load_xgb():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "universal_ga_xgboost.json"))
    return model


@st.cache_data(show_spinner=False)
def load_fallback_latest():
    path = MODEL_DIR / "latest_predictions_all_stocks.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def supported_tickers(stocks_meta):
    tickers = [x["ticker"] for x in stocks_meta.get("trained_on", [])]
    return sorted(set(tickers))


def training_data_through(stocks_meta):
    """Latest training date recorded in supported_stocks.json (e.g. 2023-12-29)."""
    dates = [str(x.get("last_date", "")) for x in stocks_meta.get("trained_on", []) if x.get("last_date")]
    return max(dates) if dates else "unknown"


def company_name(stocks_meta, ticker):
    for row in stocks_meta.get("trained_on", []):
        if row.get("ticker") == ticker or row.get("symbol") == ticker:
            return row.get("symbol") or ticker
    return ticker


def normalise_market_df(df: pd.DataFrame, selected_stock: str) -> pd.DataFrame:
    """Convert common OHLCV variants into the exact schema expected by feature_engineering."""
    d = df.copy()

    if isinstance(d.columns, pd.MultiIndex):
        # yfinance can return MultiIndex columns in some versions/settings.
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]

    rename = {}
    for c in d.columns:
        key = str(c).strip().lower().replace("_", " ")
        if key in {"date", "datetime", "timestamp"}:
            rename[c] = "Date"
        elif key == "open":
            rename[c] = "Open"
        elif key == "high":
            rename[c] = "High"
        elif key == "low":
            rename[c] = "Low"
        elif key == "close":
            rename[c] = "Close"
        elif key in {"adj close", "adjusted close", "adjclose"}:
            rename[c] = "AdjClose"
        elif key in {"volume", "vol"}:
            rename[c] = "Volume"
        elif key in {"symbol", "ticker", "stock", "company"}:
            rename[c] = "Symbol"

    d = d.rename(columns=rename)

    if "Date" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={d.index.name or "index": "Date"})
        else:
            raise ValueError("The data needs a Date column or a DatetimeIndex.")

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "Symbol" not in d.columns:
        d["Symbol"] = selected_stock

    d["Symbol"] = d["Symbol"].astype(str).str.strip()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=required)
    d = d[(d[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    d = d[d["Volume"] >= 0]
    d = d[d["High"] >= d["Low"]]
    d = d.drop_duplicates(subset=["Symbol", "Date"], keep="last")
    d = d.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    # For uploaded single-stock files, make sure the selected stock is used.
    if selected_stock in d["Symbol"].values:
        d = d[d["Symbol"] == selected_stock].copy()
    elif d["Symbol"].nunique() == 1:
        d["Symbol"] = selected_stock
    else:
        d = d[d["Symbol"].str.upper() == selected_stock.upper()].copy()

    if d.empty:
        raise ValueError(f"No rows found for {selected_stock} in the supplied data.")

    return d


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance(selected_stock: str, period: str = "5y") -> pd.DataFrame:
    ticker = f"{selected_stock}.NS"
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    if data is None or data.empty:
        raise ValueError(f"No market data was returned for {ticker}.")
    data = data.reset_index()
    return normalise_market_df(data, selected_stock)


def infer(selected_stock: str, market_df: pd.DataFrame, models):
    """Run the saved GA-XGBoost + LSTM models. Logic unchanged from the original app."""
    xgb_model, lstm_model, scaler, features, config, _ = models

    g = market_df.copy()
    g = g.sort_values("Date").reset_index(drop=True)

    if len(g) < int(config["feature_engineering"]["min_rows_needed_for_lstm_inference"]):
        need = int(config["feature_engineering"]["min_rows_needed_for_lstm_inference"])
        raise ValueError(
            f"{selected_stock}: only {len(g)} rows of history were supplied, but the model needs at "
            f"least {need} rows — {config['feature_engineering']['max_warmup_rows']} rows for feature "
            f"warm-up plus the {config['lstm']['window']}-day LSTM sequence window. "
            "Choose a longer Yahoo Finance history or upload a CSV with more rows."
        )

    feat = build_features_for_stock(g)
    all_features = features["all_features"]
    ga_features = features["ga_selected_features"]
    lstm_features = features["lstm_features"]

    # Exact training-time feature availability and ordering.
    missing_all = [c for c in all_features if c not in feat.columns]
    if missing_all:
        raise ValueError(f"Feature-engineering mismatch. Missing features: {missing_all}")

    usable = feat.replace([np.inf, -np.inf], np.nan).dropna(subset=all_features)
    if len(usable) < int(config["lstm"]["window"]):
        raise ValueError("Not enough fully formed indicator rows after warm-up period.")

    latest_idx = usable.index[-1]
    latest_row = usable.loc[latest_idx]

    xgb_x = latest_row[ga_features].to_numpy(dtype=np.float32).reshape(1, -1)
    xgb_prob = float(xgb_model.predict_proba(xgb_x)[0, 1])
    xgb_thr = float(config["thresholds"].get("GA-XGBoost", 0.5))
    xgb_direction = "UP" if xgb_prob >= xgb_thr else "DOWN"

    window = int(features["lstm_window"])
    clip_z = float(config["lstm"]["clip_z"])
    seq_df = usable.loc[:latest_idx, lstm_features].tail(window).to_numpy(dtype=np.float32)
    seq_scaled = scaler.transform(seq_df)
    seq_scaled = np.clip(seq_scaled, -clip_z, clip_z)

    # Direct Keras call for low-latency single-sample inference with predict() fallback
    input_tensor = seq_scaled[np.newaxis, ...]
    try:
        lstm_out = lstm_model(input_tensor, training=False)
        lstm_prob = float(np.asarray(lstm_out).ravel()[0])
    except Exception:
        lstm_prob = float(lstm_model.predict(input_tensor, verbose=0).ravel()[0])

    lstm_thr = float(config["thresholds"].get("Universal LSTM", 0.5))
    lstm_direction = "UP" if lstm_prob >= lstm_thr else "DOWN"

    close = float(g.loc[latest_idx, "Close"])
    date = pd.Timestamp(g.loc[latest_idx, "Date"])

    return {
        "stock": selected_stock,
        "date": date,
        "close": close,
        "xgb_prob": xgb_prob,
        "xgb_threshold": xgb_thr,
        "xgb_direction": xgb_direction,
        "lstm_prob": lstm_prob,
        "lstm_threshold": lstm_thr,
        "lstm_direction": lstm_direction,
        "models_agree": xgb_direction == lstm_direction,
        "feature_rows": len(usable),
        # Display-only additions below; they do not influence any prediction.
        "features_df": feat,
        "rows_supplied": int(len(g)),
    }
