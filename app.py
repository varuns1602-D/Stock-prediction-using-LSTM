"""NSE AI Stock Analyzer — Streamlit entry point.

Layout / navigation only. The prediction pipeline itself lives in
prediction_backend.py (moved verbatim from the original app) and
feature_engineering.py — both untouched by this redesign.
"""

import hashlib
import traceback

import pandas as pd
import streamlit as st

from dashboard_ui import inject_global_styles, pretty_date, render_limitation_strip, render_top_header
from prediction_backend import (
    company_name,
    fetch_yfinance,
    infer,
    load_models,
    normalise_market_df,
    supported_tickers,
    training_data_through,
)
from ui_charts import render_charts_page
from ui_methodology import render_methodology_page
from ui_performance import render_performance_page
from ui_prediction import render_prediction_page

st.set_page_config(page_title="NSE AI Stock Analyzer", page_icon="📈", layout="wide")
inject_global_styles()

PAGES = ["Prediction", "Charts", "Model Performance", "Methodology"]
MARKET_ERROR = "Unable to retrieve market data. Try again or upload an OHLCV CSV."


# ---------------- centralised, cached model loading ----------------
try:
    models = load_models()
except Exception as exc:
    st.markdown(
        f'<div class="warn-box"><div class="box-title">Model loading failed</div>{exc}</div>',
        unsafe_allow_html=True,
    )
    st.stop()

xgb_model, lstm_model, scaler, features, config, stocks_meta = models
tickers = supported_tickers(stocks_meta)
train_through = training_data_through(stocks_meta)


# ---------------- prediction run ----------------
def execute_run(settings: dict) -> dict:
    """Fetch market data + run the trained models. Returns a display-ready dict."""
    stock = settings["stock"]
    try:
        if settings["source"] == "Yahoo Finance":
            try:
                market_df = fetch_yfinance(stock, period=settings["period"])
            except ValueError as exc:
                return {"ok": False, "pending": False, "error": str(exc)}
            except Exception:
                return {"ok": False, "pending": False,
                        "error": f"{MARKET_ERROR} (stock: {stock})"}
        else:
            if settings["uploaded"] is None:
                return {"ok": False, "pending": True, "error": None}
            market_df = normalise_market_df(pd.read_csv(settings["uploaded"]), stock)

        result = infer(stock, market_df, models)
        return {
            "ok": True,
            "pending": False,
            "market_df": market_df,
            "result": result,
            "company": company_name(stocks_meta, stock),
            "config": config,
        }
    except Exception as exc:
        return {"ok": False, "pending": False, "error": str(exc)}


st.session_state.setdefault("page", "Prediction")
st.session_state.setdefault("run", None)

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown(
        '<div style="font-size:0.76rem; letter-spacing:0.16em; text-transform:uppercase; '
        'color:#8FA3C0; font-weight:800; margin-bottom:10px;">⚙ Prediction Settings</div>',
        unsafe_allow_html=True,
    )
    selected_stock = st.selectbox(
        "Stock",
        tickers,
        index=tickers.index("TCS") if "TCS" in tickers else 0,
    )
    source = st.radio(
        "Market Data Source",
        ["Yahoo Finance", "Upload OHLCV CSV"],
        index=0,
    )
    period = st.selectbox(
        "Yahoo Finance History",
        ["1y", "2y", "3y", "5y", "10y"],
        index=3,
        disabled=(source != "Yahoo Finance"),
    )
    uploaded = None
    if source == "Upload OHLCV CSV":
        uploaded = st.file_uploader("Upload OHLCV CSV", type=["csv"],
                                    help="Minimum columns: Date, Open, High, Low, Close, Volume")

    run_clicked = st.button("⚡ Generate Prediction", type="primary", width="stretch")

    st.divider()
    st.markdown(
        '<div style="font-size:0.74rem; letter-spacing:0.14em; text-transform:uppercase; '
        'color:#8FA3C0; font-weight:800; margin-bottom:8px;">Model Information</div>'
        f'<div style="font-size:0.85rem; line-height:1.9; color:#B9C6DC;">'
        f'<span style="color:#7C8CA8;">Training universe:</span> <b>{config["universe_size"]} stocks</b><br>'
        f'<span style="color:#7C8CA8;">LSTM window:</span> <b>{config["lstm"]["window"]} trading days</b><br>'
        f'<span style="color:#7C8CA8;">Training data through:</span> <b>{train_through}</b><br>'
        f'<span style="color:#7C8CA8;">Features after GA:</span> <b>{config["n_features_after_ga"]}</b>'
        f' <span style="color:#7C8CA8;">of {config["n_features_before_ga"]}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.76rem; color:#6B7C99; margin-top:14px; line-height:1.6;">'
        "Models are loaded once and cached — switching widgets never reloads TensorFlow or XGBoost.</div>",
        unsafe_allow_html=True,
    )

file_sig = ""
if uploaded is not None:
    file_sig = hashlib.sha256(uploaded.getbuffer()).hexdigest()[:16]

settings = {
    "stock": selected_stock,
    "source": source,
    "period": period,
    "uploaded": uploaded,
}
settings_key = f"{selected_stock}|{source}|{period}|{file_sig}"

# ---------------- run prediction: only on button click ----------------
run = st.session_state.run
if run_clicked:
    if settings["source"] == "Upload OHLCV CSV" and settings["uploaded"] is None:
        st.sidebar.warning("Please upload an OHLCV CSV file before generating prediction.")
    else:
        with st.spinner("Fetching market data and running the trained GA-XGBoost + LSTM models..."):
            new_run = execute_run(settings)
        new_run["key"] = settings_key
        st.session_state.run = new_run
        run = new_run

# Display prediction only if it matches current settings; otherwise show tailored pending state
if run is not None and run.get("key") == settings_key:
    render_run = run
else:
    if settings["source"] == "Upload OHLCV CSV":
        if settings["uploaded"] is None:
            msg = "Upload an OHLCV CSV to generate a prediction."
        else:
            msg = f"CSV file <b>{uploaded.name}</b> selected. Click <b>⚡ Generate Prediction</b> in the sidebar to run."
    elif run is None:
        msg = "Select a stock and click <b>⚡ Generate Prediction</b>."
    else:
        prev_stock = run.get("key", "").split("|")[0] if run.get("key") else ""
        if prev_stock and prev_stock != selected_stock:
            msg = f"Prediction not yet generated for <b>{selected_stock}</b>. Click <b>⚡ Generate Prediction</b> in the sidebar to run."
        else:
            msg = f"Settings changed. Click <b>⚡ Generate Prediction</b> in the sidebar to run."

    render_run = {"ok": False, "pending": True, "pending_message": msg, "error": None}

# ---------------- header + navigation ----------------
if render_run and render_run.get("ok"):
    hdr_stock = render_run["result"]["stock"]
    hdr_date = render_run["result"]["date"]
else:
    hdr_stock = selected_stock
    hdr_date = None
render_top_header(hdr_stock, hdr_date)

page = st.segmented_control(
    "Navigation",
    PAGES,
    default=st.session_state.page,
    key="nav_ctrl",
    label_visibility="collapsed",
    width="stretch",
)
if page:
    st.session_state.page = page
active = st.session_state.page if st.session_state.page in PAGES else "Prediction"

render_limitation_strip(pretty_date(train_through))

# ---------------- page dispatch (tracebacks stay collapsed) ----------------
try:
    if active == "Prediction":
        render_prediction_page(render_run, settings)
    elif active == "Charts":
        render_charts_page(render_run, settings)
    elif active == "Model Performance":
        render_performance_page(models, stocks_meta, train_through)
    elif active == "Methodology":
        render_methodology_page(models, train_through)
except Exception as exc:
    st.error(f"Something went wrong while rendering the {active} page: {exc}")
    with st.expander("Technical details"):
        st.text(traceback.format_exc())

st.caption(config.get("disclaimer", ""))
