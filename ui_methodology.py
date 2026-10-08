"""Methodology / About page — visual pipeline flow + project explanation.

Every number shown here is read from universal_config.json /
supported_stocks.json at runtime, so it always matches the trained project.
"""

import streamlit as st

from dashboard_ui import card, fmt_pct, pretty_date, section_title


def _node(title: str, sub: str = "", cls: str = "") -> str:
    sub_html = '<div class="fn-sub">' + sub + "</div>" if sub else ""
    return '<div class="flow-node ' + cls + '"><div class="fn-title">' + title + "</div>" + sub_html + "</div>"


def _flow(config: dict) -> str:
    universe = config.get("universe_size")
    before = config.get("n_features_before_ga")
    after = config.get("n_features_after_ga")
    window = config.get("lstm", {}).get("window")
    return f"""
    <div class="flow">
      {_node("NSE Historical OHLCV", f"Yahoo Finance / uploaded CSV · {universe}-stock training universe")}
      <div class="flow-arrow">↓</div>
      {_node("Data Cleaning", "sort by date · duplicate handling · OHLCV validation")}
      <div class="flow-arrow">↓</div>
      {_node("Technical / Relative Feature Engineering", f"feature_engineering.py · {before} scale-aware features")}
      <div class="flow-arrow">↓</div>
      {_node("GA Feature Selection", f"genetic algorithm keeps {after} informative features", "accent")}

      <div class="branch-connector">
        <div style="position:relative; width:70%; max-width:560px; height:24px; margin:0 auto;">
          <div style="position:absolute; left:50%; top:0; height:55%; width:2px; background:rgba(126,166,255,0.6);"></div>
          <div style="position:absolute; left:25%; right:25%; top:55%; height:2px; background:rgba(126,166,255,0.6);"></div>
          <div style="position:absolute; left:25%; top:55%; bottom:0; width:2px; background:rgba(126,166,255,0.6);"></div>
          <div style="position:absolute; right:25%; top:55%; bottom:0; width:2px; background:rgba(126,166,255,0.6);"></div>
        </div>
      </div>

      <div class="flow-branch">
        {_node("GA-XGBoost", "tree-based direction classifier", "green")}
        {_node("LSTM", f"{window}-day sequential deep-learning model", "green")}
      </div>

      <div class="branch-connector">
        <div style="position:relative; width:70%; max-width:560px; height:24px; margin:0 auto;">
          <div style="position:absolute; left:25%; top:0; height:45%; width:2px; background:rgba(126,166,255,0.6);"></div>
          <div style="position:absolute; right:25%; top:0; height:45%; width:2px; background:rgba(126,166,255,0.6);"></div>
          <div style="position:absolute; left:25%; right:25%; top:45%; height:2px; background:rgba(126,166,255,0.6);"></div>
          <div style="position:absolute; left:50%; top:45%; bottom:0; width:2px; background:rgba(126,166,255,0.6);"></div>
        </div>
      </div>

      <div class="flow-node green">
        <div class="fn-title">Next-Day Direction &nbsp;·&nbsp; UP / DOWN</div>
        <div class="fn-sub">P(UP) from each model compared against its threshold</div>
      </div>
    </div>
    <div class="branch-connector" style="text-align:center; color:#7C8CA8; font-size:0.8rem; margin-top:8px;">
      Both models run on every inference — neither is retrained by the app.
    </div>
    """


def _info_cards(config: dict) -> str:
    split = config.get("split", {})
    ga = config.get("ga", {})
    lstm = config.get("lstm", {})
    xgb_p = config.get("xgb_params", {})
    thr = config.get("thresholds", {})
    fe = config.get("feature_engineering", {})

    items = [
        (
            "Dataset",
            f"<b>NSE historical stock data</b><br>"
            f"Source: <code>{config.get('dataset_handle', '—')}</code><br>"
            f"File: {config.get('data_file', '—')} · symbol column: {config.get('symbol_column', 'Symbol')}<br>"
            f"{config.get('universe_size')} stocks · split "
            f"{int(split.get('train', 0.7) * 100)}/{int(split.get('val', 0.15) * 100)}/{int(split.get('test', 0.15) * 100)} "
            f"per stock (train/val/test), {split.get('embargo_rows', 1)}-row embargo",
        ),
        (
            "Preprocessing",
            "Cleaning, sorting by date, duplicate-row removal and OHLCV validation, "
            "followed by the project's feature engineering. The app applies the exact same "
            "cleaning to Yahoo/CSV input before inference.",
        ),
        (
            "Feature Engineering",
            f"<b>{config.get('n_features_before_ga')} relative / scale-aware features</b> from "
            f"<code>feature_engineering.py</code> v{fe.get('version', '1.0')}<br>"
            f"Warm-up: up to {fe.get('max_warmup_rows', '—')} rows · minimum rows for inference: "
            f"{fe.get('min_rows_needed_for_lstm_inference', '—')}<br>"
            "Returns, moving-average distances, RSI, MACD, Bollinger, ADX, MFI and volume features.",
        ),
        (
            "GA (Genetic Algorithm)",
            f"Population {ga.get('population')} · {ga.get('generations')} generations · "
            f"crossover {ga.get('crossover')} · mutation {ga.get('mutation')}<br>"
            f"Fitness: {ga.get('fitness')}<br>"
            f"Best validation accuracy recorded: <b>{fmt_pct(ga.get('best_validation_accuracy', 0), 2)}</b><br>"
            f"{config.get('n_features_before_ga')} → <b>{config.get('n_features_after_ga')} features</b>",
        ),
        (
            "XGBoost",
            f"Tree-based binary classifier (GA-selected features)<br>"
            f"n_estimators {xgb_p.get('n_estimators')} · max_depth {xgb_p.get('max_depth')} · "
            f"learning_rate {xgb_p.get('learning_rate')}<br>"
            f"Decision threshold: <b>{fmt_pct(thr.get('GA-XGBoost', 0.5), 1)}</b> (from universal_config.json)",
        ),
        (
            "LSTM",
            f"Deep sequential model<br>Architecture: <code>{lstm.get('architecture', '—')}</code><br>"
            f"Window {lstm.get('window')} days · clip ±{lstm.get('clip_z')}σ · trained {lstm.get('epochs_trained')} epochs<br>"
            f"Decision threshold: <b>{fmt_pct(thr.get('Universal LSTM', 0.5), 1)}</b> (from universal_config.json)",
        ),
        (
            "Output",
            f"<b>Next-day UP / DOWN direction</b><br>{config.get('target_definition', '')}<br>"
            "Each model outputs P(UP); the direction follows the stored threshold. "
            "This is a direction estimate — not a forecast of the future price.",
        ),
    ]

    cards_html = "".join(card(title, f'<div style="font-size:0.86rem; line-height:1.65; color:#B9C6DC;">{body}</div>')
                         for title, body in items)
    return f'<div class="tech-grid">{cards_html}</div>'


def render_methodology_page(models, train_through: str):
    _, _, _, _, config, _ = models

    section_title("Methodology", config.get("project_title", ""))
    st.markdown(
        card(
            "Prediction Pipeline",
            _flow(config),
        ),
        unsafe_allow_html=True,
    )

    st.markdown("", unsafe_allow_html=True)
    section_title("How It Works", "Each stage of the project, as implemented in the notebook and app")
    st.markdown(_info_cards(config), unsafe_allow_html=True)

    st.markdown("", unsafe_allow_html=True)
    section_title("Important Information")
    st.markdown(
        f"""
        <div class="warn-box">
          <div class="box-title">⚠ Model limitation</div>
          The trained model was trained using historical data through <b>{pretty_date(train_through)}</b>.
          Predictions made using newer market data are inference on a historically trained model and
          do <b>not</b> mean the model has been retrained on recent market conditions.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-box">
          <div class="box-title">Interpretation</div>
          {config.get("disclaimer", "")} The app displays directional probabilities, thresholds and
          model agreement so results can be interpreted honestly — never as guaranteed returns or
          exact future prices.
        </div>
        """,
        unsafe_allow_html=True,
    )
