"""Prediction page — the main screen of the NSE AI Stock Analyzer.

All numbers rendered here come from prediction_backend.infer(), i.e. the
saved GA-XGBoost + LSTM models and the thresholds stored in
universal_config.json. Nothing on this page is hard-coded or estimated.
"""

import streamlit as st

from dashboard_ui import (
    card,
    consensus_info,
    fmt_inr,
    fmt_pct,
    kpi_card,
    pretty_date,
    section_title,
    signal_strength,
    summary_row,
)


def _model_box(name: str, prob: float, threshold: float, direction: str) -> str:
    tone = "green" if direction == "UP" else "red"
    fill = (
        "linear-gradient(90deg, rgba(34,197,94,0.45), #22C55E)"
        if direction == "UP"
        else "linear-gradient(90deg, rgba(239,68,68,0.45), #EF4444)"
    )
    thr_pct = threshold * 100
    prob_pct = prob * 100
    return f"""
    <div class="model-box">
      <div class="mb-name">{name}</div>
      <div class="mb-prob">{fmt_pct(prob, 2)}</div>
      <div class="mb-meta">
        P(UP): {fmt_pct(prob, 2)}<br>
        Threshold: {fmt_pct(threshold, 1)}
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{prob_pct:.2f}%; background:{fill};"></div>
        <div class="bar-thr" style="left:{thr_pct:.2f}%;"></div>
      </div>
      <div class="bar-scale"><span>0%</span><span>threshold {thr_pct:.1f}%</span><span>100%</span></div>
      <div class="mb-dir">
        <span class="badge {tone}">Prediction: {direction}</span>
      </div>
    </div>
    """


def _render_error(run: dict):
    st.markdown(
        card(
            "Prediction unavailable",
            f'<div style="color:#FCA5A5; font-size:0.95rem; line-height:1.6;">{run.get("error", "Unknown error.")}</div>'
            '<div class="note-line">Check the settings in the sidebar and press <b>Generate Prediction</b>.</div>',
            tone="red",
        ),
        unsafe_allow_html=True,
    )
    st.info(
        "For CSV input, use columns Date, Open, High, Low, Close, Volume. "
        "The file can contain one stock, or a Symbol column for multiple stocks."
    )


def _render_pending(settings: dict, message: str = ""):
    if message:
        msg = message
    elif settings["source"] == "Upload OHLCV CSV":
        msg = "Upload an OHLCV CSV to generate a prediction."
    else:
        msg = "Select a stock and click <b>⚡ Generate Prediction</b>."
    st.markdown(
        card(
            "Waiting for market data",
            f'<div style="color:#C9D8F5; font-size:0.95rem;">{msg}</div>',
            tone="blue",
        ),
        unsafe_allow_html=True,
    )


def render_prediction_page(run: dict, settings: dict):
    if run.get("pending"):
        _render_pending(settings, run.get("pending_message", ""))
        return
    if not run.get("ok"):
        _render_error(run)
        return

    result = run["result"]
    market_df = run["market_df"]
    agree = consensus_info(result)
    strength = signal_strength(result)
    hero_tone = "up" if (result["models_agree"] and result["xgb_direction"] == "UP") else (
        "down" if (result["models_agree"] and result["xgb_direction"] == "DOWN") else ""
    )

    # ---- today's change from the fetched data (previous close -> latest close) ----
    md = market_df.sort_values("Date")
    past = md[md["Date"] <= result["date"]]
    chg_abs, chg_pct = None, None
    if len(past) >= 2:
        prev_close = float(past["Close"].iloc[-2])
        chg_abs = result["close"] - prev_close
        chg_pct = chg_abs / prev_close

    # ---------------- stock overview ----------------
    section_title("Stock Overview", "Latest available market data for the selected stock")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card(
                "Stock Symbol",
                result["stock"],
                sub=f"NSE · {run.get('company') or result['stock']}",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        chg_tone = "neutral" if chg_pct is None else ("green" if chg_pct >= 0 else "red")
        st.markdown(
            kpi_card(
                "Latest Close",
                fmt_inr(result["close"]),
                sub=f"as of {pretty_date(result['date'])}",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        if chg_pct is None:
            st.markdown(kpi_card("Today's Change", "—", sub="not enough rows to compare"), unsafe_allow_html=True)
        else:
            arrow = "▲" if chg_pct >= 0 else "▼"
            st.markdown(
                kpi_card(
                    "Today's Change",
                    f"{arrow} {chg_pct * 100:+.2f}%",
                    sub=f"{chg_abs:+,.2f} vs previous close",
                    tone=chg_tone,
                ),
                unsafe_allow_html=True,
            )
    with c4:
        st.markdown(
            kpi_card("Latest Data Date", pretty_date(result["date"]), sub="most recent trading row used"),
            unsafe_allow_html=True,
        )

    st.markdown("", unsafe_allow_html=True)

    # ---------------- primary prediction card ----------------
    if result["models_agree"]:
        if result["xgb_direction"] == "UP":
            big_dir = '<div class="pred-dir up">UP ↑</div>'
            sub_line = '<div class="pred-dir-sub">Consensus of both trained models</div>'
        else:
            big_dir = '<div class="pred-dir down">DOWN ↓</div>'
            sub_line = '<div class="pred-dir-sub">Consensus of both trained models</div>'
    else:
        big_dir = '<div class="pred-dir" style="color:#FCD34D;">↑ ↓</div>'
        sub_line = '<div class="pred-dir-sub" style="color:#FCD34D;">Model disagreement — no consensus direction</div>'

    st.markdown(
        f"""
        <div class="pred-hero {hero_tone}">
          <div class="pred-grid">
            <div class="pred-dir-wrap">
              <div class="pred-eyebrow">Next-Day Prediction</div>
              {big_dir}
              {sub_line}
              <div style="margin-top:10px;">{f'<span class="badge {agree["tone"]}">{agree["headline"]}</span>'}</div>
            </div>
            <div class="model-row">
              {_model_box("XGBoost", result["xgb_prob"], result["xgb_threshold"], result["xgb_direction"])}
              {_model_box("LSTM", result["lstm_prob"], result["lstm_threshold"], result["lstm_direction"])}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("", unsafe_allow_html=True)

    # ---------------- consensus + signal strength ----------------
    left, right = st.columns(2, gap="large")

    with left:
        rows = (
            summary_row("XGBoost", f'{f"""<span class="badge {"green" if result["xgb_direction"] == "UP" else "red"}">{result["xgb_direction"]}</span>"""}')
            + summary_row("LSTM", f'{f"""<span class="badge {"green" if result["lstm_direction"] == "UP" else "red"}">{result["lstm_direction"]}</span>"""}')
        )
        body = f"""
          <div style="font-size:1.12rem; font-weight:800; color:{'#4ADE80' if agree['agree'] and agree['direction'] == 'UP' else '#F87171' if agree['agree'] else '#FCD34D'};">
            {agree['headline']}
          </div>
          <div style="font-size:0.86rem; color:#A9B8D1; margin-top:6px;">{agree['detail']}</div>
          <div style="margin-top:12px;">{rows}</div>
        """
        st.markdown(card("Model Consensus", body, tone=agree["tone"]), unsafe_allow_html=True)

    with right:
        level_tone_color = {"amber": "#FCD34D", "blue": "#93C5FD", "green": "#6EE7A0"}.get(strength["tone"], "#E8EEF9")
        body = f"""
          <div style="font-size:1.12rem; font-weight:800; color:{level_tone_color};">{strength['label']}</div>
          <div style="font-size:0.86rem; color:#A9B8D1; margin-top:6px;">{strength['detail']}</div>
          <div class="note-line">Signal strength measures how far P(UP) sits from the model's own decision
          threshold — it is directional confidence, <b>not</b> the expected size of a price move.</div>
        """
        st.markdown(card("Signal Strength", body, tone=strength["tone"]), unsafe_allow_html=True)

    st.markdown("", unsafe_allow_html=True)

    # ---------------- prediction summary ----------------
    summary_rows = "".join(
        [
            summary_row("Selected Stock", result["stock"]),
            summary_row("Latest Close", fmt_inr(result["close"])),
            summary_row(
                "XGBoost Prediction",
                f'{result["xgb_direction"]} &nbsp;·&nbsp; P(UP) {fmt_pct(result["xgb_prob"], 2)}',
            ),
            summary_row(
                "LSTM Prediction",
                f'{result["lstm_direction"]} &nbsp;·&nbsp; P(UP) {fmt_pct(result["lstm_prob"], 2)}',
            ),
            summary_row("Model Agreement", "Yes — models agree" if agree["agree"] else "No — models disagree"),
            summary_row("Signal Strength", strength["label"]),
            summary_row("Latest Data Date", pretty_date(result["date"])),
        ]
    )
    body = f"""
      {summary_rows}
      <div class="note-line">Prediction represents next-day price direction (UP/DOWN),
      not the exact future price.</div>
    """
    st.markdown(card("Prediction Summary", body), unsafe_allow_html=True)

    # ---------------- technical details ----------------
    with st.expander("Model / data details"):
        st.write(
            {
                "Selected stock": result["stock"],
                "Rows supplied": result["rows_supplied"],
                "Rows after feature warm-up": int(result["feature_rows"]),
                "Features before GA": run["config"]["n_features_before_ga"],
                "GA-selected features": run["config"]["n_features_after_ga"],
                "LSTM sequence length": run["config"]["lstm"]["window"],
                "Training universe": run["config"]["universe_size"],
                "XGBoost threshold": result["xgb_threshold"],
                "LSTM threshold": result["lstm_threshold"],
                "Threshold source": "universal_config.json",
            }
        )
