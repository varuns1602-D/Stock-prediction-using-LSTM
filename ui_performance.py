"""Model Performance page.

Two sources of truth are shown, clearly separated:

  1. RECORDED artifacts — values that already exist in the project files
     (universal_config.json, latest_predictions_all_stocks.csv,
     feature_importance.csv). Read directly, never recomputed, never invented.
  2. RUNTIME REPLAY — the saved models are re-run over downloaded Yahoo
     history (no retraining) to produce genuine accuracy / precision /
     recall / F1 / ROC-AUC, confusion matrices and per-stock accuracy.

Metrics that exist in neither source are displayed as "not recorded"
instead of being fabricated.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_ui import card, fmt_pct, kpi_card, section_title
from model_replay import run_evaluation
from prediction_backend import MODEL_DIR, load_fallback_latest

GRID = "rgba(148,163,184,0.12)"
METRIC_ROWS = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1"),
    ("ROC-AUC", "roc_auc"),
]


def _style(fig: go.Figure, height: int):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        font=dict(family="Inter, sans-serif", color="#B9C6DC", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0E1729", font_color="#E8EEF9", bordercolor="rgba(148,163,184,0.4)"),
    )
    fig.update_xaxes(gridcolor=GRID, showline=False)
    fig.update_yaxes(gridcolor=GRID, showline=False, zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def _feature_importance() -> pd.DataFrame:
    path = MODEL_DIR / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["Feature", "Importance"])


def _fmt_metric(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value * 100:.1f}%"


def _pct_or_none(v):
    return None if v is None else v * 100


# --------------------------------------------------------------------------- sections
def _render_interpretation():
    st.markdown(
        """
        <div class="warn-box">
          <div class="box-title">Important interpretation</div>
          Predicting the <b>next-day direction</b> of an individual stock is genuinely difficult — a
          majority-class baseline is already close to 50%. Results near 50% represent a small, noisy
          edge and must <b>not</b> be presented as highly certain. This dashboard reports the project's
          real numbers as measured; no accuracy above ~90% is claimed anywhere, and probabilities close
          to the decision threshold are labelled as weak signals. Educational project only — not
          investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_cards(res: dict):
    m = res["metrics"]
    thr = res["thresholds"]
    c1, c2, c3 = st.columns(3)

    with c1:
        body = (
            '<div style="font-size:0.83rem; color:#93A4C0; margin-bottom:6px;">Accuracy on replayed held-out rows</div>'
            f'<div class="kpi-value">{_fmt_metric(m["xgb"]["accuracy"])}</div>'
            f'<div class="kpi-sub">ROC-AUC {_fmt_metric(m["xgb"]["roc_auc"])} · F1 {_fmt_metric(m["xgb"]["f1"])}<br>'
            f'Threshold {fmt_pct(thr["xgb"], 1)} · GA-selected features</div>'
        )
        st.markdown(card("GA-XGBoost", body, tone="blue"), unsafe_allow_html=True)
    with c2:
        body = (
            '<div style="font-size:0.83rem; color:#93A4C0; margin-bottom:6px;">Accuracy on replayed held-out rows</div>'
            f'<div class="kpi-value">{_fmt_metric(m["lstm"]["accuracy"])}</div>'
            f'<div class="kpi-sub">ROC-AUC {_fmt_metric(m["lstm"]["roc_auc"])} · F1 {_fmt_metric(m["lstm"]["f1"])}<br>'
            f'Threshold {fmt_pct(thr["lstm"], 1)} · 60-day sequences</div>'
        )
        st.markdown(card("Universal LSTM", body, tone="blue"), unsafe_allow_html=True)
    with c3:
        body = (
            '<div style="font-size:0.83rem; color:#93A4C0; margin-bottom:6px;">Always predicts the majority class</div>'
            f'<div class="kpi-value">{_fmt_metric(m["baseline"]["accuracy"])}</div>'
            f'<div class="kpi-sub">Majority class: {thr["baseline_class"]} · {thr["reference_rows"]:,} reference rows'
            '<br>ROC-AUC — (constant scores carry no ranking)</div>'
        )
        st.markdown(card("Majority Baseline", body), unsafe_allow_html=True)


def _metrics_table(res: dict):
    header = (
        '<div class="summary-row" style="border-bottom:1px solid rgba(148,163,184,0.3);">'
        '<span class="s-label" style="color:#7C8CA8;">METRIC</span>'
        '<span class="s-value" style="display:flex; gap:26px; color:#7C8CA8; font-size:0.72rem; '
        'letter-spacing:0.08em;">'
        '<span style="min-width:74px; text-align:right;">XGBOOST</span>'
        '<span style="min-width:74px; text-align:right;">LSTM</span>'
        '<span style="min-width:74px; text-align:right;">BASELINE</span></span></div>'
    )
    rows = ""
    for label, key in METRIC_ROWS:
        rows += (
            f'<div class="summary-row"><span class="s-label">{label}</span>'
            '<span class="s-value" style="display:flex; gap:26px;">'
            f'<span style="min-width:74px; text-align:right;">{_fmt_metric(res["metrics"]["xgb"][key])}</span>'
            f'<span style="min-width:74px; text-align:right;">{_fmt_metric(res["metrics"]["lstm"][key])}</span>'
            f'<span style="min-width:74px; text-align:right;">{_fmt_metric(res["metrics"]["baseline"][key])}</span>'
            '</span></div>'
        )
    st.markdown(card("Metric Comparison", header + rows + (
        '<div class="note-line">Computed at runtime by replaying the saved models over downloaded '
        'history — these values are not stored in the project artifacts.</div>'
    )), unsafe_allow_html=True)


def _comparison_chart(res: dict):
    metrics_labels = [m[0] for m in METRIC_ROWS]
    fig = go.Figure()
    for name, key, color in (
        ("GA-XGBoost", "xgb", "#3B82F6"),
        ("Universal LSTM", "lstm", "#22C55E"),
        ("Majority Baseline", "baseline", "#64748B"),
    ):
        vals = [_pct_or_none(res["metrics"][key][k]) for _, k in METRIC_ROWS]
        fig.add_trace(go.Bar(
            name=name, x=metrics_labels, y=vals, marker_color=color,
            text=[None if v is None else f"{v:.1f}%" for v in vals],
            textposition="outside", textfont=dict(color="#C9D8F5"),
        ))
    fig.update_layout(barmode="group", title=dict(text="<b>Model vs baseline</b>", x=0.02, font=dict(size=14)))
    fig.update_yaxes(title_text="%", range=[0, 100], nticks=6)
    _style(fig, 400)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _roc_figure(res: dict):
    auc_x = _fmt_metric(res["metrics"]["xgb"]["roc_auc"])
    auc_l = _fmt_metric(res["metrics"]["lstm"]["roc_auc"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color="#64748B", width=1.5, dash="dash"), showlegend=False))
    fig.add_trace(go.Scatter(x=res["roc"]["xgb"]["fpr"], y=res["roc"]["xgb"]["tpr"], mode="lines",
                             name=f"GA-XGBoost (AUC {auc_x})",
                             line=dict(color="#3B82F6", width=2.4)))
    fig.add_trace(go.Scatter(x=res["roc"]["lstm"]["fpr"], y=res["roc"]["lstm"]["tpr"], mode="lines",
                             name=f"Universal LSTM (AUC {auc_l})",
                             line=dict(color="#22C55E", width=2.4)))
    fig.update_layout(title=dict(text="<b>ROC curve (pooled replayed predictions)</b>", x=0.02, font=dict(size=14)))
    fig.update_xaxes(title_text="False positive rate")
    fig.update_yaxes(title_text="True positive rate", range=[0, 1])
    _style(fig, 420)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _confusion_figures(res: dict):
    labels = ["DOWN", "UP"]
    for name, key, color in (
        ("GA-XGBoost — confusion matrix", "xgb", "#3B82F6"),
        ("Universal LSTM — confusion matrix", "lstm", "#22C55E"),
    ):
        cm = np.array(res["confusion"][key])
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "rgba(255,255,255,0.02)"], [1, color]],
            showscale=False,
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}", textfont=dict(size=15, color="#F4F8FF"),
            hovertemplate="actual %{y}<br>predicted %{x}<br>%{z:,} rows<extra></extra>",
        ))
        fig.update_layout(title=dict(text=f"<b>{name}</b>", x=0.02, font=dict(size=13)),
                          yaxis=dict(autorange="reversed"))
        fig.update_xaxes(title_text="predicted")
        fig.update_yaxes(title_text="actual")
        _style(fig, 300)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _per_stock_chart(res: dict):
    df = pd.DataFrame(res["per_stock"])
    if df.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Box(x=df["xgb_acc"] * 100, name="GA-XGBoost", marker_color="#3B82F6",
                         fillcolor="rgba(59,130,246,0.35)", line=dict(color="#93C5FD")))
    fig.add_trace(go.Box(x=df["lstm_acc"] * 100, name="Universal LSTM", marker_color="#22C55E",
                         fillcolor="rgba(34,197,94,0.35)", line=dict(color="#6EE7A0")))
    fig.update_layout(title=dict(text="<b>Per-stock accuracy distribution</b>", x=0.02, font=dict(size=14)),
                      boxmode="group", showlegend=False)
    fig.update_xaxes(title_text="accuracy (%)", nticks=8)
    _style(fig, 300)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption(
        f"{res['stocks_evaluated']} of {res['stocks_requested']} stocks evaluated "
        f"({res['rows']:,} held-out rows, {res['date_min']} → {res['date_max']}); "
        f"UP share of labels: {res['up_rate'] * 100:.1f}%. "
        f"{len(res.get('stocks_skipped', []))} stock(s) skipped (insufficient history)."
    )


def _recorded_section(config: dict, train_through: str):
    section_title(
        "Recorded Project Results",
        "Values already stored in the saved project artifacts — read directly, never recomputed",
    )
    snap = load_fallback_latest()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card("GA best validation accuracy",
                     fmt_pct(config["ga"]["best_validation_accuracy"], 2),
                     sub="recorded in universal_config.json"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card("Recorded thresholds",
                     f'{fmt_pct(config["thresholds"]["GA-XGBoost"], 1)} / '
                     f'{fmt_pct(config["thresholds"]["Universal LSTM"], 1)}',
                     sub="XGBoost / LSTM · threshold mode: val balanced-accuracy"),
            unsafe_allow_html=True,
        )
    if not snap.empty:
        agree_pct = float((snap["xgb_direction"] == snap["lstm_direction"]).mean())
        with c3:
            st.markdown(
                kpi_card("Snapshot model agreement", f"{agree_pct * 100:.1f}%",
                         sub=f"{len(snap)} stocks · snapshot {snap['Date'].max()}",
                         tone="green" if agree_pct >= 0.6 else "amber"),
                unsafe_allow_html=True,
            )
        with c4:
            up_xgb = float((snap["xgb_direction"] == "UP").mean())
            up_lstm = float((snap["lstm_direction"] == "UP").mean())
            st.markdown(
                kpi_card("Snapshot UP calls", f"{up_xgb * 100:.0f}% / {up_lstm * 100:.0f}%",
                         sub="XGBoost / LSTM share of UP calls"),
                unsafe_allow_html=True,
            )
    else:
        with c3:
            st.markdown(kpi_card("Snapshot agreement", "—", sub="snapshot CSV not found"), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi_card("Snapshot UP calls", "—", sub="snapshot CSV not found"), unsafe_allow_html=True)

    st.markdown("", unsafe_allow_html=True)

    fi = _feature_importance()
    left, right = st.columns([3, 2])
    with left:
        if not fi.empty:
            top = fi.sort_values("Importance", ascending=True).tail(15)
            fig = go.Figure(go.Bar(
                x=top["Importance"], y=top["Feature"], orientation="h",
                marker_color="#3B82F6", marker_line_color="rgba(126,166,255,0.6)", marker_line_width=0.6,
                hovertemplate="%{y}<br>importance %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(title=dict(text="<b>Feature importance (saved artifact · top 15)</b>",
                                         x=0.02, font=dict(size=14)),
                              showlegend=False, yaxis=dict(tickfont=dict(size=11)))
            fig.update_xaxes(title_text="importance")
            _style(fig, 430)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        else:
            st.info("feature_importance.csv was not found in model_artifacts.")

    with right:
        body = (
            '<div style="font-size:0.86rem; line-height:1.7; color:#B9C6DC;">'
            f'<b>Training data through</b> {train_through}<br>'
            f'<b>Training universe</b> {config["universe_size"]} stocks<br>'
            f'<b>Features</b> {config["n_features_before_ga"]} → {config["n_features_after_ga"]} after GA<br>'
            f'<b>LSTM window</b> {config["lstm"]["window"]} days · {config["lstm"]["epochs_trained"]} epochs<br>'
            f'<b>GA</b> population {config["ga"]["population"]} × {config["ga"]["generations"]} generations · '
            f'fitness: {config["ga"]["fitness"]}<br>'
            f'<b>Leakage audit</b> {"passed" if config.get("leakage_audit_all_passed") else "not recorded"}<br>'
            f'<b>Split</b> {int(config["split"]["train"] * 100)}/{int(config["split"]["val"] * 100)}/'
            f'{int(config["split"]["test"] * 100)} per stock · {config["split"]["embargo_rows"]}-row embargo'
            '</div>'
            '<div class="note-line">Ablation results, saved test-set precision/recall/F1/ROC-AUC and saved '
            'confusion-matrix/ROC images are <b>not present</b> in the saved artifacts — that is why the '
            'metrics above are produced by a runtime replay of the same saved models instead of being '
            'invented.</div>'
        )
        st.markdown(card("Saved Training Record", body, tone="blue"), unsafe_allow_html=True)


# --------------------------------------------------------------------------- page
def render_performance_page(models, stocks_meta, train_through: str):
    config = models[4]
    tickers = sorted({x["ticker"] for x in stocks_meta.get("trained_on", [])})
    st.session_state.setdefault("perf_results", {})
    st.session_state.setdefault("perf_errors", {})

    section_title(
        "Model Performance",
        "Recorded project artifacts & training results (instant) + optional live runtime evaluation",
    )
    _render_interpretation()

    # ---------------- 1. recorded artifacts (always shown immediately) ----------------
    _recorded_section(config, train_through)

    st.markdown("", unsafe_allow_html=True)
    st.divider()

    # ---------------- 2. optional live runtime replay evaluation ----------------
    section_title(
        "Fresh Runtime Evaluation",
        "Replay the saved models over downloaded Yahoo Finance history (optional & computation-heavy)",
    )
    st.markdown(
        card(
            "Fresh Replay Evaluation (Optional & Slow)",
            '<div style="font-size:0.87rem; color:#B9C6DC; line-height:1.6;">'
            'Re-runs the <b>saved</b> GA-XGBoost and LSTM models over freshly downloaded NSE history to compute '
            'live test metrics against next-day direction. The models are never retrained, replaced or '
            'modified. Market data downloads and feature engineering for multiple stocks may take 30–60+ seconds. '
            '<b>If you only want to review existing model performance, the recorded results above are already complete.</b>'
            '</div>',
            tone="blue",
        ),
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns([5, 2, 3])
    with m1:
        mode_label = st.radio(
            "Evaluation window",
            ["Held-out test split", "Forward (after training cut-off)"],
            horizontal=True,
            help="Test split: reproduces the project's 70/15/15 per-stock split with a 1-row embargo on data "
                 "through the training cut-off. Forward: rows dated after the cut-off — unseen market history.",
        )
    with m2:
        n_stocks = st.selectbox("Stocks", ["10", "25", "50", "100"], index=1)
    with m3:
        st.markdown("")
        run_clicked = st.button("▶ Run Fresh Evaluation", type="primary", width="stretch")

    mode = "test_split" if mode_label == "Held-out test split" else "forward"
    new_key = f"{mode}:{n_stocks}"
    res = st.session_state["perf_results"].get(new_key)
    error_msg = st.session_state["perf_errors"].get(new_key)

    # Replay runs ONLY when explicitly clicked by the user
    if run_clicked:
        progress_ph = st.empty()
        bar = progress_ph.progress(0.0, text="Starting fresh evaluation…")

        def _cb(pct, name):
            bar.progress(min(float(pct), 1.0), text=f"Replaying saved models — {name}")

        try:
            res = run_evaluation(models, mode, train_through, int(n_stocks), tickers, progress=_cb)
            st.session_state["perf_results"][new_key] = res
            st.session_state["perf_errors"][new_key] = None
            error_msg = None
        except Exception as exc:
            error_msg = str(exc)
            st.session_state["perf_results"][new_key] = None
            st.session_state["perf_errors"][new_key] = error_msg
            res = None
        finally:
            progress_ph.empty()

    if error_msg:
        st.error(f"Fresh evaluation failed: {error_msg}")
        st.info("Fresh evaluation failed. The recorded test results are still available above.")
    elif res is not None:
        st.success(
            f'Replay complete — {res["stocks_evaluated"]}/{res["stocks_requested"]} stocks, '
            f'{res["rows"]:,} held-out rows ({res["date_min"]} → {res["date_max"]}). '
            + (f'Download failed for {len(res["download_failed"])} stock(s). ' if res.get("download_failed") else "")
            + "Metrics computed at runtime from the saved models."
        )
        st.markdown("", unsafe_allow_html=True)

        window_label = "held-out test split" if mode == "test_split" else "forward out-of-sample"
        section_title(
            "Live Test Metrics",
            f"Pooled over the evaluated rows — {window_label} window · thresholds from universal_config.json",
        )
        _metric_cards(res)
        st.markdown("", unsafe_allow_html=True)
        _metrics_table(res)
        st.markdown("", unsafe_allow_html=True)
        _comparison_chart(res)

        left, right = st.columns(2, gap="large")
        with left:
            _roc_figure(res)
        with right:
            _confusion_figures(res)

        st.markdown("", unsafe_allow_html=True)
        section_title("Per-Stock Performance", "Accuracy spread across the evaluated stocks")
        _per_stock_chart(res)
        st.markdown("", unsafe_allow_html=True)
    else:
        st.info("To run a live replay over downloaded NSE history, click **▶ Run Fresh Evaluation** above (optional).")
