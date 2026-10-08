"""Shared visual language for the NSE AI Stock Analyzer dashboard.

Dark professional theme, reusable card builders and small pure helpers
(consensus / signal strength) used by all four pages. No model logic here.
"""

import pandas as pd
import streamlit as st

GREEN = "#22C55E"
RED = "#EF4444"
AMBER = "#F59E0B"
BLUE = "#3B82F6"
MUTED = "#93A4C0"
TEXT = "#E8EEF9"

_TONES = {
    "green": ("rgba(34,197,94,0.10)", "rgba(34,197,94,0.35)", GREEN),
    "red": ("rgba(239,68,68,0.10)", "rgba(239,68,68,0.35)", RED),
    "amber": ("rgba(245,158,11,0.10)", "rgba(245,158,11,0.35)", AMBER),
    "blue": ("rgba(59,130,246,0.10)", "rgba(59,130,246,0.35)", BLUE),
    "neutral": ("rgba(148,163,184,0.07)", "rgba(148,163,184,0.18)", "#7FA6E8"),
}


# --------------------------------------------------------------------------- theme
_GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#070B14; --panel:rgba(255,255,255,0.035); --border:rgba(148,163,184,0.16);
  --text:#E8EEF9; --muted:#93A4C0;
  --green:#22C55E; --red:#EF4444; --amber:#F59E0B; --blue:#3B82F6;
}
html, body { background:#070B14 !important; }
body, .stApp {
  font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
  color: var(--text);
}
.stApp {
  background:
    radial-gradient(1200px 620px at 12% -8%, rgba(37,99,235,0.20), transparent 60%),
    radial-gradient(1000px 520px at 97% -4%, rgba(14,116,144,0.16), transparent 55%),
    linear-gradient(180deg, #070B14 0%, #080D1A 45%, #070B14 100%) !important;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stMainBlockContainer"] { max-width: 1280px; padding: 1.0rem 2rem 5rem; }
.block-container { max-width: 1280px; padding-top: 1rem; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0B1220 0%, #080D18 100%);
  border-right: 1px solid rgba(148,163,184,0.14);
}
[data-testid="stSidebar"] > div { padding-left: 1.2rem; padding-right: 1.2rem; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p { color: #C7D3E8; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#EEF3FB !important; }

/* ---- headings & text ---- */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color:#F1F5FC; letter-spacing:-0.01em; }
p, li { color:#B9C6DC; }
a { color:#7EA6FF; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted); }
hr { border-color: rgba(148,163,184,0.15); }

/* ---- buttons ---- */
[data-testid="stButton"] > button {
  border-radius: 10px !important;
  border: 1px solid rgba(148,163,184,0.28) !important;
  background: rgba(255,255,255,0.04) !important;
  color: #DCE6F5 !important;
  font-weight: 600 !important;
  transition: all .15s ease-in-out;
}
[data-testid="stButton"] > button:hover {
  border-color: rgba(126,166,255,0.6) !important;
  color: #FFFFFF !important;
  background: rgba(59,130,246,0.14) !important;
}
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stButton"] > button[data-kind="primary"],
[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"],
button[kind="primary"] {
  background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
  border: 1px solid rgba(96,165,250,0.55) !important;
  color: #FFFFFF !important;
  box-shadow: 0 8px 22px rgba(37,99,235,0.35);
}

/* ---- segmented top navigation ---- */
[data-testid="stSegmentedControl"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 12px;
  padding: 4px;
}
[data-testid="stSegmentedControl"] button,
[data-testid="stSegmentedControlItem"] {
  border-radius: 9px !important;
  font-weight: 600 !important;
  font-size: 0.90rem !important;
  color: #9FB0CC !important;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[data-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[data-state="active"],
div[data-testid="stSegmentedControl"] [data-testid="stSegmentedControlItem"][aria-checked="true"] {
  background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
  color: #FFFFFF !important;
  border-color: transparent !important;
  box-shadow: 0 6px 16px rgba(37,99,235,0.35);
}

/* ---- form widgets ---- */
[data-testid="stSelectbox"], [data-testid="stRadio"], [data-testid="stUploader"] { color:#C7D3E8; }
[data-testid="stSelectbox"] label, [data-testid="stRadio"] label,
[data-testid="stTextInput"], label { color: #C7D3E8 !important; font-weight: 500; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
  background: rgba(255,255,255,0.04) !important;
  border-color: rgba(148,163,184,0.28) !important;
  border-radius: 10px !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"] {
  background-color: #0E1729 !important;
  border: 1px solid rgba(148,163,184,0.25) !important;
  border-radius: 10px !important;
}
li[role="option"] {
  color: #E8EEF9 !important;
  background-color: transparent !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
  background-color: rgba(59,130,246,0.22) !important;
  color: #FFFFFF !important;
}
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 14px;
}
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(148,163,184,0.14);
  border-radius: 16px;
  padding: 10px 6px 4px;
}
[data-testid="stDataFrame"] {
  border: 1px solid rgba(148,163,184,0.14);
  border-radius: 14px;
  overflow: hidden;
}
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid rgba(148,163,184,0.18); }

/* ---- custom cards ---- */
.card {
  background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 10px 30px rgba(2,6,16,0.45);
  height: 100%;
  position: relative;
  overflow: hidden;
}
.card.tone-green { border-color: rgba(34,197,94,0.38); background: linear-gradient(160deg, rgba(34,197,94,0.10), rgba(34,197,94,0.03)); }
.card.tone-red { border-color: rgba(239,68,68,0.38); background: linear-gradient(160deg, rgba(239,68,68,0.10), rgba(239,68,68,0.03)); }
.card.tone-amber { border-color: rgba(245,158,11,0.38); background: linear-gradient(160deg, rgba(245,158,11,0.10), rgba(245,158,11,0.03)); }
.card.tone-blue { border-color: rgba(59,130,246,0.38); background: linear-gradient(160deg, rgba(59,130,246,0.10), rgba(59,130,246,0.03)); }
.card-title {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  color: #8FA3C0; margin-bottom: 10px; display:flex; align-items:center; gap:8px; justify-content: space-between;
}
.card-label { font-size: 0.78rem; color: var(--muted); font-weight: 600; letter-spacing: 0.04em; }
.kpi-value { font-size: 1.65rem; font-weight: 800; color: #F4F8FF; line-height: 1.15; margin-top: 4px; }
.kpi-sub { font-size: 0.78rem; color: var(--muted); margin-top: 6px; }
.badge {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
}
.badge.green { background: rgba(34,197,94,0.16); color: #6EE7A0; border: 1px solid rgba(34,197,94,0.4); }
.badge.red { background: rgba(239,68,68,0.16); color: #FCA5A5; border: 1px solid rgba(239,68,68,0.4); }
.badge.amber { background: rgba(245,158,11,0.16); color: #FCD34D; border: 1px solid rgba(245,158,11,0.4); }
.badge.blue { background: rgba(59,130,246,0.16); color: #93C5FD; border: 1px solid rgba(59,130,246,0.4); }
.badge.neutral { background: rgba(148,163,184,0.14); color: #C3CFE4; border: 1px solid rgba(148,163,184,0.3); }

/* ---- brand header ---- */
.brand-bar { display:flex; align-items:flex-start; justify-content:space-between; gap:1.5rem; flex-wrap: wrap; margin-bottom: 0.4rem; }
.brand-title {
  font-size: 1.75rem; font-weight: 800; letter-spacing: 0.055em;
  background: linear-gradient(90deg, #F5F9FF 0%, #9EC1FF 55%, #3B82F6 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  line-height: 1.15;
}
.brand-sub { color: #8FA3C0; font-size: 0.92rem; font-weight: 500; margin-top: 4px; }
.brand-chips { display:flex; gap: 10px; justify-content:flex-end; flex-wrap: wrap; }
.chip {
  background: rgba(255,255,255,0.045); border: 1px solid rgba(148,163,184,0.2);
  border-radius: 12px; padding: 8px 14px; text-align: right; min-width: 120px;
}
.chip .chip-label { display:block; font-size:0.66rem; letter-spacing:0.14em; color:#8FA3C0; text-transform:uppercase; font-weight:700; }
.chip .chip-value { display:block; font-size:1.0rem; font-weight:700; color:#F4F8FF; margin-top:2px; }

/* ---- limitation strip ---- */
.limit-strip {
  display:flex; gap:10px; align-items:flex-start;
  background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.30);
  border-left: 4px solid #3B82F6;
  color: #C9D8F5; border-radius: 12px; padding: 10px 14px;
  font-size: 0.82rem; line-height: 1.45; margin: 6px 0 14px;
}
.limit-strip b { color: #EAF1FF; }
.warn-box {
  background: rgba(245,158,11,0.09); border: 1px solid rgba(245,158,11,0.35);
  border-left: 4px solid #F59E0B; border-radius: 14px; padding: 16px 18px; color:#F6E3C0;
}
.warn-box .box-title { font-weight: 800; color: #FCD34D; letter-spacing:0.06em; text-transform:uppercase; font-size:0.78rem; margin-bottom:6px; }
.info-box {
  background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.32);
  border-left: 4px solid #3B82F6; border-radius: 14px; padding: 16px 18px; color:#C9D8F5;
}
.info-box .box-title { font-weight: 800; color: #93C5FD; letter-spacing:0.06em; text-transform:uppercase; font-size:0.78rem; margin-bottom:6px; }

/* ---- section titles ---- */
.section-title { font-size: 1.05rem; font-weight: 750; color:#F1F5FC; letter-spacing:-0.01em; margin-top: 0.6rem; }
.section-sub { font-size: 0.82rem; color: var(--muted); margin-bottom: 10px; }

/* ---- prediction hero ---- */
.pred-hero {
  border-radius: 22px; padding: 26px 26px; border: 1px solid rgba(148,163,184,0.2);
  background: linear-gradient(150deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018));
  box-shadow: 0 18px 44px rgba(2,6,16,0.55);
}
.pred-hero.up { border-color: rgba(34,197,94,0.45); background: linear-gradient(150deg, rgba(34,197,94,0.13), rgba(255,255,255,0.02) 55%); }
.pred-hero.down { border-color: rgba(239,68,68,0.45); background: linear-gradient(150deg, rgba(239,68,68,0.13), rgba(255,255,255,0.02) 55%); }
.pred-grid { display:flex; gap: 26px; flex-wrap: wrap; align-items: center; }
.pred-dir-wrap { min-width: 240px; text-align:center; padding: 8px 6px; }
.pred-eyebrow { font-size:0.72rem; letter-spacing:0.18em; text-transform:uppercase; color:#8FA3C0; font-weight:700; }
.pred-dir { font-size: 4.2rem; font-weight: 850; line-height: 1.02; letter-spacing: 0.02em; margin-top: 8px; }
.pred-dir.up { color: #4ADE80; text-shadow: 0 0 34px rgba(34,197,94,0.45); }
.pred-dir.down { color: #F87171; text-shadow: 0 0 34px rgba(239,68,68,0.45); }
.pred-dir-sub { font-size:0.85rem; color:#A9B8D1; margin-top:8px; font-weight:600; }
.model-row { display:flex; gap: 16px; flex-wrap: wrap; flex: 1 1 420px; }
.model-box {
  flex: 1 1 210px;
  background: rgba(8,13,26,0.65); border:1px solid rgba(148,163,184,0.2);
  border-radius: 16px; padding: 16px 18px;
}
.model-box .mb-name { font-size:0.78rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#9FB0CC; }
.model-box .mb-prob { font-size:1.9rem; font-weight:800; color:#F4F8FF; margin-top:6px; }
.model-box .mb-meta { font-size:0.80rem; color:#A9B8D1; margin-top:6px; line-height:1.6; }
.model-box .mb-dir { margin-top: 10px; }
.bar-track {
  position:relative; height: 8px; border-radius: 999px; margin-top:12px;
  background: rgba(148,163,184,0.18); overflow: visible;
}
.bar-fill { position:absolute; left:0; top:0; bottom:0; border-radius:999px; }
.bar-thr { position:absolute; top:-4px; bottom:-4px; width:2px; background:#E8EEF9; opacity:.85; border-radius:2px; }
.bar-scale { display:flex; justify-content:space-between; font-size:0.66rem; color:#7C8CA8; margin-top:5px; }

/* ---- summary card ---- */
.summary-row {
  display:flex; justify-content: space-between; gap: 12px;
  padding: 9px 2px; border-bottom: 1px dashed rgba(148,163,184,0.16); font-size: 0.88rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-row .s-label { color: #93A4C0; font-weight: 500; }
.summary-row .s-value { color: #F1F5FC; font-weight: 700; text-align:right; }
.note-line { font-size: 0.80rem; color: var(--muted); font-style: italic; margin-top: 10px; }

/* ---- methodology flow ---- */
.flow { display:flex; flex-direction: column; align-items:center; gap: 6px; padding: 12px 0 4px; }
.flow-node {
  background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.025));
  border: 1px solid rgba(148,163,184,0.25); border-radius: 14px;
  padding: 12px 22px; text-align:center; min-width: 240px; max-width: 100%;
}
.flow-node .fn-title { font-weight: 700; color: #F1F5FC; font-size: 0.95rem; }
.flow-node .fn-sub { font-size: 0.76rem; color: #93A4C0; margin-top: 3px; }
.flow-node.accent { border-color: rgba(59,130,246,0.5); background: linear-gradient(160deg, rgba(59,130,246,0.14), rgba(59,130,246,0.04)); }
.flow-node.green { border-color: rgba(34,197,94,0.5); background: linear-gradient(160deg, rgba(34,197,94,0.14), rgba(34,197,94,0.04)); }
.flow-arrow { color:#7EA6FF; font-size: 1.15rem; line-height: 1; text-align:center; }
.flow-branch { display:flex; gap: 46px; justify-content:center; flex-wrap: wrap; align-items: stretch; }
.flow-branch .flow-node { flex: 1 1 230px; max-width: 340px; }
.flow-split { width: 2px; height: 20px; background: linear-gradient(180deg, rgba(126,166,255,0.1), rgba(126,166,255,0.7)); margin: 0 auto; }

/* ---- misc ---- */
.tech-grid { display:flex; gap: 14px; flex-wrap: wrap; }
.tech-grid .card { flex: 1 1 180px; }
@media (max-width: 760px) {
  .pred-dir { font-size: 3.2rem; }
  .brand-title { font-size: 1.35rem; }
  .flow-branch { flex-direction: column; align-items: center; }
  [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
}
"""


def inject_global_styles():
    st.markdown(f"<style>\n{_GLOBAL_CSS}\n</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
def _html(fragment: str):
    st.markdown(fragment, unsafe_allow_html=True)


def pretty_date(value) -> str:
    """'2023-12-29' or Timestamp -> '29 Dec 2023'."""
    if value is None:
        return "—"
    try:
        return pd.Timestamp(value).strftime("%d %b %Y")
    except Exception:
        return str(value)


def fmt_inr(x: float) -> str:
    return f"₹{x:,.2f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="badge {tone}">{text}</span>'


def render_top_header(stock: str, data_date):
    _html(
        f"""
        <div class="brand-bar">
          <div>
            <div class="brand-title">NSE AI STOCK ANALYZER</div>
            <div class="brand-sub">Universal GA-XGBoost + LSTM &nbsp;•&nbsp; Next-Day Market Direction Prediction</div>
          </div>
          <div class="brand-chips">
            <div class="chip"><span class="chip-label">Selected Stock</span><span class="chip-value">{stock}</span></div>
            <div class="chip"><span class="chip-label">Latest Data Date</span><span class="chip-value">{pretty_date(data_date)}</span></div>
          </div>
        </div>
        """
    )


def render_limitation_strip(train_through_pretty: str):
    _html(
        f"""
        <div class="limit-strip">
          <span>ⓘ</span>
          <span><b>Model limitation —</b> the trained model was fitted on historical data through
          <b>{train_through_pretty}</b>. Predictions made with newer market data are inference on a
          historically trained model — it has <b>not</b> been retrained on recent market conditions.</span>
        </div>
        """
    )


def kpi_card(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    bg, border, accent = _TONES.get(tone, _TONES["neutral"])
    return f"""
    <div class="card" style="background:{bg}; border-color:{border};">
      <div class="card-label">{label}</div>
      <div class="kpi-value" style="color:{accent if tone != 'neutral' else '#F4F8FF'};">{value}</div>
      {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
    </div>
    """


def card(title: str, body_html: str = "", tone: str = "", right_html: str = "") -> str:
    tone_cls = f"tone-{tone}" if tone else ""
    title_html = f'<div class="card-title"><span>{title}</span><span>{right_html}</span></div>' if title else ""
    return f'<div class="card {tone_cls}">{title_html}{body_html}</div>'


def summary_row(label: str, value: str) -> str:
    return f'<div class="summary-row"><span class="s-label">{label}</span><span class="s-value">{value}</span></div>'


def section_title(title: str, sub: str = ""):
    sub_html = f'<div class="section-sub">{sub}</div>' if sub else ""
    _html(f'<div class="section-title">{title}</div>{sub_html}')


# --------------------------------------------------------------------------- pure logic helpers
def consensus_info(result: dict) -> dict:
    """Model-consensus wording + tone (pure display logic, no model maths)."""
    agree = result["models_agree"]
    direction = result["xgb_direction"]
    if agree:
        tone = "green" if direction == "UP" else "red"
        headline = f"Both trained models indicate {direction}"
        detail = "XGBoost and LSTM agree on the next-day direction."
    else:
        tone = "amber"
        headline = "Model disagreement"
        detail = "The two trained models point in opposite directions."
    return {
        "agree": agree,
        "direction": direction if agree else "—",
        "tone": tone,
        "headline": headline,
        "detail": detail,
    }


def signal_strength(result: dict) -> dict:
    """Distance of each P(UP) from its own decision threshold.

    Uses only the existing model probabilities and thresholds — no new
    probability model is introduced. Labels stay conservative: anything
    within 3 percentage points of its threshold is a "Weak" signal.
    """
    xgb_margin = abs(result["xgb_prob"] - result["xgb_threshold"]) * 100.0
    lstm_margin = abs(result["lstm_prob"] - result["lstm_threshold"]) * 100.0
    margin = min(xgb_margin, lstm_margin)

    if not result["models_agree"]:
        return {
            "level": "Mixed",
            "tone": "amber",
            "label": "Mixed signal — models disagree",
            "margin_pp": margin,
            "detail": (
                f"Model margins: XGBoost {xgb_margin:.1f} pp, LSTM {lstm_margin:.1f} pp. "
                "With no consensus, treat the output as an experimental model signal."
            ),
        }

    level = "Weak" if margin < 3.0 else ("Moderate" if margin < 6.0 else "Stronger")
    tone = "amber" if level == "Weak" else ("blue" if level == "Moderate" else "green")
    direction = "bullish" if result["xgb_direction"] == "UP" else "bearish"
    return {
        "level": level,
        "tone": tone,
        "label": f"{level} {direction} signal",
        "margin_pp": margin,
        "detail": (
            f"Proximity to the decision threshold — XGBoost {xgb_margin:.1f} pp, "
            f"LSTM {lstm_margin:.1f} pp (closest {margin:.1f} pp)."
        ),
    }
