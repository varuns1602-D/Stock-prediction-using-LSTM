"""Charts page — interactive price/volume chart, market data table and
indicators already produced by the project's feature_engineering.py.

No prediction overlays are drawn here; everything is real market data or
existing backend feature outputs.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard_ui import card, kpi_card, section_title

GREEN = "#22C55E"
RED = "#EF4444"
BLUE = "#3B82F6"
GRID = "rgba(148,163,184,0.12)"

RANGE_ROWS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "All": None}


def _style(fig: go.Figure, height: int, unified_hover: bool = True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=34, b=8),
        font=dict(family="Inter, sans-serif", color="#B9C6DC", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified" if unified_hover else "closest",
        hoverlabel=dict(bgcolor="#0E1729", font_color="#E8EEF9", bordercolor="rgba(148,163,184,0.4)"),
    )
    fig.update_xaxes(gridcolor=GRID, showspikes=True, spikethickness=1, spikemode="across",
                     spikecolor="rgba(148,163,184,0.55)", showline=False)
    fig.update_yaxes(gridcolor=GRID, showline=False, zeroline=False)
    return fig


def _features_aligned(market_df: pd.DataFrame, feat) -> pd.DataFrame:
    md = market_df.sort_values("Date").reset_index(drop=True)
    if feat is None or len(feat) != len(md):
        from feature_engineering import build_features_for_stock  # existing backend, unchanged

        feat = build_features_for_stock(md)
        return md, feat.reset_index(drop=True)
    return md, feat.reset_index(drop=True)


def render_charts_page(run: dict, settings: dict):
    if run.get("pending") or not run.get("ok"):
        st.markdown(
            card(
                "Charts unavailable",
                '<div style="color:#C9D8F5;">Generate a prediction first — the charts use the same '
                'market data that was fed to the trained models.</div>',
                tone="blue",
            ),
            unsafe_allow_html=True,
        )
        return

    result = run["result"]
    market_df = run["market_df"]
    md, feat = _features_aligned(market_df, result.get("features_df"))

    # ---------------- controls ----------------
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 3])
    with c1:
        range_label = st.selectbox("Date range", list(RANGE_ROWS), index=2)
    with c2:
        chart_type = st.selectbox("Chart type", ["Candlestick", "Line"])
    with c3:
        sma20 = st.toggle("SMA 20", value=True)
    with c4:
        sma50 = st.toggle("SMA 50", value=True)
    with c5:
        show_bb = st.toggle("Bollinger Bands", value=False)

    # md and feat share the same positional index (0..n-1), so tails align.
    n_rows = RANGE_ROWS[range_label]
    view = (md.tail(n_rows) if n_rows else md).copy()
    vfeat = feat.iloc[-len(view):].copy()

    # ---------------- recent price history ----------------
    section_title("Recent Price History", f"Interactive chart — hover for values, drag to zoom, {len(view)} sessions shown")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.74, 0.26], subplot_titles=("<b>Price</b>", "<b>Volume</b>"),
    )

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=view["Date"], open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
                name="OHLC", increasing_line_color=GREEN, decreasing_line_color=RED,
                increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=view["Date"], y=view["Close"], name="Close", mode="lines",
                       line=dict(color=BLUE, width=2), fill="tozeroy",
                       fillcolor="rgba(59,130,246,0.08)"),
            row=1, col=1,
        )

    if sma20:
        sma20_px = view["Close"] / (1 + vfeat["sma_20_dist"]) if "sma_20_dist" in vfeat else np.nan
        fig.add_trace(go.Scatter(x=view["Date"], y=sma20_px, name="SMA 20", mode="lines",
                                 line=dict(color="#F59E0B", width=1.4)), row=1, col=1)
    if sma50:
        sma50_px = view["Close"] / (1 + vfeat["sma_50_dist"]) if "sma_50_dist" in vfeat else np.nan
        fig.add_trace(go.Scatter(x=view["Date"], y=sma50_px, name="SMA 50", mode="lines",
                                 line=dict(color="#A78BFA", width=1.4)), row=1, col=1)
    if show_bb:
        up = view["Close"] * (1 + vfeat["bb_upper_rel"])
        mid = view["Close"] * (1 + vfeat["bb_middle_rel"])
        lo = view["Close"] * (1 + vfeat["bb_lower_rel"])
        fig.add_trace(go.Scatter(x=view["Date"], y=up, name="BB upper", mode="lines",
                                 line=dict(color="rgba(126,166,255,0.55)", width=1, dash="dot"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=view["Date"], y=lo, name="BB lower", mode="lines",
                                 line=dict(color="rgba(126,166,255,0.55)", width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(59,130,246,0.07)",
                                 legendgroup="bb", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=view["Date"], y=mid, name="BB middle", mode="lines",
                                 line=dict(color="rgba(126,166,255,0.75)", width=1, dash="dash"),
                                 legendgroup="bb", showlegend=False), row=1, col=1)

    vol_colors = [GREEN if c >= o else RED for c, o in zip(view["Close"], view["Open"])]
    fig.add_trace(go.Bar(x=view["Date"], y=view["Volume"], name="Volume", marker_color=vol_colors,
                         marker_line_width=0, opacity=0.75), row=2, col=1)

    fig.update_yaxes(title_text="₹", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    _style(fig, height=620)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    # ---------------- range statistics ----------------
    if len(view) >= 2:
        period_ret = float(view["Close"].iloc[-1] / view["Close"].iloc[0] - 1)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(kpi_card("Period Return", f"{period_ret * 100:+.2f}%",
                                 sub=f"last {len(view)} sessions", tone="green" if period_ret >= 0 else "red"),
                        unsafe_allow_html=True)
        with s2:
            st.markdown(kpi_card("Period High", f"₹{view['High'].max():,.2f}"), unsafe_allow_html=True)
        with s3:
            st.markdown(kpi_card("Period Low", f"₹{view['Low'].min():,.2f}"), unsafe_allow_html=True)
        with s4:
            st.markdown(kpi_card("Avg Daily Volume", f"{view['Volume'].mean():,.0f}"), unsafe_allow_html=True)

    # ---------------- market data table ----------------
    section_title("Market Data", "Recent OHLCV rows for the selected range — real Yahoo Finance / uploaded data")
    tbl = view.tail(200)[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    prev_close = view["Close"].shift(1).tail(len(tbl))
    raw_chg = ((view["Close"] / prev_close - 1) * 100).tail(len(tbl)).values

    tbl["Date"] = tbl["Date"].dt.strftime("%d %b %Y")
    tbl["Chg %"] = [f"{x:+.2f}%" if pd.notna(x) else "--" for x in raw_chg]
    tbl = tbl[["Date", "Open", "High", "Low", "Close", "Chg %", "Volume"]].reset_index(drop=True)

    col_config = {
        "Date": st.column_config.TextColumn("Date"),
        "Open": st.column_config.NumberColumn("Open", format="₹%.2f"),
        "High": st.column_config.NumberColumn("High", format="₹%.2f"),
        "Low": st.column_config.NumberColumn("Low", format="₹%.2f"),
        "Close": st.column_config.NumberColumn("Close", format="₹%.2f"),
        "Chg %": st.column_config.TextColumn("Chg %"),
        "Volume": st.column_config.NumberColumn("Volume", format="%d"),
    }
    st.dataframe(
        tbl,
        column_config=col_config,
        hide_index=True,
        width="stretch",
        height=min(460, 36 + 35 * len(tbl)),
    )

    # ---------------- technical indicators ----------------
    section_title(
        "Technical Indicators",
        "Values computed by the project's existing feature_engineering.py — displayed as-is, not modified",
    )
    latest = vfeat.iloc[-1] if len(vfeat) else None
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        rsi = latest.get("rsi_14", np.nan) if latest is not None else np.nan
        tone = "green" if (rsi == rsi and rsi < 30) else ("red" if (rsi == rsi and rsi > 70) else "neutral")
        st.markdown(kpi_card("RSI (14)", "—" if rsi != rsi else f"{rsi:.1f}",
                             sub="oversold <30 · overbought >70", tone=tone), unsafe_allow_html=True)
    with k2:
        macd = latest.get("macd_hist_pct", np.nan) if latest is not None else np.nan
        st.markdown(kpi_card("MACD Hist", "—" if macd != macd else f"{macd:+.3f}%",
                             sub="% of price · signal above/below", tone="green" if macd == macd and macd > 0 else "red"),
                    unsafe_allow_html=True)
    with k3:
        bbpos = latest.get("bb_position", np.nan) if latest is not None else np.nan
        st.markdown(kpi_card("Bollinger Position", "—" if bbpos != bbpos else f"{bbpos * 100:.0f}%",
                             sub="0% = lower band · 100% = upper band"), unsafe_allow_html=True)
    with k4:
        sma_dist = latest.get("sma_20_dist", np.nan) if latest is not None else np.nan
        st.markdown(kpi_card("vs SMA 20", "—" if sma_dist != sma_dist else f"{sma_dist * 100:+.2f}%",
                             sub="close distance from 20-day average",
                             tone="green" if sma_dist == sma_dist and sma_dist > 0 else "red"),
                    unsafe_allow_html=True)

    ind = go.Figure()
    ind = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.5, 0.5], subplot_titles=("<b>RSI (14)</b>", "<b>MACD (% of price)</b>"))

    rsi_series = vfeat["rsi_14"] if "rsi_14" in vfeat else pd.Series(np.nan, index=vfeat.index)
    ind.add_trace(go.Scatter(x=view["Date"], y=rsi_series, name="RSI 14", mode="lines",
                             line=dict(color="#7EA6FF", width=1.8)), row=1, col=1)
    ind.add_hline(y=70, line_dash="dot", line_color=RED, opacity=0.7, row=1, col=1)
    ind.add_hline(y=30, line_dash="dot", line_color=GREEN, opacity=0.7, row=1, col=1)
    ind.add_hline(y=50, line_dash="dot", line_color="rgba(148,163,184,0.5)", opacity=0.6, row=1, col=1)
    ind.update_yaxes(range=[0, 100], row=1, col=1, title_text="RSI")

    if "macd_pct" in vfeat:
        hist = vfeat["macd_hist_pct"].fillna(0)
        ind.add_trace(go.Bar(x=view["Date"], y=hist, name="Hist", marker_color=[GREEN if v >= 0 else RED for v in hist],
                             opacity=0.55), row=2, col=1)
        ind.add_trace(go.Scatter(x=view["Date"], y=vfeat["macd_pct"], name="MACD", mode="lines",
                                 line=dict(color="#7EA6FF", width=1.6)), row=2, col=1)
        ind.add_trace(go.Scatter(x=view["Date"], y=vfeat["macd_signal_pct"], name="Signal", mode="lines",
                                 line=dict(color="#F59E0B", width=1.4, dash="dash")), row=2, col=1)
    ind.update_yaxes(title_text="%", row=2, col=1)
    _style(ind, height=520)
    st.plotly_chart(ind, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
