import numpy as np
import pandas as pd

try:
    import talib as _talib
except Exception:
    _talib = None

FEATURE_ENGINEERING_VERSION = "1.0"

def _sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def _ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def _wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

def _days_since_extreme(a, w, use_max):
    # number of bars since the max/min inside the trailing window of length w (past data only)
    out = np.full(len(a), np.nan)
    if len(a) >= w:
        win = np.lib.stride_tricks.sliding_window_view(a, w)[:, ::-1]
        out[w - 1:] = win.argmax(axis=1) if use_max else win.argmin(axis=1)
    return out

def build_features_for_stock(d):
    # d: OHLCV rows of ONE stock, sorted by Date ascending. Returns a float32 feature frame with the same index.
    assert d["Date"].is_monotonic_increasing, "rows must be sorted by Date inside each stock"
    o = d["Open"].astype(float); h = d["High"].astype(float); l = d["Low"].astype(float)
    c = d["Close"].astype(float); v = d["Volume"].astype(float)
    idx = d.index
    f = {}
    prev_c = c.shift(1)
    ret1 = c.pct_change(1)
    delta = c.diff()

    # ---- price / return features ----
    for k in (1, 2, 3, 5, 10, 20):
        f["ret_%dd" % k] = c.pct_change(k)
    f["hl_pct"] = (h - l) / c
    f["oc_pct"] = (c - o) / o
    f["gap_pct"] = o / prev_c - 1.0
    for k in (5, 10, 20):
        f["roll_vol_%d" % k] = ret1.rolling(k, min_periods=k).std()

    # ---- lag features (relative) ----
    for k in (1, 2, 3, 5, 10, 20):
        f["close_lag_%d_rel" % k] = c.shift(k) / c - 1.0
        f["ret_lag_%d" % k] = ret1.shift(k)

    # ---- moving averages: distance of price + slope ----
    for n in (5, 10, 20, 50, 100):
        s_ = _sma(c, n)
        e_ = _ema(c, n)
        f["sma_%d_dist" % n] = c / s_ - 1.0
        f["ema_%d_dist" % n] = c / e_ - 1.0
        if n in (10, 20, 50):
            f["sma_%d_slope" % n] = s_ / s_.shift(5) - 1.0
            f["ema_%d_slope" % n] = e_ / e_.shift(5) - 1.0

    # ---- momentum indicators ----
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rs = _wilder(gain, 14) / _wilder(loss, 14)
    f["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
    f["roc_12"] = (c / c.shift(12) - 1.0) * 100.0

    tp = (h + l + c) / 3.0
    tp_arr = tp.values
    mad = np.full(len(tp_arr), np.nan)
    if len(tp_arr) >= 20:
        win = np.lib.stride_tricks.sliding_window_view(tp_arr, 20)
        mad[19:] = np.abs(win - win.mean(axis=1, keepdims=True)).mean(axis=1)
    f["cci_20"] = (tp - _sma(tp, 20)) / (0.015 * pd.Series(mad, index=idx))

    hh14 = h.rolling(14, min_periods=14).max()
    ll14 = l.rolling(14, min_periods=14).min()
    rng14 = hh14 - ll14
    f["willr_14"] = -100.0 * (hh14 - c) / rng14
    stoch_k = 100.0 * (c - ll14) / rng14
    f["stoch_k_14"] = stoch_k
    f["stoch_d_3"] = _sma(stoch_k, 3)

    rmf = tp * v
    pos_mf = rmf.where(tp > tp.shift(1), 0.0)
    neg_mf = rmf.where(tp < tp.shift(1), 0.0)
    mfr = pos_mf.rolling(14, min_periods=14).sum() / neg_mf.rolling(14, min_periods=14).sum()
    f["mfi_14"] = 100.0 - 100.0 / (1.0 + mfr)

    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=idx)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=idx)
    atr14 = _wilder(tr, 14)
    pdi = 100.0 * _wilder(plus_dm, 14) / atr14
    mdi = 100.0 * _wilder(minus_dm, 14) / atr14
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi)
    f["adx_14"] = _wilder(dx, 14)
    f["plus_di_14"] = pdi
    f["minus_di_14"] = mdi
    f["dx_14"] = dx
    f["plus_dm_14"] = _wilder(plus_dm, 14) / c
    f["minus_dm_14"] = _wilder(minus_dm, 14) / c

    min_lc = pd.concat([l, prev_c], axis=1).min(axis=1)
    max_hc = pd.concat([h, prev_c], axis=1).max(axis=1)
    bp = c - min_lc
    trr = max_hc - min_lc
    def _avg(n):
        return bp.rolling(n, min_periods=n).sum() / trr.rolling(n, min_periods=n).sum()
    f["ultosc"] = 100.0 * (4.0 * _avg(7) + 2.0 * _avg(14) + _avg(28)) / 7.0

    t3 = _ema(_ema(_ema(c, 15), 15), 15)
    f["trix_15"] = t3.pct_change(1) * 100.0

    n_ar = 25
    ds_h = _days_since_extreme(h.values, n_ar + 1, True)
    ds_l = _days_since_extreme(l.values, n_ar + 1, False)
    aroon_up = pd.Series(100.0 * (n_ar - ds_h) / n_ar, index=idx)
    aroon_dn = pd.Series(100.0 * (n_ar - ds_l) / n_ar, index=idx)
    f["aroon_up_25"] = aroon_up
    f["aroon_down_25"] = aroon_dn
    f["aroon_osc_25"] = aroon_up - aroon_dn

    f["bop"] = (c - o) / (h - l)
    su = gain.rolling(14, min_periods=14).sum()
    sd = loss.rolling(14, min_periods=14).sum()
    f["cmo_14"] = 100.0 * (su - sd) / (su + sd)
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    f["ppo"] = 100.0 * (ema12 - ema26) / ema26
    f["momentum_10_atr"] = (c - c.shift(10)) / atr14

    # ---- MACD family (normalised by price) ----
    macd = ema12 - ema26
    macd_sig = _ema(macd, 9)
    macd_hist = macd - macd_sig
    f["macd_pct"] = macd / c * 100.0
    f["macd_signal_pct"] = macd_sig / c * 100.0
    f["macd_hist_pct"] = macd_hist / c * 100.0
    f["macd_hist_chg"] = macd_hist.diff(1) / c * 100.0

    # ---- volatility ----
    f["natr_14"] = atr14 / c * 100.0
    f["tr_pct"] = tr / c * 100.0
    f["std_20_pct"] = c.rolling(20, min_periods=20).std() / c
    bb_mid = _sma(c, 20)
    bb_sd = c.rolling(20, min_periods=20).std(ddof=0)
    bb_up = bb_mid + 2.0 * bb_sd
    bb_lo = bb_mid - 2.0 * bb_sd
    f["bb_upper_rel"] = bb_up / c - 1.0
    f["bb_lower_rel"] = bb_lo / c - 1.0
    f["bb_middle_rel"] = bb_mid / c - 1.0
    f["bb_width"] = (bb_up - bb_lo) / bb_mid
    f["bb_position"] = (c - bb_lo) / (bb_up - bb_lo)

    # ---- volume indicators ----
    obv = (np.sign(delta).fillna(0.0) * v).cumsum()
    vsum10 = v.rolling(10, min_periods=10).sum()
    f["obv_roc_10"] = (obv - obv.shift(10)) / vsum10
    clv = (((c - l) - (h - c)) / (h - l)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ad = (clv * v).cumsum()
    f["ad_slope_10"] = (ad - ad.shift(10)) / vsum10
    f["adosc_norm"] = (_ema(ad, 3) - _ema(ad, 10)) / v.rolling(10, min_periods=10).mean()
    f["volume_change"] = v.pct_change(1).clip(-1.0, 10.0)
    f["rel_volume_5"] = v / _sma(v, 5)
    f["rel_volume_20"] = v / _sma(v, 20)

    # ---- price-transform features (relative to Close) ----
    f["avg_price_rel"] = (o + h + l + c) / 4.0 / c - 1.0
    f["median_price_rel"] = (h + l) / 2.0 / c - 1.0
    f["typical_price_rel"] = tp / c - 1.0
    f["weighted_close_rel"] = (h + l + 2.0 * c) / 4.0 / c - 1.0

    # ---- optional Hilbert-Transform cycle features (only if TA-Lib is installed) ----
    if _talib is not None:
        try:
            cv = c.values.astype(float)
            ht = {}
            ht["ht_trendmode"] = _talib.HT_TRENDMODE(cv).astype(float)
            ht["ht_dcperiod"] = _talib.HT_DCPERIOD(cv)
            ht["ht_dcphase"] = _talib.HT_DCPHASE(cv)
            inph, quad = _talib.HT_PHASOR(cv)
            ht["ht_phasor_inphase"] = inph / cv
            ht["ht_phasor_quadrature"] = quad / cv
            sine, lead = _talib.HT_SINE(cv)
            ht["ht_sine"] = sine
            ht["ht_leadsine"] = lead
            for k_, v_ in ht.items():
                f[k_] = pd.Series(v_, index=idx)
        except Exception:
            pass

    out = pd.DataFrame(f, index=idx)
    out = out.replace([np.inf, -np.inf], np.nan).ffill()
    return out.astype("float32")

def compute_features(df, symbol_col="Symbol"):
    # df must be sorted by [symbol_col, Date] with a unique index
    parts = []
    for _, g in df.groupby(symbol_col, sort=False):
        parts.append(build_features_for_stock(g))
    return pd.concat(parts).reindex(df.index)
