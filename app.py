"""ITG Analytics - market cycle dashboard (Streamlit).

Educational tool. Several metrics are heuristics/approximations inspired by
intothecryptoverse.com and are labelled as such in the UI. Not financial advice.
"""
import inspect
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# =============================================================================
# 1. CONSTANTS
# =============================================================================
ASSET_TICKERS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "S&P 500": "^GSPC",
}
# Start dates match Yahoo Finance coverage (earlier dates return nothing extra).
ASSET_START = {
    "Bitcoin (BTC)": "2014-09-17",
    "Ethereum (ETH)": "2017-11-09",
    "S&P 500": "1927-12-30",
}
CRYPTO_ASSETS = ["Bitcoin (BTC)", "Ethereum (ETH)"]
GENESIS_DATES = {  # used only by on-chain style calculations
    "Bitcoin (BTC)": pd.Timestamp("2009-01-03"),
    "Ethereum (ETH)": pd.Timestamp("2015-07-30"),
}

HALVING_BASE = 2024
HALV_MAP = {0: "Halving Year", 1: "Post-Halving Year", 2: "Bear Year", 3: "Pre-Halving Year"}
PRES_BASE = 2024
PRES_MAP = {0: "Election Year", 1: "Post-Election Year", 2: "Midterm Year", 3: "Pre-Election Year"}

CURRENT_YEAR = datetime.now().year
HALV_CURRENT = HALV_MAP[(CURRENT_YEAR - HALVING_BASE) % 4]
PRES_CURRENT = PRES_MAP[(CURRENT_YEAR - PRES_BASE) % 4]

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
QUARTER_LABELS = ["Q1", "Q2", "Q3", "Q4"]
MONTH_TICKS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
# Calendar labels for a non-leap year (Feb 29 is dropped from the data).
BASE_DATES = [(datetime(2023, 1, 1) + timedelta(days=i)).strftime("%d/%b") for i in range(365)]

# Risk metric parameters
RISK_SHORT_WINDOW = 140   # 20 weeks
RISK_LONG_WINDOW = 700    # ~100 weeks
RISK_WEIGHTS = (0.4, 0.6)  # short, long
RISK_MIN_HISTORY = 365
RISK_BUY_MAX = 0.30
RISK_AGGRESSIVE_BUY_MAX = 0.10
RISK_SELL_MIN = 0.60
RISK_SELL_FRACTION = "2/15"

# Profitability proxy parameters
PROFIT_SMA_WINDOW = 365
PROFIT_SIGMOID_K = 3.5
PROFIT_OVERHEATED = 90.0
PROFIT_CAPITULATION = 30.0

# Cycle repeat: 4 years * 365 rows (Feb 29 is removed from the data)
CYCLE_DAYS = 1460
MAX_PAST_CYCLES = 4

# Rainbow ribbon (illustrative multiples of the 20W SMA)
RAINBOW_MULTIPLIERS = [3.5, 2.4, 1.6, 1.0, 0.75, 0.55]
RAINBOW_COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"]
RAINBOW_NAMES = ["Maximum Bubble", "DCA Out", "Overheating", "20W SMA", "Accumulation", "Deep Value"]

# Social risk
X_SENTIMENT_WEIGHTS = {
    "Total Apathy / Disinterest 😴": 0.10,
    "Despair / Capitulation 📉": 0.05,
    "Healthy Discussion / Accumulation ⚪": 0.30,
    "Initial FOMO / Greed 🚀": 0.65,
    "Maximum Parabolic Euphoria 🔥": 0.95,
}

# Bitcoin issuance schedule (approximate, 144 blocks/day)
HALVING_SCHEDULE = [
    (pd.Timestamp("2009-01-03"), 50.0),
    (pd.Timestamp("2012-11-28"), 25.0),
    (pd.Timestamp("2016-07-09"), 12.5),
    (pd.Timestamp("2020-05-11"), 6.25),
    (pd.Timestamp("2024-04-20"), 3.125),
]
BLOCKS_PER_DAY = 144
MAX_SUPPLY = 21_000_000

COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

CSS = """
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155; }
    .stat-card {
        background-color: #1e293b; padding: 20px; border-radius: 12px;
        border: 1px solid #334155; margin-bottom: 15px; text-align: center;
    }
    .stat-val { font-size: 24px; font-weight: bold; margin-top: 5px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; }
</style>
"""


# =============================================================================
# 2. DATA
# =============================================================================
def process_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Enrich raw prices with calendar and cycle features."""
    df = raw.copy()
    df.columns = ["Date", "Price"]
    df["Date_Clean"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna(subset=["Price"])
    df = df[~((df["Date_Clean"].dt.month == 2) & (df["Date_Clean"].dt.day == 29))]
    df = df.sort_values("Date_Clean").reset_index(drop=True)

    df["Year"] = df["Date_Clean"].dt.year
    df["Month"] = df["Date_Clean"].dt.month
    df["Quarter"] = df["Date_Clean"].dt.quarter

    # Calendar day of year on a non-leap reference year (works for crypto and
    # for the S&P 500, which only trades on business days).
    ref = pd.to_datetime(pd.DataFrame({"year": 2023, "month": df["Month"], "day": df["Date_Clean"].dt.day}))
    df["DayOfYear"] = ref.dt.dayofyear
    df["HoverDate"] = df["DayOfYear"].map(lambda d: BASE_DATES[d - 1])

    df["HalvCycle"] = ((df["Year"] - HALVING_BASE) % 4).map(HALV_MAP)
    df["PresCycle"] = ((df["Year"] - PRES_BASE) % 4).map(PRES_MAP)

    df["YearStartPrice"] = df.groupby("Year")["Price"].transform("first")
    df["ROI"] = df["Price"] / df["YearStartPrice"]
    return df


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def _load_data_cached(asset_name: str) -> pd.DataFrame:
    raw = yf.download(ASSET_TICKERS[asset_name], start=ASSET_START[asset_name],
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("empty response")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return process_data(raw[["Close"]].reset_index())


def load_data(asset_name: str) -> pd.DataFrame:
    """Download + process prices. Failures are not cached and stop the page."""
    try:
        return _load_data_cached(asset_name)
    except Exception as exc:
        st.error(f"Could not load data for {asset_name}: {exc}")
        st.stop()


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading on-chain data...")
def fetch_coinmetrics_btc() -> pd.DataFrame:
    """Real BTC market cap and MVRV from the CoinMetrics community API."""
    params = {
        "assets": "btc", "metrics": "CapMrktCurUSD,CapMVRVCur", "frequency": "1d",
        "start_time": "2010-07-18", "page_size": 10000,
    }
    rows, url = [], COINMETRICS_URL
    while url:
        resp = requests.get(url, params=params if url == COINMETRICS_URL else None, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("data", []))
        url = payload.get("next_page_url")
    df = pd.DataFrame(rows)
    df["Date_Clean"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df["MC"] = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce")
    df["MVRV"] = pd.to_numeric(df["CapMVRVCur"], errors="coerce")
    df = df.dropna(subset=["MC", "MVRV"])
    df["RC"] = df["MC"] / df["MVRV"]
    return df[["Date_Clean", "MC", "RC", "MVRV"]].reset_index(drop=True)


# =============================================================================
# 3. CALCULATIONS (pure functions)
# =============================================================================
def period_returns(df: pd.DataFrame, key: str, n_cols: int) -> pd.DataFrame:
    """Period-over-period returns (%) by Year x Month/Quarter.

    Returns are computed on the full series first, so January/Q1 correctly
    compares against the previous December/Q4 regardless of cycle filtering.
    """
    last = df.groupby(["Year", key])["Price"].last()
    ret = last.pct_change(fill_method=None) * 100
    return ret.unstack(key).reindex(columns=range(1, n_cols + 1))


def roi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """DayOfYear (1..365) x Year matrix of YTD ROI; years that start late are dropped."""
    first_day = df.groupby("Year")["DayOfYear"].min()
    full_years = first_day[first_day <= 10].index
    df = df[df["Year"].isin(full_years)]
    piv = df.pivot_table(index="DayOfYear", columns="Year", values="ROI", aggfunc="last")
    return piv.reindex(range(1, 366)).ffill().bfill()


def _expanding_norm(s: pd.Series, min_periods: int = RISK_MIN_HISTORY) -> pd.Series:
    lo = s.expanding(min_periods=min_periods).min()
    hi = s.expanding(min_periods=min_periods).max()
    return (s - lo) / (hi - lo).replace(0, np.nan)


def compute_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic 0-1 risk score from log deviations vs two moving averages.

    Uses only expanding statistics (no look-ahead) and leaves the warm-up
    period as NaN instead of filling it with zeros.
    """
    out = df.copy()
    price = out["Price"]
    dev_short = np.log(price / price.rolling(RISK_SHORT_WINDOW).mean())
    dev_long = np.log(price / price.rolling(RISK_LONG_WINDOW).mean())
    raw = RISK_WEIGHTS[0] * _expanding_norm(dev_short) + RISK_WEIGHTS[1] * _expanding_norm(dev_long)
    out["Risk"] = _expanding_norm(raw)
    return out.dropna(subset=["Risk"])


def strategy_for_risk(risk: float) -> tuple[str, str]:
    if risk >= RISK_SELL_MIN:
        return "SELL", f"Sell {RISK_SELL_FRACTION} of position"
    if risk <= RISK_BUY_MAX:
        mult = 1.5 if risk < RISK_AGGRESSIVE_BUY_MAX else 1.0
        return "BUY", f"DCA x{mult:.1f}"
    return "HOLD", "No action"


def project_cycles(df: pd.DataFrame, cycle_days: int = CYCLE_DAYS, max_cycles: int = MAX_PAST_CYCLES):
    """Normalised paths from the same cycle position in previous cycles."""
    prices = df["Price"].to_numpy()
    n = len(prices)
    paths = []
    for k in range(1, max_cycles + 1):
        start = n - 1 - k * cycle_days
        if start < 0:
            break
        seg = prices[start:start + cycle_days + 1]
        paths.append((df["Date_Clean"].iloc[start], seg / seg[0]))
    return paths


def supply_btc_approximate(dates: pd.Series) -> pd.Series:
    """Approximate BTC circulating supply from the halving schedule (vectorised)."""
    supply = pd.Series(0.0, index=dates.index)
    for i, (start, reward) in enumerate(HALVING_SCHEDULE):
        upper = HALVING_SCHEDULE[i + 1][0] if i + 1 < len(HALVING_SCHEDULE) else None
        clipped = dates.clip(lower=start, upper=upper) if upper is not None else dates.clip(lower=start)
        supply += (clipped - start).dt.days * BLOCKS_PER_DAY * reward
    return supply.clip(upper=MAX_SUPPLY)


def mvrv_zscore(mc: pd.Series, rc: pd.Series) -> pd.Series:
    """Z-score of (MC - RC) using the expanding std of the market cap (no look-ahead)."""
    return (mc - rc) / mc.expanding(min_periods=365).std()


def profitability_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Price-vs-365D-SMA logistic proxy (NOT real supply-in-profit data)."""
    out = df.copy()
    ratio = out["Price"] / out["Price"].rolling(PROFIT_SMA_WINDOW).mean()
    out["Profit"] = 100.0 / (1.0 + np.exp(-PROFIT_SIGMOID_K * (ratio - 1.0)))
    out["Loss"] = 100.0 - out["Profit"]
    out["Profit_50D"] = out["Profit"].rolling(50).mean()
    return out.dropna(subset=["Profit_50D"])


# =============================================================================
# 4. UI HELPERS
# =============================================================================
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_chart(fig: go.Figure) -> None:
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def stat_card(label: str, value: str, color: str | None = None, size: int = 24) -> None:
    style = f"font-size:{size}px;" + (f"color:{color};" if color else "")
    st.markdown(
        f'<div class="stat-card">{label}<div class="stat-val" style="{style}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def dual_axis_layout(fig: go.Figure, left_title: str, right_title: str,
                     left_range=None, height: int = 600) -> None:
    """Metric on the left axis, log price/market cap on the right axis."""
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        yaxis=dict(title=left_title, range=left_range, side="left",
                   showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(title=right_title, type="log", overlaying="y", side="right", showgrid=False),
    )


def select_cycle(asset_name: str, col_perspective, col_phase, key: str) -> tuple[str, str]:
    """Return (cycle column, selected phase) for the asset."""
    if asset_name == "S&P 500":
        perspectives = ["US Political Cycle"]
    else:
        perspectives = ["Halving Cycle", "US Political Cycle"]
    perspective = col_perspective.selectbox("Analysis Perspective", perspectives, key=f"{key}_persp")
    if perspective == "Halving Cycle":
        column, mapping, current, label = "HalvCycle", HALV_MAP, HALV_CURRENT, "Halving Phase"
    else:
        column, mapping, current, label = "PresCycle", PRES_MAP, PRES_CURRENT, "Political Cycle Phase"
    options = list(mapping.values())
    phase = col_phase.selectbox(label, options, index=options.index(current), key=f"{key}_phase")
    return column, phase


# =============================================================================
# 5. TABS
# =============================================================================
def render_seasonality() -> None:
    st.header("📅 Seasonality Returns")
    c1, c2, c3 = st.columns(3)
    asset = c1.selectbox("Asset", list(ASSET_TICKERS), key="seas_asset")
    view = c2.selectbox("Frequency", ["Monthly Returns (%)", "Quarterly Returns (%)"])

    is_sp = asset == "S&P 500"
    cycle_col = "PresCycle" if is_sp else "HalvCycle"
    mapping = PRES_MAP if is_sp else HALV_MAP
    current = PRES_CURRENT if is_sp else HALV_CURRENT
    options = ["All Years"] + list(mapping.values())
    phase = c3.selectbox("Filter by Cycle", options, index=options.index(current))

    df = load_data(asset)
    monthly = "Monthly" in view
    key = "Month" if monthly else "Quarter"
    labels = MONTH_LABELS if monthly else QUARTER_LABELS

    ret = period_returns(df, key, len(labels))
    if phase != "All Years":
        year_cycle = df.groupby("Year")[cycle_col].first()
        ret = ret.loc[ret.index.isin(year_cycle[year_cycle == phase].index)]
    ret.columns = labels
    if ret.empty:
        st.info("No years available for this filter.")
        return

    # The month/quarter still in progress is displayed but excluded from stats.
    now = datetime.now()
    in_progress = now.month if monthly else (now.month - 1) // 3 + 1
    stats_src = ret.copy()
    if now.year in stats_src.index:
        stats_src.loc[now.year, labels[in_progress - 1]] = np.nan
    avg, med = stats_src.mean(), stats_src.median()

    years = [str(y) for y in ret.index.tolist()[::-1]]
    header_vals = ["<b>Year</b>"] + labels
    cell_vals = [years + ["<b>Average</b>", "<b>Median</b>"]]
    cell_colors = [["#1e293b"] * (len(years) + 2)]
    for col in labels:
        vals = ret[col].tolist()[::-1] + [avg[col], med[col]]
        cell_vals.append([f"{v:+.2f}%" if pd.notnull(v) else "—" for v in vals])
        colors = []
        for i, v in enumerate(vals):
            if i >= len(vals) - 2:
                colors.append("#334155")
            elif pd.isnull(v):
                colors.append("#0f172a")
            else:
                colors.append("#10b981" if v > 0 else "#ef4444")
        cell_colors.append(colors)

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_vals, fill_color="#334155", align="center", font=dict(color="white")),
        cells=dict(values=cell_vals, fill_color=cell_colors, align="center",
                   font=dict(color="white"), height=30),
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=300 + len(years) * 30,
                      paper_bgcolor="rgba(0,0,0,0)")
    show_chart(fig)
    st.caption("The current in-progress period is shown but excluded from Average/Median. "
               "ETH uses the Bitcoin halving cycle.")


def render_cycles(advanced: bool) -> None:
    key = "adv" if advanced else "cls"
    st.header("📈 Market Cycle ROI Comparison" + (" (ITC Advanced Standard)" if advanced else " (Classic Lines)"))
    c1, c2, c3 = st.columns(3)
    asset = c1.selectbox("Asset", list(ASSET_TICKERS), key=f"{key}_asset")
    cycle_col, phase = select_cycle(asset, c2, c3, key)

    df = load_data(asset)
    hist = df[(df[cycle_col] == phase) & (df["Year"] < CURRENT_YEAR)]
    piv = roi_matrix(hist) if not hist.empty else pd.DataFrame()

    fig = go.Figure()
    if advanced and not piv.empty:
        mean = piv.mean(axis=1)
        std = piv.std(axis=1).fillna(0)
        fig.add_trace(go.Scatter(x=piv.index, y=mean + std, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=piv.index, y=mean - std, mode="lines", fill="tonexty",
                                 fillcolor="rgba(148, 163, 184, 0.12)", line=dict(width=0),
                                 name="Std. Deviation Range (1σ)", hoverinfo="skip"))

    for yr in piv.columns:
        line = dict(width=1, color="rgba(148, 163, 184, 0.3)") if advanced else dict(width=1)
        fig.add_trace(go.Scatter(
            x=piv.index, y=piv[yr], name=str(yr), text=BASE_DATES,
            hovertemplate="%{text}<br>ROI: %{y:.2f}x", line=line, opacity=1 if advanced else 0.3))

    if not piv.empty:
        fig.add_trace(go.Scatter(x=piv.index, y=piv.mean(axis=1), name="Cycle Average",
                                 line=dict(color="#ffffff", dash="dash", width=2)))
    else:
        st.info("No complete historical years for this phase yet.")

    curr = df[df["Year"] == CURRENT_YEAR]
    if not curr.empty:
        fig.add_trace(go.Scatter(
            x=curr["DayOfYear"], y=curr["ROI"], name=f"{CURRENT_YEAR} (Current)", text=curr["HoverDate"],
            hovertemplate="%{text}<br>ROI: %{y:.2f}x",
            line=dict(color="#ef4444" if advanced else "#00FFA3", width=3)))

    fig.update_layout(
        template="plotly_dark", height=650, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Month", tickmode="array", tickvals=MONTH_TICKS, ticktext=MONTH_LABELS,
                   showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="ROI (Year-To-Date)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        hovermode="x unified")
    show_chart(fig)
    st.caption("Years that start after January 10 (partial first year of data) are excluded. "
               "Past cycles are not a guarantee of future behaviour.")


def render_risk() -> None:
    st.header("📊 Risk Metric & Dynamic DCA (Heuristic)")
    st.caption("Heuristic score built from deviations vs two moving averages, normalised with expanding "
               "min/max. It is inspired by, but is not, the Into The Cryptoverse risk metric.")
    asset = st.selectbox("Select Asset for Risk Analysis", CRYPTO_ASSETS)
    df = compute_risk(load_data(asset))
    if df.empty:
        st.warning("Not enough history to compute the risk metric.")
        return

    cur = df.iloc[-1]
    mode, action = strategy_for_risk(cur["Risk"])
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Current Price", f"${cur['Price']:,.2f}")
    with c2: stat_card("Risk Score", f"{cur['Risk']:.4f}")
    with c3: stat_card("Signal", mode)
    with c4: stat_card("Suggested Action", action, size=18)

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=RISK_BUY_MAX, fillcolor="rgba(34,197,94,0.08)", line_width=0)
    fig.add_hrect(y0=RISK_SELL_MIN, y1=1, fillcolor="rgba(239,68,68,0.08)", line_width=0)
    fig.add_trace(go.Scatter(x=df["Date_Clean"], y=df["Price"], name="Price (USD)",
                             line=dict(color="rgba(255,255,255,0.25)"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["Date_Clean"], y=df["Risk"], name="Risk Score",
                             line=dict(color="#3b82f6"), yaxis="y1"))
    dual_axis_layout(fig, "Risk (0-1)", "Price (USD, log)", left_range=[0, 1], height=500)
    show_chart(fig)


def render_cycle_repeat() -> None:
    st.header("🔄 Cycle Repeat & Trajectory Scenarios")
    st.warning("Scenario tool, not a forecast: it rescales what happened from the same cycle position in "
               "previous cycles onto today's price. Real outcomes will differ.")
    asset = st.selectbox("Select Asset for Projection", CRYPTO_ASSETS)
    df = load_data(asset)

    paths = project_cycles(df)
    if not paths:
        st.info(f"Not enough history for a full {CYCLE_DAYS}-day cycle on {asset}.")
        return

    last_price, last_date = df["Price"].iloc[-1], df["Date_Clean"].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(CYCLE_DAYS + 1)]
    scaled = [last_price * p for _, p in paths]

    fig = go.Figure()
    recent = df[df["Date_Clean"] >= last_date - timedelta(days=700)]
    fig.add_trace(go.Scatter(x=recent["Date_Clean"], y=recent["Price"], name="Historical Price",
                             line=dict(color="#f8fafc")))
    for (start, _), path in zip(paths, scaled):
        fig.add_trace(go.Scatter(x=future_dates, y=path, name=f"Cycle from {start:%b %Y}",
                                 line=dict(width=1), opacity=0.45))
    fig.add_trace(go.Scatter(x=future_dates, y=np.median(np.vstack(scaled), axis=0),
                             name="Median scenario", line=dict(color="#38bdf8", dash="dash", width=3)))
    fig.update_layout(template="plotly_dark", height=600, yaxis_type="log",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    show_chart(fig)
    st.caption(f"{len(paths)} past cycle(s) of {CYCLE_DAYS} days used. With so few cycles, "
               "treat the spread between lines as the real message.")


def render_rainbow() -> None:
    st.header("🌈 Rainbow Ribbon (20W SMA multiples)")
    st.caption("Illustrative bands at fixed multiples of the 20-week SMA (140 daily candles). "
               "Multiples are not the official Into The Cryptoverse values.")
    asset = st.selectbox("Select Asset for Rainbow", CRYPTO_ASSETS)
    df = load_data(asset)
    df["SMA_20W"] = df["Price"].rolling(140).mean()
    view = df[df["Date_Clean"] >= df["Date_Clean"].iloc[-1] - timedelta(days=1200)]

    fig = go.Figure()
    for m, c, n in zip(RAINBOW_MULTIPLIERS, RAINBOW_COLORS, RAINBOW_NAMES):
        fig.add_trace(go.Scatter(x=view["Date_Clean"], y=view["SMA_20W"] * m, name=f"{n} ({m}x)",
                                 line=dict(color=c, width=1.5)))
    fig.add_trace(go.Scatter(x=view["Date_Clean"], y=view["Price"], name="Price",
                             line=dict(color="#ffffff", width=2.5)))
    fig.update_layout(template="plotly_dark", height=650, yaxis_type="log",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    show_chart(fig)


def render_social_risk() -> None:
    st.header("📢 Social Sentiment Risk (Manual Inputs)")
    st.caption("Not real-time: enter the values you observe yourself. The score is a weighted heuristic.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📺 YouTube Channels Metrics")
        views = st.slider("Weekly Average Views (Retail/Mass Channels)", 10_000, 500_000, 85_000, step=5_000)
        subs = st.slider("Weekly New Subscribers (Macro/Cycle Channels)", 0, 50_000, 2_500, step=500)
    with col2:
        st.subheader("📱 Apps & Social Networks Metrics")
        rank = st.number_input("Coinbase App Position on US App Store", min_value=1, max_value=500,
                               value=310, step=10)
        x_mood = st.selectbox("Dominant Sentiment on X", list(X_SENTIMENT_WEIGHTS))

    v_risk = (views - 10_000) / (500_000 - 10_000)
    s_risk = subs / 50_000
    c_risk = (500 - rank) / (500 - 1)
    x_risk = X_SENTIMENT_WEIGHTS[x_mood]
    score = float(np.clip(v_risk * 0.3 + s_risk * 0.2 + c_risk * 0.3 + x_risk * 0.2, 0.0, 1.0))

    if score < 0.25:
        diagnostic, color = "Low social risk", "#22c55e"
    elif score < 0.60:
        diagnostic, color = "Moderate", "#eab308"
    else:
        diagnostic, color = "Elevated social risk", "#ef4444"

    a, b = st.columns(2)
    with a: stat_card("Social Risk Score", f"{score:.4f}")
    with b: stat_card("Diagnostic", diagnostic, color=color)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": "Social Risk Dial"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#38bdf8"}, "bgcolor": "#1e293b",
               "steps": [{"range": [0, 0.25], "color": "rgba(5, 150, 105, 0.2)"},
                         {"range": [0.25, 0.6], "color": "rgba(148, 163, 184, 0.12)"},
                         {"range": [0.6, 1.0], "color": "rgba(239, 68, 68, 0.2)"}]}))
    gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300, font={"color": "white"})
    show_chart(gauge)


def render_profitability() -> None:
    st.header("📊 Profitability Proxy (not real on-chain data)")
    st.warning("Real Supply in Profit/Loss needs UTXO-level realised prices (Glassnode, CryptoQuant). "
               f"This is a logistic transform of price vs its {PROFIT_SMA_WINDOW}-day SMA. "
               "It is a proxy and gives no timing probabilities.")
    df = profitability_proxy(load_data("Bitcoin (BTC)"))
    if df.empty:
        st.warning("Not enough history.")
        return

    cur = df.iloc[-1]
    rising = df["Profit_50D"].diff().iloc[-1] > 0
    if cur["Profit_50D"] >= PROFIT_OVERHEATED:
        signal, color = "Overheated zone (proxy)", "#f97316"
    elif cur["Profit"] <= PROFIT_CAPITULATION:
        signal, color = "Capitulation zone (proxy)", "#38bdf8"
    elif rising:
        signal, color = "Improving profitability", "#22c55e"
    else:
        signal, color = "Neutral / weakening", "#e2e8f0"

    a, b, c = st.columns(3)
    with a:
        stat_card("Proxy Profit / Loss",
                  f'<span style="color:#22c55e;">{cur["Profit"]:.1f}%</span> / '
                  f'<span style="color:#ef4444;">{cur["Loss"]:.1f}%</span>')
    with b: stat_card("50D Avg of Profit Proxy", f"{cur['Profit_50D']:.1f}%")
    with c: stat_card("Heuristic Signal", signal, color=color, size=18)

    view = df[df["Date_Clean"] >= "2015-01-01"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["Date_Clean"], y=view["Price"], name="BTC Price (USD)",
                             line=dict(color="rgba(255,255,255,0.22)", width=1.5), yaxis="y2"))
    fig.add_trace(go.Scatter(x=view["Date_Clean"], y=view["Profit"], name="Profit proxy (%)",
                             line=dict(color="#22c55e", width=2), yaxis="y1"))
    fig.add_trace(go.Scatter(x=view["Date_Clean"], y=view["Loss"], name="Loss proxy (%)",
                             line=dict(color="#ef4444", width=2), yaxis="y1"))
    fig.add_trace(go.Scatter(x=view["Date_Clean"], y=view["Profit_50D"], name="50D avg of profit proxy",
                             line=dict(color="#fb923c", width=1.2, dash="dash"), yaxis="y1"))
    fig.add_hline(y=50.0, line_dash="dot", line_color="rgba(255,255,255,0.3)", annotation_text="50/50")
    fig.add_hline(y=PROFIT_OVERHEATED, line_color="#ef4444", line_width=1,
                  annotation_text=f"{PROFIT_OVERHEATED:.0f}% overheated band")
    fig.add_hline(y=PROFIT_CAPITULATION, line_color="#22c55e", line_width=1,
                  annotation_text=f"{PROFIT_CAPITULATION:.0f}% capitulation band")
    dual_axis_layout(fig, "Percentage (%)", "BTC Price (USD, log)", left_range=[0, 105], height=650)
    show_chart(fig)


def render_mvrv() -> None:
    st.header("📈 Bitcoin MVRV Z-Score")
    source = st.radio("Data source", ["On-chain (CoinMetrics)", "Approximation (Yahoo prices)"],
                      horizontal=True)

    data = None
    if source.startswith("On-chain"):
        try:
            data = fetch_coinmetrics_btc()
            st.caption("Realised cap derived from CoinMetrics community MVRV. "
                       "Z = (MC − RC) / expanding std(MC).")
        except Exception as exc:
            st.warning(f"CoinMetrics unavailable ({exc}). Falling back to the approximation.")
    if data is None:
        df = load_data("Bitcoin (BTC)")
        data = pd.DataFrame({"Date_Clean": df["Date_Clean"]})
        data["MC"] = df["Price"] * supply_btc_approximate(df["Date_Clean"])
        data["RC"] = df["Price"].rolling(365).mean() * supply_btc_approximate(df["Date_Clean"])
        st.caption("⚠️ Approximation: realised cap ≈ 365D SMA × estimated supply. "
                   "Not real on-chain data; levels will differ from the official metric.")

    data["Z"] = mvrv_zscore(data["MC"], data["RC"])
    data = data.dropna(subset=["Z"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date_Clean"], y=data["MC"], name="Market Cap",
                             line=dict(color="rgba(255,255,255,0.5)"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=data["Date_Clean"], y=data["Z"], name="Z-Score",
                             line=dict(color="#f39c12"), yaxis="y1"))
    fig.add_hline(y=7, line_color="#ef4444", line_width=1, line_dash="dot", annotation_text="7 (classic overheated)")
    fig.add_hline(y=0, line_color="#22c55e", line_width=1, line_dash="dot", annotation_text="0 (classic undervalued)")
    dual_axis_layout(fig, "Z-Score", "Market Cap (USD, log)")
    show_chart(fig)


def render_moving_averages() -> None:
    st.header("📉 Weekly Moving Averages")
    asset = st.selectbox("Asset Selection", list(ASSET_TICKERS))
    df = load_data(asset)
    weekly = df.set_index("Date_Clean")["Price"].resample("W").last().to_frame()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["Price"], name="Price", line=dict(color="white")))
    for period, color in zip([20, 50, 100, 200], ["#3b82f6", "#f59e0b", "#ec4899", "#a855f7"]):
        sma = weekly["Price"].rolling(window=period).mean()
        fig.add_trace(go.Scatter(x=weekly.index, y=sma, name=f"{period}W SMA",
                                 line=dict(color=color), opacity=0.7))
    fig.update_layout(template="plotly_dark", height=600, yaxis_type="log",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    show_chart(fig)


# =============================================================================
# 6. APP
# =============================================================================
TABS = {
    "Seasonality (Heatmap)": render_seasonality,
    "Market Cycles": lambda: render_cycles(advanced=False),
    "Cycles (ITC Advanced)": lambda: render_cycles(advanced=True),
    "Risk Metric (DCA)": render_risk,
    "Cycle Repeat (Scenarios)": render_cycle_repeat,
    "Rainbow Ribbon": render_rainbow,
    "Social Risk (Manual)": render_social_risk,
    "Profitability Proxy": render_profitability,
    "MVRV Z-Score": render_mvrv,
    "Moving Averages": render_moving_averages,
}


def main() -> None:
    st.set_page_config(page_title="ITG Analytics", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🚀 ITG Analytics")
        st.markdown("---")
        st.info(f"📅 Current Year: **{CURRENT_YEAR}**\n\n"
                f"🇺🇸 Cycle: **{PRES_CURRENT}**\n\n"
                f"₿ Cycle: **{HALV_CURRENT}**")
        choice = st.radio("Select Analysis:", list(TABS))
        st.markdown("---")
        st.caption("Educational tool, not financial advice. Several metrics are heuristics or "
                   "approximations and are labelled as such.")

    TABS[choice]()


if __name__ == "__main__":
    main()
