import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION AND CONSTANTS ---
st.set_page_config(page_title="ITG Analytics", layout="wide")

current_year = datetime.now().year

# Tickers and start dates per asset
ASSET_TICKERS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "S&P 500": "^GSPC",
}
ASSET_START = {
    "Bitcoin (BTC)": "2010-07-17",
    "Ethereum (ETH)": "2015-08-07",
    "S&P 500": "1927-12-30",
}

# Halving Cycles
HALVING_BASE = 2024
HALV_MAP = {
    0: "Halving Year",
    1: "Post-Halving Year",
    2: "Bear Year",
    3: "Pre-Halving Year",
}

# Presidential Cycles
PRES_BASE = 2024
PRES_MAP = {
    0: "Election Year",
    1: "Post-Election Year",
    2: "Midterm Year",
    3: "Pre-Election Year",
}

halv_cycle_current = HALV_MAP.get((current_year - HALVING_BASE) % 4, "—")
pres_cycle_current = PRES_MAP.get((current_year - PRES_BASE) % 4, "—")

# Modern Dark Mode CSS Styling
st.markdown("""
    <style>
        .stApp { background-color: #0f172a; color: #f8fafc; }
        [data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155; }
        .stat-card {
            background-color: #1e293b;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #334155;
            margin-bottom: 15px;
            text-align: center;
        }
        .stat-val { font-size: 24px; font-weight: bold; margin-top: 5px; }
        h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)


# --- 2. DATA CACHE ---
@st.cache_data(ttl=3600)
def fetch_raw_prices(ticker: str, start_date: str) -> pd.DataFrame:
    """Download raw price data from yfinance."""
    try:
        raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw[["Close"]].reset_index()
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data
def process_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Data processing and feature enrichment."""
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = ["Date", "Price"]
    df["Date_Clean"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df[~((df["Date_Clean"].dt.month == 2) & (df["Date_Clean"].dt.day == 29))]

    df["Year"] = df["Date_Clean"].dt.year
    df["Month"] = df["Date_Clean"].dt.month
    df["Quarter"] = df["Date_Clean"].dt.quarter
    df["DayOfYear"] = df.groupby("Year").cumcount() + 1

    df["HalvCycle"] = df["Year"].apply(lambda y: HALV_MAP.get((y - HALVING_BASE) % 4, "—"))
    df["PresCycle"] = df["Year"].apply(lambda y: PRES_MAP.get((y - PRES_BASE) % 4, "—"))

    df["YearStartPrice"] = df.groupby("Year")["Price"].transform("first")
    df["ROI"] = df["Price"] / df["YearStartPrice"]

    base_dates = [(datetime(2023, 1, 1) + timedelta(days=i)).strftime("%d/%b") for i in range(366)]
    df["HoverDate"] = df["DayOfYear"].apply(lambda x: base_dates[min(int(x) - 1, 365)])
    return df


def load_data(asset_name: str) -> pd.DataFrame:
    ticker = ASSET_TICKERS[asset_name]
    start = ASSET_START[asset_name]
    raw = fetch_raw_prices(ticker, start)
    if raw.empty:
        st.warning(f"No data available for {asset_name}.")
        return pd.DataFrame()
    return process_data(raw)


def supply_btc_approximate(date: datetime) -> float:
    halvings = [
        (datetime(2009, 1, 3), 50),
        (datetime(2012, 11, 28), 25),
        (datetime(2016, 7, 9), 12.5),
        (datetime(2020, 5, 11), 6.25),
        (datetime(2024, 4, 20), 3.125),
    ]
    supply = 0.0
    prev_date, prev_reward = halvings[0]
    for i, (h_date, reward) in enumerate(halvings[1:], start=1):
        end = h_date if date >= h_date else date
        days = (end - prev_date).days
        blocks = days * 144
        supply += blocks * prev_reward
        if date < h_date:
            break
        prev_date, prev_reward = h_date, reward
    else:
        days = (date - prev_date).days
        supply += days * 144 * prev_reward
    return min(supply, 21_000_000)


# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🚀 ITG Analytics")
    st.markdown("---")
    st.info(
        f"📅 Current Year: **{current_year}**\n\n"
        f"🇺🇸 Cycle: **{pres_cycle_current}**\n\n"
        f"₿ Cycle: **{halv_cycle_current}**"
    )
    aba = st.radio(
        "Select Analysis:",
        [
            "Seasonality (Heatmap)", 
            "Market Cycles", 
            "Cycles (ITC Advanced)", 
            "Risk Metric (DCA)", 
            "Cycle Repeat (Bitbo)", 
            "Rainbow Ribbon (Smart)", 
            "Social Risk (Sentiment)", 
            "Supply in Profit/Loss", 
            "MVRV Z-Score", 
            "Moving Averages"
        ],
    )


# --- ABA 1: SEASONALITY ---
if aba == "Seasonality (Heatmap)":
    st.header("📅 Seasonality Returns")
    c1, c2, c3 = st.columns(3)
    asset_name = c1.selectbox("Asset", list(ASSET_TICKERS.keys()))
    view_mode = c2.selectbox("Frequency", ["Monthly Returns (%)", "Quarterly Returns (%)"])

    is_sp500 = asset_name == "S&P 500"
    cycle_col = "PresCycle" if is_sp500 else "HalvCycle"
    
    opcoes_ciclo = ["All Years"] + (list(PRES_MAP.values()) if is_sp500 else list(HALV_MAP.values()))
    target_string = pres_cycle_current if is_sp500 else halv_cycle_current
    idx_default = opcoes_ciclo.index(target_string) if target_string in opcoes_ciclo else 0
    
    filter_type = c3.selectbox("Filter by Cycle", opcoes_ciclo, index=idx_default)

    df_main = load_data(asset_name)
    if df_main.empty: st.stop()

    df_h = df_main[df_main[cycle_col] == filter_type] if filter_type != "All Years" else df_main

    is_monthly = "Monthly" in view_mode
    group_key = "Month" if is_monthly else "Quarter"
    cols_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if is_monthly else ["Q1", "Q2", "Q3", "Q4"]

    pivot_df = df_h.groupby(["Year", group_key])["Price"].last().unstack()
    all_prices = df_main.groupby(["Year", group_key])["Price"].last().unstack()
    returns_df = pivot_df.pct_change(axis=1) * 100

    last_col = 12 if is_monthly else 4
    for yr in returns_df.index:
        if yr - 1 in all_prices.index:
            try:
                returns_df.at[yr, 1] = ((all_prices.at[yr, 1] / all_prices.at[yr - 1, last_col]) - 1) * 100
            except KeyError:
                pass

    returns_df.columns = cols_names
    avg = returns_df.mean()
    med = returns_df.median()
    years = [str(y) for y in returns_df.index.tolist()[::-1]]

    header_vals = ["<b>Year</b>"] + list(returns_df.columns)
    cell_vals = [years + ["<b>Average</b>", "<b>Median</b>"]]
    cell_colors = [["#1e293b"] * (len(years) + 2)]

    for col in returns_df.columns:
        vals = returns_df[col].tolist()[::-1] + [avg[col], med[col]]
        cell_vals.append([f"{v:+.2f}%" if pd.notnull(v) else "—" for v in vals])
        colors = []
        for i, v in enumerate(vals):
            if i >= len(vals) - 2: colors.append("#334155")
            elif pd.isnull(v): colors.append("#0f172a")
            elif v > 0: colors.append("#10b981")
            else: colors.append("#ef4444")
        cell_colors.append(colors)

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_vals, fill_color="#334155", align="center", font=dict(color="white")),
        cells=dict(values=cell_vals, fill_color=cell_colors, align="center", font=dict(color="white"), height=30)
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=300 + (len(years) * 30), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 2: MARKET CYCLES ---
elif aba == "Market Cycles":
    st.header("📈 Market Cycle ROI Comparison (Classic Lines)")
    c1, c2, c3 = st.columns(3)
    asset_name = c1.selectbox("Asset", list(ASSET_TICKERS.keys()))

    is_sp500 = asset_name == "S&P 500"
    col_c = "PresCycle" if is_sp500 else "HalvCycle"
    
    if is_sp500:
        lista_opcoes = list(PRES_MAP.values())
        idx_pres = lista_opcoes.index(pres_cycle_current) if pres_cycle_current in lista_opcoes else 0
        ciclo = c2.selectbox("Political Cycle Phase", lista_opcoes, index=idx_pres)
    else:
        ciclo_tipo = c2.selectbox("Analysis Perspective", ["Halving Cycle", "US Political Cycle"], index=0)
        if ciclo_tipo == "Halving Cycle":
            lista_opcoes = list(HALV_MAP.values())
            idx_halv = lista_opcoes.index(halv_cycle_current) if halv_cycle_current in lista_opcoes else 0
            ciclo = c3.selectbox("Halving Phase", list(HALV_MAP.values()), index=idx_halv)
            col_c = "HalvCycle"
        else:
            lista_opcoes = list(PRES_MAP.values())
            idx_pres = lista_opcoes.index(pres_cycle_current) if pres_cycle_current in lista_opcoes else 0
            ciclo = c3.selectbox("Political Cycle Phase", lista_opcoes, index=idx_pres)
            col_c = "PresCycle"

    df_cycle = load_data(asset_name)
    if df_cycle.empty: st.stop()

    fig = go.Figure()
    df_hist = df_cycle[(df_cycle[col_c] == ciclo) & (df_cycle["Year"] < current_year)]
    for yr in sorted(df_hist["Year"].unique()):
        df_yr = df_hist[df_hist["Year"] == yr]
        fig.add_trace(go.Scatter(x=df_yr["DayOfYear"], y=df_yr["ROI"], name=str(yr), text=df_yr["HoverDate"], hovertemplate="%{text}<br>ROI: %{y:.2f}x", line=dict(width=1), opacity=0.3))

    stats = df_hist.groupby("DayOfYear")["ROI"].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats["DayOfYear"], y=stats["ROI"], name="Historical Average", line=dict(color="white", dash="dash", width=2)))

    df_curr = df_cycle[df_cycle["Year"] == current_year]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr["DayOfYear"], y=df_curr["ROI"], name=str(current_year), text=df_curr["HoverDate"], hovertemplate="%{text}<br>ROI: %{y:.2f}x", line=dict(color="#00FFA3", width=3)))

    fig.update_layout(template="plotly_dark", height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Day of Year", yaxis_title="YTD ROI")
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 3: CYCLES (ITC ADVANCED) ---
elif aba == "Cycles (ITC Advanced)":
    st.header("📈 Market Cycle ROI Comparison (ITC Advanced Standard)")
    c1, c2, c3 = st.columns(3)
    asset_name = c1.selectbox("Asset (Advanced)", list(ASSET_TICKERS.keys()))

    is_sp500 = asset_name == "S&P 500"
    col_c = "PresCycle" if is_sp500 else "HalvCycle"
    
    if is_sp500:
        lista_opcoes = list(PRES_MAP.values())
        idx_pres = lista_opcoes.index(pres_cycle_current) if pres_cycle_current in lista_opcoes else 0
        ciclo = c2.selectbox("Political Cycle Phase (Advanced)", lista_opcoes, index=idx_pres)
    else:
        ciclo_tipo = c2.selectbox("Analysis Perspective (Advanced)", ["Halving Cycle", "US Political Cycle"], index=0)
        if ciclo_tipo == "Halving Cycle":
            lista_opcoes = list(HALV_MAP.values())
            idx_halv = lista_opcoes.index(halv_cycle_current) if halv_cycle_current in lista_opcoes else 0
            ciclo = c3.selectbox("Halving Phase (Advanced)", lista_opcoes, index=idx_halv)
            col_c = "HalvCycle"
        else:
            lista_opcoes = list(PRES_MAP.values())
            idx_pres = lista_opcoes.index(pres_cycle_current) if pres_cycle_current in lista_opcoes else 0
            ciclo = c3.selectbox("Political Cycle Phase (Advanced)", lista_opcoes, index=idx_pres)
            col_c = "PresCycle"

    df_cycle = load_data(asset_name)
    if df_cycle.empty: st.stop()

    fig = go.Figure()
    df_hist = df_cycle[(df_cycle[col_c] == ciclo) & (df_cycle["Year"] < current_year)]
    
    if not df_hist.empty:
        stats_bounds = df_hist.groupby
