import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURAÇÃO E CONSTANTES ---
st.set_page_config(page_title="ITG Analytics", layout="wide")

ano_atual = datetime.now().year

# Tickers e datas de início por ativo
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

# Ciclos de Halving — ancorado no halving real de 2024
HALVING_BASE = 2024
HALV_MAP = {
    0: "Halving Year",
    1: "Post-Halving Year",
    2: "Bear Year",
    3: "Pre-Halving Year",
}

# Ciclos Presidenciais — ancorado em 2024 (Election Year)
PRES_BASE = 2024
PRES_MAP = {
    0: "Election Year",
    1: "Post-Election Year",
    2: "Midterm Year",
    3: "Pre-Election Year",
}

halv_cycle_atual = HALV_MAP.get((ano_atual - HALVING_BASE) % 4, "—")
pres_cycle_atual = PRES_MAP.get((ano_atual - PRES_BASE) % 4, "—")

# Estilo CSS para Dark Mode Moderno
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


# --- 2. CACHE DE DADOS ---
@st.cache_data(ttl=3600)
def fetch_raw_prices(ticker: str, start_date: str) -> pd.DataFrame:
    """Download dos preços brutos do yfinance."""
    try:
        raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw[["Close"]].reset_index()
    except Exception as e:
        st.error(f"Erro ao descarregar dados para {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data
def process_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Processamento e enriquecimento dos dados."""
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
    """Carrega e processa dados para o ativo selecionado."""
    ticker = ASSET_TICKERS[asset_name]
    start = ASSET_START[asset_name]
    raw = fetch_raw_prices(ticker, start)
    if raw.empty:
        st.warning(f"Sem dados disponíveis para {asset_name}.")
        return pd.DataFrame()
    return process_data(raw)


def supply_btc_aproximado(date: datetime) -> float:
    """Estima o supply circulante do BTC com base na data."""
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
        f"📅 Ano Atual: **{ano_atual}**\n\n"
        f"🇺🇸 Ciclo: **{pres_cycle_atual}**\n\n"
        f"₿ Ciclo: **{halv_cycle_atual}**"
    )
    aba = st.radio(
        "Selecione a Análise:",
        ["Sazonalidade (Heatmap)", "Ciclos de Mercado", "Risk Metric (DCA)", "MVRV Z-Score", "Médias Móveis"],
    )


# --- ABA 1: SAZONALIDADE ---
if aba == "Sazonalidade (Heatmap)":
    help_heat = """Retornos percentuais históricos.  \n
Verde: Meses/Trimestres positivos.  \n
Vermelho: Meses/Trimestres negativos.  \n
AVERAGE/MEDIAN no final ajudam a identificar vieses sazonais."""
    st.header("📅 Seasonality Returns", help=help_heat)
    
    c1, c2, c3 = st.columns(3)
    asset_name = c1.selectbox("Ativo", list(ASSET_TICKERS.keys()))
    view_mode = c2.selectbox("Frequência", ["Monthly Returns (%)", "Quarterly Returns (%)"])

    is_sp500 = asset_name == "S&P 500"
    if is_sp500:
        cycle_options = ["Todos os Anos"] + list(PRES_MAP.values())
        filter_type = c3.selectbox("Filtrar por Ciclo Político", cycle_options)
        cycle_col = "PresCycle"
    else:
        cycle_options = ["Todos os Anos"] + list(HALV_MAP.values())
        filter_type = c3.selectbox("Filtrar por Ciclo Halving", cycle_options)
        cycle_col = "HalvCycle"

    df_main = load_data(asset_name)
    if df_main.empty:
        st.stop()

    df_h = df_main[df_main[cycle_col] == filter_type] if filter_type != "Todos os Anos" else df_main

    is_monthly = "Monthly" in view_mode
    group_key = "Month" if is_monthly else "Quarter"
    cols_names = (
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if is_monthly
        else ["Q1", "Q2", "Q3", "Q4"]
    )

    pivot_df = df_h.groupby(["Year", group_key])["Price"].last().unstack()
    all_prices = df_main.groupby(["Year", group_key])["Price"].last().unstack()
    returns_df = pivot_df.pct_change(axis=1) * 100

    # LINHA CORRIGIDA: Cálculo direto e linear para evitar quebras de parênteses no interpretador
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
            if i >= len(vals) - 2:
                colors.append("#334155")
            elif pd.isnull(v):
                colors.append("#0f172a")
            elif v > 0:
                colors.append("#10b981")
            else:
                colors.append("#ef4444")
        cell_colors.append(colors)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header_vals, fill_color="#334155", align="center", font=dict(color="white")),
                cells=dict(values=cell_vals, fill_color=cell_colors, align="center", font=dict(color="white"), height=30)
            )
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=450 + (len(years) * 30), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 2: CICLOS DE MERCADO ---
elif aba == "Ciclos de Mercado":
    help_cycle = """Compara o ROI YTD do ativo atual com anos anteriores na mesma fase do ciclo estrutural.  \n
A linha verde sólida representa o ano corrente."""
    st.header("📈 Market Cycle ROI Comparison", help=help_cycle)
    c1, c2 = st.columns(2)
    asset_name = c1.selectbox("Ativo", list(ASSET_TICKERS.keys()))

    is_sp500 = asset_name == "S&P 500"
    col_c = "PresCycle" if is_sp500 else "HalvCycle"
    
    if is_sp500:
        ciclo = c2.selectbox("Ciclo Político", list(PRES_MAP.values()))
    else:
        ciclo = c2.selectbox("Ciclo Halving", list(HALV_MAP.values()))

    df_cycle = load_data(asset_name)
    if df_cycle.empty:
        st.stop()

    fig = go.Figure()

    df_hist = df_cycle[(df_cycle[col_c] == ciclo) & (df_cycle["Year"] < ano_atual)]
    for yr in sorted(df_hist["Year"].unique()):
        df_yr = df_hist[df_hist["Year"] == yr]
        fig.add_trace(
            go.Scatter(
                x=df_yr["DayOfYear"], y=df_yr["ROI"], name=str(yr), text=df_yr["HoverDate"],
                hovertemplate="%{text}<br>ROI: %{y:.2f}x<extra>" + str(yr) + "</extra>",
                line=dict(width=1), opacity=0.3
            )
        )

    stats = df_hist.groupby("DayOfYear")["ROI"].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats["DayOfYear"], y=stats["ROI"], name="Média Histórica", line=dict(color="white", dash="dash", width=2)))

    df_curr = df_cycle[df_cycle["Year"] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(
            go.Scatter(
                x=df_curr["DayOfYear"], y=df_curr["ROI"], name=str(ano_atual), text=df_curr["HoverDate"],
                hovertemplate="%{text}<br>ROI: %{y:.2f}x<extra>" + str(ano_atual) + "</extra>",
                line=dict(color="#00FFA3", width=3)
            )
        )

    fig.update_layout(
        template="plotly_dark", height=700, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Dia do Ano", yaxis_title="ROI (desde 1 Jan)", legend=dict(orientation="v", x=1.01, y=1)
    )
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 3: RISK METRIC & DYNAMIC DCA ---
elif aba == "Risk Metric (DCA)":
    help_risk = """Métrica de risco de 0 a 1 inspirada no Into The Cryptoverse (Benjamin Cowen).  \n
Abaixo de 0.5: Zona de acumulação e compras estratégicas (DCA In).  \n
Acima de 0.5: Zona de distribuição e realização de lucros (DCA Out)."""
    st.header("📊 Asset Risk Metric & Dynamic DCA Simulator", help=help_risk)
    
    asset_name = st.selectbox("Selecione o Ativo para Análise de Risco", ["Bitcoin (BTC)", "Ethereum (ETH)"])
    df_risk = load_data(asset_name)
    
    if df_risk.empty:
        st.stop()
        
    df_risk['SMA_140'] = df_risk['Price'].rolling(140).mean()
    df_risk['Dev_SMA'] = np.log(df_risk['Price'] / df_risk['SMA_140'])
    
    GENESIS = pd.Timestamp("2009-01-03")
    df_risk['Time_Index'] = (df_risk['Date_Clean'] - GENESIS).dt.days + 1
    df_risk['Fair_Value_Log'] = np.log(df_risk['Time_Index']) * 1.8
    df_risk['Dev_Fair'] = np.log(df_risk['Price']) - df_risk['Fair_Value_Log']
    
    raw_risk = df_risk['Dev_SMA'].fillna(0) * 0.5 + df_risk['Dev_Fair'].fillna(0) * 0.5
    min_r = raw_risk.expanding().min()
    max_r = raw_risk.expanding().max()
    df_risk['Risk'] = (raw_risk - min_r) / (max_r - min_r)
    
    df_risk = df_risk.dropna(subset=['SMA_140']).copy()
    
    current_row = df_risk.iloc[-1]
    current_price = current_row['Price']
    current_risk = current_row['Risk']
    
    if current_risk >= 0.8: zone_desc, zone_color, mult = "Distribuição Máxima 🔴", "#ef4444", 0.0
    elif current_risk >= 0.6: zone_desc, zone_color, mult = "DCA Out Ativo (Vender) 🟠", "#f59e0b", 0.0
    elif current_risk >= 0.5: zone_desc, zone_color, mult = "Zona Neutra / Alerta 🟡", "#eab308", 0.5
    elif current_risk >= 0.4: zone_desc, zone_color, mult = "Zona Neutra / Acumulação ⚪", "#94a3b8", 1.0
    elif current_risk >= 0.2: zone_desc, zone_color, mult = "DCA In Ativo (Comprar) 🟢", "#10b981", 1.5
    else: zone_desc, zone_color, mult = "Acumulação Pesada 💎", "#059669", 2.5
        
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        st.markdown(f'<div class="stat-card">Preço Atual ({asset_name})<div class="stat-val">${current_price:,.2f}</div></div>', unsafe_allow_html=True)
    with cd2:
        st.markdown(f'<div class="stat-card">Métrica de Risco<div class="stat-val" style="color: {zone_color};">{current_risk:.4f}</div></div>', unsafe_allow_html=True)
    with cd3:
        st.markdown(f'<div class="stat-card">Estado do Mercado<div class="stat-val" style="color: {zone_color};">{zone_desc}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.subheader("🧮 Simulador de Compras Dinâmicas (Budget Allocation)")
    cc1, cc2 = st.columns(2)
    budget_total = cc1.number_input("Budget Inicial Total (€)", min_value=100, max_value=1000000, value=5000, step=500)
    total_parts = cc2.number_input("Número de Parcelas Desejadas (Semanas/Meses)", min_value=4, max_value=104, value=20, step=1)
    
    base_tranche = budget_total / total_parts
    dynamic_buy = base_tranche * mult
    
    st.markdown(f"""
    <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
        <h4>📋 Plano de Ação para este Período:</h4>
        <ul>
            <li>O teu investimento base padrão por parcela é de: <b>{base_tranche:,.2f} €</b></li>
            <li>Multiplicador atual aplicado pelo Risco (x{mult}): <b style="color:{zone_color};">{mult}x</b></li>
            <li><b>Valor a investir AGORA: <span style="font-size:20px; color:#00FFA3;">{dynamic_buy:,.2f} €</span></b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_risk['Date_Clean'], y=df_risk['Price'], name="Preço (USD)", line=dict(color='rgba(255,255,255,0.15)', width=1), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_risk['Date_Clean'], y=df_risk['Risk'], name="Risk Metric", line=dict(color='#3b82f6', width=2), yaxis="y1"))
    
    fig.add_hrect(y0=0.0, y1=0.2, fillcolor="green", opacity=0.12, layer="below", line_width=0)
    fig.add_hrect(y0=0.2, y1=0.4, fillcolor="green", opacity=0.04, layer="below", line_width=0)
    fig.add_hrect(y0=0.6, y1=0.8, fillcolor="red", opacity=0.04, layer="below", line_width=0)
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor="red", opacity=0.12, layer="below", line_width=0)
    
    fig.update_layout(
        template="plotly_dark", height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Risk Level (0 - 1)", side="right", range=[0, 1], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis2=dict(type="log", overlaying="y", side="left", showgrid=False, title="Preço (USD)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 4: MVRV Z-SCORE ---
elif aba == "MVRV Z-Score":
    help_mvrv = """Mete em perspetiva a sobrevalorização ou subvalorização do Bitcoin.  \n
Z-Score na zona vermelha sugere tetos de ciclo; na zona verde sugere fundos históricos de acumulação."""
    st.header("📈 Bitcoin MVRV Z-Score (On-Chain Approximado)", help=help_mvrv)

    st.info(
        "⚠️ O Z-Score aqui é uma **aproximação** calculada com preços públicos e supply estimado. "
        "Para dados on-chain reais, consulta [Glassnode](https://glassnode.com) ou [LookIntoBitcoin](https://www.lookintobitcoin.com).",
        icon="ℹ️"
    )

    df_mvrv = load_data("Bitcoin (BTC)")
    if df_mvrv.empty:
        st.stop()

    df_mvrv["Supply"] = df_mvrv["Date_Clean"].apply(supply_btc_aproximado)
    df_mvrv["MC"] = df_mvrv["Price"] * df_mvrv["Supply"]
    df_mvrv["RC"] = df_mvrv["Price"].rolling(365).mean() * df_mvrv["Supply"]
    df_mvrv["Z"] = (df_mvrv["MC"] - df_mvrv["RC"]) / (df_mvrv["MC"].rolling(365).std())
    df_mvrv["Z_Calib"] = (df_mvrv["Z"] * 2.5) + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_mvrv["Date_Clean"], y=df_mvrv["MC"], name="Market Cap", line=dict(color="white"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_mvrv["Date_Clean"], y=df_mvrv["RC"], name="Realized Cap (aprox.)", line=dict(color="#3498db", dash="dot"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_mvrv["Date_Clean"], y=df_mvrv["Z_Calib"], name="Z-Score (calibrado)", line=dict(color="#f39c12"), yaxis="y1"))
    
    fig.add_hrect(y0=7, y1=10, fillcolor
