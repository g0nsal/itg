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
    # LINHA ADICIONADA: "Ciclos (ITC Advanced)" acoplada aqui no menu de seleção
    aba = st.radio(
        "Selecione a Análise:",
        ["Sazonalidade (Heatmap)", "Ciclos de Mercado", "Ciclos (ITC Advanced)", "Risk Metric (DCA)", "Cycle Repeat (Bitbo)", "Rainbow Ribbon (Smart)", "Social Risk (Sentiment)", "MVRV Z-Score", "Médias Móveis"],
    )


# --- ABA 1: SAZONALIDADE ---
if aba == "Sazonalidade (Heatmap)":
    help_heat = """Retornos percentuais históricos. Verde: positivo. Vermelho: negativo."""
    st.header("📅 Seasonality Returns", help=help_heat)
    
    c1, c2, c3 = st.columns(3)
    asset_name = c1.selectbox("Ativo", list(ASSET_TICKERS.keys()))
    view_mode = c2.selectbox("Frequência", ["Monthly Returns (%)", "Quarterly Returns (%)"])

    is_sp500 = asset_name == "S&P 500"
    cycle_col = "PresCycle" if is_sp500 else "HalvCycle"
    cycle_options = ["Todos os Anos"] + (list(PRES_MAP.values()) if is_sp500 else list(HALV_MAP.values()))
    filter_type = c3.selectbox("Filtrar por Ciclo", cycle_options)

    df_main = load_data(asset_name)
    if df_main.empty:
        st.stop()

    df_h = df_main[df_main[cycle_col] == filter_type] if filter_type != "Todos os Anos" else df_main

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


# --- ABA 2: CICLOS DE MERCADO (A TUA VERSÃO ORIGINAL COM LINHAS) ---
elif aba == "Ciclos de Mercado":
    st.header("📈 Market Cycle ROI Comparison (Classic Lines)")
    c1, c2, c3 = st.columns(3)
    asset_name = c1.selectbox("Ativo", list(ASSET_TICKERS.keys()))

    is_sp500 = asset_name == "S&P 500"
    col_c = "PresCycle" if is_sp500 else "HalvCycle"
    
    if is_sp500:
        ciclo = c2.selectbox("Fase do Ciclo Político", list(PRES_MAP.values()))
    else:
        ciclo_tipo = c2.selectbox("Perspetiva de Análise", ["Ciclo de Halving", "Ciclo Político Americano"])
        if ciclo_tipo == "Ciclo de Halving":
            ciclo = c3.selectbox("Fase do Halving", list(HALV_MAP.values()))
            col_c = "HalvCycle"
        else:
            ciclo = c3.selectbox("Fase do Ciclo Político", list(PRES_MAP.values()))
            col_c = "PresCycle"

    df_cycle = load_data(asset_name)
    if df_cycle.empty: st.stop()

    fig = go.Figure()
    df_hist = df_cycle[(df_cycle[col_c] == ciclo) & (df_cycle["Year"] < ano_atual)]
    for yr in sorted(df_hist["Year"].unique()):
        df_yr = df_hist[df_hist["Year"] == yr]
        fig.add_trace(go.Scatter(x=df_yr["DayOfYear"], y=df_yr["ROI"], name=str(yr), text=df_yr["HoverDate"], hovertemplate="%{text}<br>ROI: %{y:.2f}x", line=dict(width=1), opacity=0.3))

    stats = df_hist.groupby("DayOfYear")["ROI"].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats["DayOfYear"], y=stats["ROI"], name="Média Histórica", line=dict(color="white", dash="dash", width=2)))

    df_curr = df_cycle[df_cycle["Year"] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr["DayOfYear"], y=df_curr["ROI"], name=str(ano_atual), text=df_curr["HoverDate"], hovertemplate="%{text}<br>ROI: %{y:.2f}x", line=dict(color="#00FFA3", width=3)))

    fig.update_layout(template="plotly_dark", height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Dia do Ano", yaxis_title="ROI YTD")
    st.plotly_chart(fig, use_container_width=True)


# --- VERSÃO EXPERIMENTAL SOLICITADA: CICLOS (ITC ADVANCED COM SOMBRAS E MESES) ---
elif aba == "Ciclos (ITC Advanced)":
    st.header("📈 Market Cycle ROI Comparison (ITC Advanced Standard)")
    c1, c2, c3 = st.columns(3)
    asset_name = c1.selectbox("Ativo (Advanced)", list(ASSET_TICKERS.keys()))

    is_sp500 = asset_name == "S&P 500"
    col_c = "PresCycle" if is_sp500 else "HalvCycle"
    
    if is_sp500:
        ciclo = c2.selectbox("Fase do Ciclo Político (Advanced)", list(PRES_MAP.values()))
    else:
        ciclo_tipo = c2.selectbox("Perspetiva de Análise (Advanced)", ["Ciclo de Halving", "Ciclo Político Americano"])
        if ciclo_tipo == "Ciclo de Halving":
            ciclo = c3.selectbox("Fase do Halving (Advanced)", list(HALV_MAP.values()))
            col_c = "HalvCycle"
        else:
            ciclo = c3.selectbox("Fase do Ciclo Político (Advanced)", list(PRES_MAP.values()))
            col_c = "PresCycle"

    df_cycle = load_data(asset_name)
    if df_cycle.empty: st.stop()

    fig = go.Figure()
    df_hist = df_cycle[(df_cycle[col_c] == ciclo) & (df_cycle["Year"] < ano_atual)]
    
    if not df_hist.empty:
        stats_bounds = df_hist.groupby("DayOfYear")["ROI"].agg(["min", "max", "mean"]).reset_index()
        fig.add_trace(go.Scatter(x=stats_bounds["DayOfYear"], y=stats_bounds["max"], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=stats_bounds["DayOfYear"], y=stats_bounds["min"], mode='lines', fill='tonexty', fillcolor='rgba(148, 163, 184, 0.15)', line=dict(width=0), name="Standard Deviation Range / Bounds", hoverinfo='skip'))

    for yr in sorted(df_hist["Year"].unique()):
        df_yr = df_hist[df_hist["Year"] == yr]
        fig.add_trace(go.Scatter(x=df_yr["DayOfYear"], y=df_yr["ROI"], name=str(yr), text=df_yr["HoverDate"], hovertemplate="%{text}<br>ROI: %{y:.2f}x", line=dict(width=1), opacity=0.4))
    
    if not df_hist.empty:
        fig.add_trace(go.Scatter(x=stats_bounds["DayOfYear"], y=stats_bounds["mean"], name="Média do Ciclo (Avg)", line=dict(color="#ffffff", dash="dash", width=2)))
    
    df_curr = df_cycle[df_cycle["Year"] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr["DayOfYear"], y=df_curr["ROI"], name=str(ano_atual), text=df_curr["HoverDate"], hovertemplate="%{text}<br>ROI: %{y:.2f}x", line=dict(color="#ef4444", width=3)))

    meses_ticks_pos = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    meses_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig.update_layout(
        template="plotly_dark", height=650, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Month", tickmode='array', tickvals=meses_ticks_pos, ticktext=meses_labels, showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="ROI (Year-To-Date)", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 4: RISK METRIC (DCA) ---
elif aba == "Risk Metric (DCA)":
    st.header("📊 Into The Cryptoverse Risk Metric & Dynamic DCA")
    asset_name = st.selectbox("Selecione o Ativo para Análise de Risco", ["Bitcoin (BTC)", "Ethereum (ETH)"])
    df_risk = load_data(asset_name)
    if df_risk.empty: st.stop()
        
    df_risk['SMA_140'] = df_risk['Price'].rolling(140).mean()
    df_risk['Dev_SMA'] = np.log(df_risk['Price'] / df_risk['SMA_140'])
    
    GENESIS = pd.Timestamp("2009-01-03")
    df_risk['Time_Index'] = (df_risk['Date_Clean'] - GENESIS).dt.days + 1
    df_risk['Fair_Value_Log'] = np.log(df_risk['Time_Index']) * 1.8
    df_risk['Dev_Fair'] = np.log(df_risk['Price']) - df_risk['Fair_Value_Log']
    
    raw_risk = df_risk['Dev_SMA'].fillna(0) * 0.4 + df_risk['Dev_Fair'].fillna(0) * 0.6
    min_r = raw_risk.expanding().min()
    max_r = raw_risk.expanding().max()
    df_risk['Risk'] = ((raw_risk - min_r) / (max_r - min_r)) * 0.9 + 0.05
    df_risk = df_risk.dropna(subset=['SMA_140']).copy()
    
    current_row = df_risk.iloc[-1]
    current_price, current_risk = current_row['Price'], current_row['Risk']
    
    mode, fraction_text, mult = "HOLD", "0/15", 0.0
    if current_risk >= 0.6: mode, fraction_text = "SELL", "2/15"
    elif current_risk <= 0.3: mode, mult = "BUY", 1.5 if current_risk < 0.1 else 1.0

    cd1, cd2, cd3 = st.columns(3)
    with cd1: st.markdown(f'<div class="stat-card">Preço Atual<div class="stat-val">${current_price:,.2f}</div></div>', unsafe_allow_html=True)
    with cd2: st.markdown(f'<div class="stat-card">Risk Metric (ITC)<div class="stat-val">{current_risk:.4f}</div></div>', unsafe_allow_html=True)
    with cd3: st.markdown(f'<div class="stat-card">Estratégia do Ciclo<div class="stat-val">{mode}</div></div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_risk['Date_Clean'], y=df_risk['Price'], name="Preço (USD)", line=dict(color='rgba(255,255,255,0.1)'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_risk['Date_Clean'], y=df_risk['Risk'], name="Risk Metric", line=dict(color='#3b82f6'), yaxis="y1"))
    fig.update_layout(template="plotly_dark", height=500, paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0, 1]), yaxis2=dict(type="log", overlaying="y", side="left"))
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 5: CYCLE REPEAT (BITBO) ---
elif aba == "Cycle Repeat (Bitbo)":
    st.header("🔄 Cycle Repeat & Trajectory Projection")
    asset_name = st.selectbox("Selecione o Ativo para Projeção", ["Bitcoin (BTC)", "Ethereum (ETH)"])
    df_rep = load_data(asset_name)
    if df_rep.empty: st.stop()
        
    df_rep = df_rep.sort_values('Date_Clean').reset_index(drop=True)
    ultimo_preco_real, ultima_data_real = df_rep['Price'].iloc[-1], df_rep['Date_Clean'].iloc[-1]
    
    DIAS_CICLO = 1458
    df_janela = df_rep.iloc[-DIAS_CICLO:].copy()
    retornos_historicos = df_janela['Price'].pct_change().dropna().values
    
    precos_projetados = [ultimo_preco_real]
    for r in retornos_historicos: precos_projetados.append(precos_projetados[-1] * (1 + r))
    datas_futuras = [ultima_data_real + timedelta(days=i) for i in range(len(precos_projetados))]
    
    fig = go.Figure()
    df_v = df_rep[df_rep['Date_Clean'] >= (ultima_data_real - timedelta(days=700))]
    fig.add_trace(go.Scatter(x=df_v['Date_Clean'], y=df_v['Price'], name="Histórico Real", line=dict(color="#f8fafc")))
    fig.add_trace(go.Scatter(x=datas_futuras, y=precos_projetados, name="Projeção Ciclo Repetido", line=dict(color="#38bdf8", dash="dash")))
    fig.update_layout(template="plotly_dark", height=600, yaxis_type="log", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 6: RAINBOW RIBBON (SMART) ---
elif aba == "Rainbow Ribbon (Smart)":
    st.header("🌈 Into The Cryptoverse Smart Rainbow Ribbon")
    asset_name = st.selectbox("Selecione o Ativo para o Arco-Íris", list(ASSET_TICKERS.keys()))
    df_rb = load_data(asset_name)
    if df_rb.empty: st.stop()
        
    df_rb['20W_SMA'] = df_rb['Price'].rolling(140).mean()
    multipliers = [3.5, 2.4, 1.6, 1.0, 0.75, 0.55]
    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"]
    names = ["Maximum Bubble", "DCA Out", "Overheating", "20W SMA Support", "Accumulation", "Generational Bottom"]
    
    ultima_dt = df_rb['Date_Clean'].iloc[-1]
    df_v = df_rb[df_rb['Date_Clean'] >= (ultima_dt - timedelta(days=1200))]
    
    fig = go.Figure()
    for m, c, n in zip(multipliers, colors, names):
        fig.add_trace(go.Scatter(x=df_v['Date_Clean'], y=df_v['20W_SMA'] * m, name=n, line=dict(color=c, width=1.5)))
    fig.add_trace(go.Scatter(x=df_v['Date_Clean'], y=df_v['Price'], name="Preço Real", line=dict(color="#ffffff", width=2.5)))
    fig.update_layout(template="plotly_dark", height=650, yaxis_type="log", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 7: SOCIAL RISK (SENTIMENT) ---
elif aba == "Social Risk (Sentiment)":
    st.header("📢 Social Sentiment Risk Monitor (Proxy Real-Time)")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.subheader("📺 Métricas de Canais YouTube")
        views_score = st.slider("Visualizações Médias Semanais (Canais de Massa)", 10000, 500000, 85000, step=10000)
        subs_growth = st.slider("Novos Subscritores Semanais (Canais Macro)", 0, 50000, 2500, step=2000)
    with col_in2:
        st.subheader("📱 Métricas de Apps e Redes Sociais")
        coinbase_rank = st.number_input("Posição da App Coinbase na App Store EUA", min_value=1, max_value=500, value=310, step=10)
        x_fomo = st.selectbox("Sentimento Dominante no X", ["Apatia / Desinteresse Total 😴", "Desespero / Capitulação 📉", "Discussão Saudável / Acumulação ⚪", "FOMO / Ganância Inicial 🚀", "Euforia Parabólica Máxima 🔥"])

    v_risk = (views_score - 10000) / (500000 - 10000)
    s_risk = subs_growth / 50000
    c_risk = (500 - coinbase_rank) / (500 - 1)
    x_map = {"Apatia / Desinteresse Total 😴": 0.1, "Desespero / Capitulação 📉": 0.05, "Discussão Saudável / Acumulação ⚪": 0.3, "FOMO / Ganância Inicial 🚀": 0.65, "Euforia Parabólica Máxima 🔥": 0.95}
    
    social_risk_final = np.clip((v_risk * 0.3) + (s_risk * 0.2) + (c_risk * 0.3) + (x_map[x_fomo] * 0.2), 0.0, 1.0)
    
    cs1, cs2 = st.columns(2)
    with cs1: st.markdown(f'<div class="stat-card">Social Risk Score<div class="stat-val">{social_risk_final:.4f}</div></div>', unsafe_allow_html=True)
    with cs2: st.markdown(f'<div class="stat-card">Diagnóstico<div class="stat-val">Monitorizado</div></div>', unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=social_risk_final, title={'text': "Social Risk Dial"},
        gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#38bdf8"}, 'bgcolor': "#1e293b",
               'steps': [{'range': [0, 0.25], 'color': 'rgba(5, 150, 105, 0.2)'}, {'range': [0.6, 1.0], 'color': 'rgba(239, 68, 68, 0.2)'}]}
    ))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300, font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)


# --- ABA 8: MVRV Z-SCORE ---
elif aba == "MVRV Z-Score":
    st.header("📈 Bitcoin MVRV Z-Score (On-Chain Approximado)")
    df_mvrv = load_data("Bitcoin (BTC)")
    if df_mvrv.empty: st.stop()

    df_mvrv["Supply"] = df_mvrv["Date_Clean"].apply(supply_btc_aproximado)
    df_mvrv["MC"] = df_mvrv["Price"] * df_mvrv["Supply"]
    df_mvrv["RC"] = df_mvrv["Price"].rolling(365).mean() * df_mvrv["Supply"]
    df_mvrv["Z"] = (df_mvrv["MC"] - df_mvrv["RC"]) / (df_mvrv["MC"].rolling(365).std())
    df_mvrv["Z_Calib"] = (df_mvrv["Z"] * 2.5) + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_mvrv["Date_Clean"], y=df_mvrv["MC"], name="Market Cap", line=dict(color="white"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_mvrv["Date_Clean"], y=df_mvrv["Z_Calib"], name="Z-Score", line=dict(color="#f39c12"), yaxis="y1"))
    fig.update_layout(template="plotly_dark", height=600, paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[-1.5, 11]), yaxis2=dict(type="log", overlaying="y", side="left"))
    st.plotly_chart(fig, use_container_width=True)


# --- ABA 9: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis":
    st.header("📉 Weekly Moving Averages")
    asset_name = st.selectbox("Ativo", list(ASSET_TICKERS.keys()))
    df_ma = load_data(asset_name)
    if df_ma.empty: st.stop()
        
    df_w = df_ma.set_index("Date_Clean")["Price"].resample("W").last().to_frame()
    MA_PERIODS = [20, 50, 100, 200]
    MA_COLORS = ["#3b82f6", "#f59e0b", "#ec4899", "#a855f7"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name="Preço", line=dict(color="white")))
    for p, color in zip(MA_PERIODS, MA_COLORS):
        df_w[f"{p}W SMA"] = df_w["Price"].rolling(window=p).mean()
        fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f"{p}W SMA"], name=f"{p}W SMA", line=dict(color=color), opacity=0.7))
        
    fig.update_layout(template="plotly_dark", height=600, yaxis_type="log", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
