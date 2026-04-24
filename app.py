import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. CONFIGURAÇÃO E ESTILO ---
st.set_page_config(page_title="ITG Analytics", layout="wide")
st.markdown("<style>.main { background-color: #0e1117; }</style>", unsafe_allow_html=True)

# --- 2. CACHE DE DADOS ---
@st.cache_data(ttl=3600)
def load_data(asset_type):
    tickers = {"BTC": "BTC-USD", "S&P 500": "^GSPC"}
    
    if asset_type == "BTC/SPX Ratio":
        btc = yf.download("BTC-USD", start="2010-07-17", auto_adjust=True, progress=False)
        spx = yf.download("^GSPC", start="2010-07-17", auto_adjust=True, progress=False)
        if isinstance(btc.columns, pd.MultiIndex): btc.columns = btc.columns.get_level_values(0)
        if isinstance(spx.columns, pd.MultiIndex): spx.columns = spx.columns.get_level_values(0)
        df = btc[['Close']].join(spx[['Close']], lsuffix='_btc', rsuffix='_spx').ffill().dropna()
        df['Price'] = df['Close_btc'] / df['Close_spx']
        df = df.reset_index()
    else:
        raw = yf.download(tickers.get(asset_type, "BTC-USD"), start="2010-07-17", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        df = raw[['Close']].reset_index()
        df.columns = ['Date', 'Price']

    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df.rename(columns={date_col: 'Date_Clean'}, inplace=True)
    df = df[~((df['Date_Clean'].dt.month == 2) & (df['Date_Clean'].dt.day == 29))]
    
    df['Year'] = df['Date_Clean'].dt.year
    df['DayOfYear'] = df.groupby('Year').cumcount() + 1
    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    
    df['PresCycle'] = df['Year'].apply(lambda y: {0:"Election Year", 1:"Post-Election Year", 2:"Midterm Year", 3:"Pre-Election Year"}.get(y % 4))
    df['HalvCycle'] = df['Year'].apply(lambda y: {0:"Halving Year", 1:"Post-Halving Year", 2:"Bear Year", 3:"Pre-Halving Year"}.get(y % 4))
    
    datas_h = [(datetime(2023, 1, 1) + timedelta(days=i)).strftime('%d/%b') for i in range(366)]
    df['HoverDate'] = df['DayOfYear'].apply(lambda x: datas_h[min(int(x)-1, 364)])
    return df

# --- 3. SIDEBAR ---
ano_atual = 2026 

with st.sidebar:
    st.title("🚀 ITG Analytics")
    st.subheader(f"📅 Ano Atual: {ano_atual}")
    st.info(f"🇺🇸 Ciclo Político: **Midterm Year**\n\n₿ Ciclo de Mercado: **Bear Year**")
    aba = st.radio("Selecione a Análise:", 
                   ["Ciclos de Halving (BTC)", "Ciclos Presidenciais (ROI)", "MVRV Z-Score", "Médias Móveis Semanais"])

# --- ABA 1: CICLOS DE HALVING ---
if aba == "Ciclos de Halving (BTC)":
    st.header(f"₿ BTC: Comparativo de Ciclos de Halving", 
              help="Este gráfico compara o ROI YTD do Bitcoin em anos específicos do ciclo de 4 anos. 'Bear Year' (2026) é o ano de ajuste histórico após a Bull Run do ano anterior.")
    
    fase = st.selectbox("Comparar anos de:", ["Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"], index=2)
    df = load_data("BTC")
    limite_x = st.slider("Zoom X (Dias)", 30, 365, 365)

    fig = go.Figure()
    df_hist = df[(df['HalvCycle'] == fase) & (df['Year'] < ano_atual)]
    anos_na_fase = sorted(df_hist['Year'].unique())
    cores = px.colors.qualitative.Pastel
    
    for i, yr in enumerate(anos_na_fase):
        df_yr = df_hist[df_hist['Year'] == yr]
        fig.add_trace(go.Scatter(x=df_yr['DayOfYear'], y=df_yr['ROI'], name=f"Ano {yr}",
                                 line=dict(width=1.2, color=cores[i % len(cores)]), opacity=0.3))

    stats = df_hist.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name="Média Ciclos", line=dict(color='white', width=1.5, dash='dash')))

    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}",
                                 line=dict(color='#00FFA3', width=2.5), 
                                 customdata=df_curr['HoverDate'], hovertemplate="%{customdata}<br>ROI: %{y:.2f}<extra></extra>"))

    fig.update_layout(template="plotly_dark", height=700, xaxis_range=[0, limite_x], hovermode="x unified",
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 2: CICLOS PRESIDENCIAIS ---
elif aba == "Ciclos Presidenciais (ROI)":
    st.header(f"📊 ROI vs Presidential Cycles", 
              help="Analisa o impacto do ciclo eleitoral dos EUA. O 'Midterm Year' (2026) refere-se ao ano das eleições de meio de mandato, geralmente associado a clareza política.")
    
    c1, c2 = st.columns(2)
    asset = c1.selectbox("Ativo", ["BTC", "S&P 500", "BTC/SPX Ratio"])
    cycle = c2.selectbox("Ciclo", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
    df = load_data(asset)
    limite_x = st.slider("Zoom X", 30, 365, 365)
    
    fig = go.Figure()
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}",
                                 line=dict(color='red' if asset != "S&P 500" else '#00FF00', width=2.5)))
    
    df_h = df[(df['PresCycle'] == cycle) & (df['Year'] < ano_atual)]
    stats = df_h.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name='Média Histórica', line=dict(color='white', dash='dash')))
    
    fig.update_layout(template="plotly_dark", height=700, xaxis_range=[0, limite_x], hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 3: MVRV ---
elif aba == "MVRV Z-Score":
    st.header("📈 Bitcoin MVRV Z-Score", 
              help="Indica sobrevalorização ou subvalorização. Z-Score alto (>7) sugere topo; Z-Score baixo (<0) sugere zona de acumulação.")
    df = load_data("BTC")
    s = 19600000
    df['mv'] = df['Price'] * s
    df['rp'] = df['Price'].rolling(365).mean()
    mvrv = (df['Price'] / df['rp'])
    df['z'] = (mvrv - mvrv.rolling(365).mean()) / mvrv.rolling(365).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['z'], name="Z-Score", line=dict(color='#00ffcc'), yaxis="y1"))
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['mv'], name="Market Cap", line=dict(color='white', width=1), yaxis="y2"))
    fig.update_layout(template="plotly_dark", height=750, yaxis2=dict(overlaying="y", side="right", type="log"), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 4: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis Semanais":
    st.header("📉 BTC Weekly Moving Averages", 
              help="Visualização de tendências de longo prazo em escala logarítmica. A 200W SMA é um suporte histórico crítico.")
    df = load_data("BTC")
    df.set_index('Date_Clean', inplace=True)
    df_w = df['Price'].resample('W').last().to_frame()
    periods = [8, 20, 50, 100, 200, 300, 400]
    for p in periods: df_w[f'{p}W SMA'] = df_w['Price'].rolling(window=p).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name='BTC Price', line=dict(color='white')))
    for p in periods: fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f'{p}W SMA'], name=f'{p}W SMA', opacity=0.6))
    fig.update_layout(template="plotly_dark", height=750, yaxis_type="log", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
