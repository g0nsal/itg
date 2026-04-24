import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. CONFIGURAÇÃO E ESTILO ---
st.set_page_config(page_title="ITG Crypto Analytics", layout="wide")
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
        raw = yf.download(tickers[asset_type], start="2010-07-17", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        df = raw[['Close']].reset_index()
        df.columns = ['Date', 'Price']

    # Normalização de Datas
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df.rename(columns={date_col: 'Date_Clean'}, inplace=True)
    df = df[~((df['Date_Clean'].dt.month == 2) & (df['Date_Clean'].dt.day == 29))]
    
    # Cálculos Base
    df['Year'] = df['Date_Clean'].dt.year
    df['DayOfYear'] = df.groupby('Year').cumcount() + 1
    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    
    # Lógica Ciclo Presidencial
    def get_presidential_cycle(year):
        return {0: "Election Year", 1: "Post-Election Year", 2: "Midterm Year", 3: "Pre-Election Year"}.get(year % 4)
    
    # Lógica Ciclo Halving (Referência 2024 = Halving Year = 0)
    def get_halving_cycle(year):
        return {0: "Halving Year", 1: "Post-Halving Year", 2: "Bear Year", 3: "Pre-Halving Year"}.get(year % 4)

    df['PresCycle'] = df['Year'].apply(get_presidential_cycle)
    df['HalvCycle'] = df['Year'].apply(get_halving_cycle)
    
    datas_h = [(datetime(2023, 1, 1) + timedelta(days=i)).strftime('%d/%b') for i in range(366)]
    df['HoverDate'] = df['DayOfYear'].apply(lambda x: datas_h[min(int(x)-1, 364)])
    return df

# --- 3. SIDEBAR E CABEÇALHO ---
ano_atual = 2026 # Forçado conforme solicitado

with st.sidebar:
    st.title("🚀 ITG Analytics")
    st.markdown(f"**Current Year:** {ano_atual}")
    st.info(f"🇺🇸 Presidential: **Midterm Year**\n\n₿ Halving: **Bear Year**")
    aba = st.radio("Navegação", ["Ciclos Presidenciais (ROI)", "Ciclos de Halving (BTC)", "MVRV Z-Score", "Médias Móveis Semanais"])
    st.markdown("---")

# --- ABA 1: CICLOS PRESIDENCIAIS ---
if aba == "Ciclos Presidenciais (ROI)":
    st.header(f"📊 ROI vs Presidential Cycles ({ano_atual}: Midterm)")
    c1, c2 = st.columns(2)
    asset = c1.selectbox("Ativo", ["BTC", "S&P 500", "BTC/SPX Ratio"])
    cycle = c2.selectbox("Ciclo", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
    
    df = load_data(asset)
    limite_x = st.slider("Zoom X (Dias)", 30, 365, 365)

    fig = go.Figure()
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}",
                                 line=dict(color='red', width=4), customdata=df_curr['HoverDate'], hovertemplate="%{customdata}<extra></extra>"))
    
    df_h = df[(df['PresCycle'] == cycle) & (df['Year'] < ano_atual)]
    stats = df_h.groupby('DayOfYear')['ROI'].agg(['mean', 'std']).reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean']+stats['std'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean']-stats['std'], mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor='rgba(150, 150, 150, 0.15)', name='1 SD'))
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean'], name=f'Média {cycle}', line=dict(color='white', dash='dash')))

    fig.update_layout(template="plotly_dark", height=700, xaxis_range=[0, limite_x], hovermode="x unified", yaxis_title="ROI YTD")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 2: CICLOS DE HALVING ---
elif aba == "Ciclos de Halving (BTC)":
    st.header(f"₿ BTC ROI vs Halving Cycles ({ano_atual}: Bear Year)")
    st.markdown("Comparação do ROI anual do Bitcoin baseado nos ciclos de halving de 4 anos.")
    
    cycle_h = st.selectbox("Fase do Halving", ["Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"], index=2)
    
    df = load_data("BTC")
    limite_x = st.slider("Zoom X (Dias)", 30, 365, 365)

    fig = go.Figure()
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}",
                                 line=dict(color='red', width=4), customdata=df_curr['HoverDate'], hovertemplate="%{customdata}<extra></extra>"))
    
    df_h = df[(df['HalvCycle'] == cycle_h) & (df['Year'] < ano_atual)]
    stats = df_h.groupby('DayOfYear')['ROI'].agg(['mean', 'std']).reset_index()
    
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean']+stats['std'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean']-stats['std'], mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor='rgba(150, 150, 150, 0.15)', name='1 SD'))
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean'], name=f'Média {cycle_h}', line=dict(color='orange', dash='dash')))

    # Anos individuais para contexto
    anos_ciclo = sorted(df_h['Year'].unique())
    for i, yr in enumerate(anos_ciclo):
        df_yr = df_h[df_h['Year'] == yr]
        fig.add_trace(go.Scatter(x=df_yr['DayOfYear'], y=df_yr['ROI'], name=str(yr), line=dict(width=1), opacity=0.3))

    fig.update_layout(template="plotly_dark", height=700, xaxis_range=[0, limite_x], hovermode="x unified", yaxis_title="ROI YTD")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 3: MVRV ---
elif aba == "MVRV Z-Score":
    st.header("📈 Bitcoin MVRV Z-Score")
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
    st.header("📉 BTC Weekly Moving Averages")
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
