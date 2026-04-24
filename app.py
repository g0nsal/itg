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
        # Limpeza MultiIndex
        if isinstance(btc.columns, pd.MultiIndex): btc.columns = btc.columns.get_level_values(0)
        if isinstance(spx.columns, pd.MultiIndex): spx.columns = spx.columns.get_level_values(0)
        # Join e Ratio
        df = btc[['Close']].join(spx[['Close']], lsuffix='_btc', rsuffix='_spx').ffill().dropna()
        df['Price'] = df['Close_btc'] / df['Close_spx']
    else:
        raw = yf.download(tickers[asset_type], start="2010-07-17", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
        df = raw[['Close']].reset_index()
        df.columns = ['Date', 'Price']

    df['Date'] = pd.to_datetime(df['index' if 'index' in df.columns else 'Date']).dt.tz_localize(None)
    df = df[~((df['Date'].dt.month == 2) & (df['Date'].dt.day == 29))]
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df.groupby('Year').cumcount() + 1
    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    
    def get_cycle(year):
        return {0: "Election Year", 1: "Post-Election Year", 2: "Midterm Year", 3: "Pre-Election Year"}.get(year % 4)
    df['Cycle'] = df['Year'].apply(get_cycle)
    
    datas_h = [(datetime(2023, 1, 1) + timedelta(days=i)).strftime('%d/%b') for i in range(366)]
    df['HoverDate'] = df['DayOfYear'].apply(lambda x: datas_h[min(int(x)-1, 365)])
    return df

# --- 3. SIDEBAR NAVEGAÇÃO ---
with st.sidebar:
    st.title("🚀 ITG Analytics")
    aba = st.radio("Navegação", ["Ciclos Presidenciais (ROI)", "MVRV Z-Score", "Médias Móveis Semanais"])
    st.markdown("---")
    st.caption("v2.0 - Auto-update ativado")

# --- ABA 1: CICLOS ---
if aba == "Ciclos Presidenciais (ROI)":
    st.header("📊 ROI Analysis vs Presidential Cycles")
    c1, c2 = st.columns(2)
    asset = c1.selectbox("Ativo", ["BTC", "S&P 500", "BTC/SPX Ratio"])
    cycle = c2.selectbox("Ciclo", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
    
    df = load_data(asset)
    ano_atual = datetime.now().year
    limite_x = st.slider("Zoom X (Dias)", 30, 365, 365)

    fig = go.Figure()
    # Ano Atual
    df_curr = df[df['Year'] == ano_atual]
    fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}",
                             line=dict(color='red', width=4), customdata=df_curr['HoverDate'],
                             hovertemplate="%{customdata}<extra></extra>"))
    # Médias
    df_h = df[(df['Cycle'] == cycle) & (df['Year'] < ano_atual)]
    stats = df_h.groupby('DayOfYear')['ROI'].agg(['mean', 'std']).reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean']+stats['std'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean']-stats['std'], mode='lines', line=dict(width=0), 
                             fill='tonexty', fillcolor='rgba(150, 150, 150, 0.15)', name='1 SD'))
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean'], name='Média Histórica', line=dict(color='white', dash='dash')))

    fig.update_layout(template="plotly_dark", height=700, xaxis_range=[0, limite_x], hovermode="x unified",
                      yaxis_title="ROI YTD", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 2: MVRV ---
elif aba == "MVRV Z-Score":
    st.header("📈 Bitcoin MVRV Z-Score")
    df = load_data("BTC")
    s = 19600000
    df['mv'] = df['Price'] * s
    df['rp'] = df['Price'].rolling(365).mean()
    df['rv'] = df['rp'] * s
    mvrv = (df['Price'] / df['rp'])
    df['z'] = (mvrv - mvrv.rolling(365).mean()) / mvrv.rolling(365).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['z'], name="Z-Score", line=dict(color='#00ffcc'), yaxis="y1"))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['mv'], name="Market Cap", line=dict(color='white', width=1), yaxis="y2"))
    fig.update_layout(template="plotly_dark", height=750, yaxis2=dict(overlaying="y", side="right", type="log"), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 3: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis Semanais":
    st.header("📉 BTC Weekly Moving Averages")
    df = load_data("BTC")
    df.set_index('Date', inplace=True)
    df_w = df['Price'].resample('W').last().to_frame()
    periods = [8, 20, 50, 100, 200, 300, 400]
    for p in periods: df_w[f'{p}W SMA'] = df_w['Price'].rolling(window=p).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name='BTC Price', line=dict(color='white')))
    for p in periods: fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f'{p}W SMA'], name=f'{p}W SMA', opacity=0.6))
    fig.update_layout(template="plotly_dark", height=750, yaxis_type="log", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
