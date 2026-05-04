import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="ITG Analytics", layout="wide")
st.markdown("<style>.main { background-color: #0e1117; }</style>", unsafe_allow_html=True)

# --- 2. CACHE DE DADOS ---
@st.cache_data(ttl=3600)
def load_data(asset_type):
    tickers = {
        "Bitcoin (BTC)": "BTC-USD", 
        "Ethereum (ETH)": "ETH-USD", 
        "S&P 500": "^GSPC"
    }
    ticker = tickers.get(asset_type, "BTC-USD")
    
    raw = yf.download(ticker, start="2010-07-17", auto_adjust=True, progress=False)
def load_raw_data(ticker, start_date="2010-07-17"):
    raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    
    df = raw[['Close']].reset_index()
    df.columns = ['Date', 'Price']
    df['Date_Clean'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[~((df['Date_Clean'].dt.month == 2) & (df['Date_Clean'].dt.day == 29))]

    # Colunas Temporais
    df['Year'] = df['Date_Clean'].dt.year
    df['Month'] = df['Date_Clean'].dt.month
    df['Quarter'] = df['Date_Clean'].dt.quarter
    df['DayOfYear'] = df.groupby('Year').cumcount() + 1

    # Cálculos de ROI e Ciclos
    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    # Ciclos
    df['HalvCycle'] = df['Year'].apply(lambda y: {0:"Halving Year", 1:"Post-Halving Year", 2:"Bear Year", 3:"Pre-Halving Year"}.get(y % 4))
    df['PresCycle'] = df['Year'].apply(lambda y: {0:"Election Year", 1:"Post-Election Year", 2:"Midterm Year", 3:"Pre-Election Year"}.get(y % 4))

    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    
    datas_h = [(datetime(2023, 1, 1) + timedelta(days=i)).strftime('%d/%b') for i in range(366)]
    df['HoverDate'] = df['DayOfYear'].apply(lambda x: datas_h[min(int(x)-1, 364)])
    return df
@@ -48,45 +40,48 @@
ano_atual = 2026 
with st.sidebar:
    st.title("🚀 ITG Analytics")
    asset_selected = st.selectbox("Ativo para Análise:", ["Bitcoin (BTC)", "Ethereum (ETH)", "S&P 500"])
    st.divider()
    st.info(f"📅 Ano Atual: {ano_atual}\n\n🇺🇸 Ciclo: **Midterm Year**\n\n₿ Ciclo: **Bear Year**")
    aba = st.radio("Selecione a Análise:", 
                   ["Sazonalidade (Heatmap)", "Ciclos de Halving", "Ciclos Presidenciais", "MVRV Z-Score", "Médias Móveis Semanais"])
                   ["Sazonalidade (Heatmap)", "Ciclos de Mercado", "MVRV Z-Score", "Médias Móveis"])

# Carregar dados
df_raw = load_data(asset_selected)

# --- ABA: SAZONALIDADE ---
# --- ABA 1: SAZONALIDADE ---
if aba == "Sazonalidade (Heatmap)":
    st.header(f"📅 {asset_selected} Seasonality")
    c1, c2 = st.columns(2)
    view_mode = c1.selectbox("Frequência", ["Monthly Returns (%)", "Quarterly Returns (%)"])
    cycle_filter = c2.selectbox("Filtrar por Ciclo", ["Todos os Anos", "Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"])
    st.header("📅 Seasonality Returns")
    c1, c2, c3 = st.columns(3)
    asset = c1.selectbox("Ativo", ["Bitcoin (BTC)", "Ethereum (ETH)", "S&P 500"])
    view_mode = c2.selectbox("Frequência", ["Monthly Returns (%)", "Quarterly Returns (%)"])

    df_h = df_raw.copy()
    if cycle_filter != "Todos os Anos":
        years_to_keep = df_h[df_h['HalvCycle'] == cycle_filter]['Year'].unique()
        df_h = df_h[df_h['Year'].isin(years_to_keep)]

    if "Monthly" in view_mode:
        pivot_df = df_h.groupby(['Year', 'Month'])['Price'].last().unstack()
        all_m_prices = df_raw.groupby(['Year', 'Month'])['Price'].last().unstack()
        returns_df = pivot_df.pct_change(axis=1) * 100
        for yr in returns_df.index:
            if yr-1 in all_m_prices.index:
                returns_df.at[yr, 1] = ((all_m_prices.at[yr, 1] / all_m_prices.at[yr-1, 12]) - 1) * 100
        cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Seletor dinâmico de filtro baseado no ativo
    if asset == "S&P 500":
        filter_type = c3.selectbox("Filtrar por Ciclo Político", ["Todos os Anos", "Election Year", "Midterm Year", "Pre-Election Year", "Post-Election Year"])
        df = load_raw_data("^GSPC", start_date="1927-12-30") # Dados históricos longos para SPX
        cycle_col = 'PresCycle'
    else:
        pivot_df = df_h.groupby(['Year', 'Quarter'])['Price'].last().unstack()
        all_q_prices = df_raw.groupby(['Year', 'Quarter'])['Price'].last().unstack()
        returns_df = pivot_df.pct_change(axis=1) * 100
        for yr in returns_df.index:
            if yr-1 in all_q_prices.index:
                returns_df.at[yr, 1] = ((all_q_prices.at[yr, 1] / all_q_prices.at[yr-1, 4]) - 1) * 100
        cols = ['Q1', 'Q2', 'Q3', 'Q4']
        filter_type = c3.selectbox("Filtrar por Ciclo Halving", ["Todos os Anos", "Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"])
        ticker = "BTC-USD" if "Bitcoin" in asset else "ETH-USD"
        df = load_raw_data(ticker)
        cycle_col = 'HalvCycle'

    df_h = df.copy()
    if filter_type != "Todos os Anos":
        df_h = df_h[df_h[cycle_col] == filter_type]

    # Lógica da Tabela
    is_monthly = "Monthly" in view_mode
    group_key = 'Month' if is_monthly else 'Quarter'
    cols_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] if is_monthly else ['Q1','Q2','Q3','Q4']
    
    pivot_df = df_h.groupby(['Year', group_key])['Price'].last().unstack()
    all_prices = df.groupby(['Year', group_key])['Price'].last().unstack()
    returns_df = pivot_df.pct_change(axis=1) * 100

    returns_df.columns = cols
    # Correção do primeiro período do ano
    last_col = 12 if is_monthly else 4
    for yr in returns_df.index:
        if yr-1 in all_prices.index:
            returns_df.at[yr, 1] = ((all_prices.at[yr, 1] / all_prices.at[yr-1, last_col]) - 1) * 100
    
    returns_df.columns = cols_names
    avg, med = returns_df.mean(), returns_df.median()
    years = [str(y) for y in returns_df.index.tolist()[::-1]]

@@ -106,78 +101,74 @@
        cell_colors.append(colors)

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_vals, fill_color='#1e2127', align='center', font=dict(color='white', size=14)),
        cells=dict(values=cell_vals, fill_color=cell_colors, align='center', font=dict(color='white', size=13), height=30))])
        header=dict(values=header_vals, fill_color='#1e2127', align='center', font=dict(color='white')),
        cells=dict(values=cell_vals, fill_color=cell_colors, align='center', font=dict(color='white'), height=30))])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=450 + (len(years) * 30))
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: CICLOS DE HALVING ---
elif aba == "Ciclos de Halving":
    st.header(f"₿ {asset_selected}: Halving Cycle ROI")
    fase = st.selectbox("Comparar fase:", ["Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"], index=2)
# --- ABA 2: CICLOS DE MERCADO (UNIFICADO) ---
elif aba == "Ciclos de Mercado":
    st.header("📈 Market Cycle ROI Comparison")
    c1, c2 = st.columns(2)
    asset = c1.selectbox("Ativo", ["Bitcoin (BTC)", "Ethereum (ETH)", "S&P 500", "BTC/SPX Ratio"])
    
    if asset == "S&P 500":
        ciclo = c2.selectbox("Ciclo Político", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
        df = load_raw_data("^GSPC", start_date="1950-01-01")
        col_c = 'PresCycle'
    else:
        ciclo = c2.selectbox("Ciclo Halving", ["Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"])
        ticker = "BTC-USD" if "Bitcoin" in asset else ("ETH-USD" if "Ethereum" in asset else "BTC/SPX Ratio")
        df = load_raw_data(ticker if ticker != "BTC/SPX Ratio" else "BTC-USD") # Simplificação para o ratio
        col_c = 'HalvCycle'

    fig = go.Figure()
    df_hist = df_raw[(df_raw['HalvCycle'] == fase) & (df_raw['Year'] < ano_atual)]
    df_hist = df[(df[col_c] == ciclo) & (df['Year'] < ano_atual)]
    for yr in sorted(df_hist['Year'].unique()):
        df_yr = df_hist[df_hist['Year'] == yr]
        fig.add_trace(go.Scatter(x=df_yr['DayOfYear'], y=df_yr['ROI'], name=str(yr), line=dict(width=1.2), opacity=0.4))
        fig.add_trace(go.Scatter(x=df_yr['DayOfYear'], y=df_yr['ROI'], name=str(yr), line=dict(width=1), opacity=0.3))
    
    stats = df_hist.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name="Média", line=dict(color='white', dash='dash')))
    df_curr = df_raw[df_raw['Year'] == ano_atual]
    
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=str(ano_atual), line=dict(color='#00FFA3', width=3)))
    fig.update_layout(template="plotly_dark", height=700, hovermode="x unified", yaxis_title="ROI YTD")
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"2026", line=dict(color='#00FFA3', width=3)))
    
    fig.update_layout(template="plotly_dark", height=700, yaxis_title="ROI YTD")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: MVRV Z-SCORE ---
# --- ABA 3: MVRV Z-SCORE (BITCOIN ONLY) ---
elif aba == "MVRV Z-Score":
    st.header(f"📈 {asset_selected} MVRV Z-Score")
    if "S&P" in asset_selected:
        st.warning("O MVRV Z-Score é uma métrica on-chain. Para o S&P 500, usamos uma aproximação baseada em Médias Móveis.")
    
    df = df_raw.copy()
    supply = 19750000 if "Bitcoin" in asset_selected else (120000000 if "Ethereum" in asset_selected else 1)
    st.header("📈 Bitcoin MVRV Z-Score (On-Chain)")
    df = load_raw_data("BTC-USD")
    supply = 19750000 
    df['MC'] = df['Price'] * supply
    df['RC'] = (df['Price'].rolling(365).mean()) * supply
    df['Z'] = (df['MC'] - df['RC']) / (df['MC'].rolling(365).std())
    df['Z_Calib'] = (df['Z'] * 2.5) + 0.5 
    df['Z_Calib'] = (df['Z'] * 2.5) + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['MC'], name="Market Cap", line=dict(color='white', width=1.5), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['RC'], name="Realized Cap (Proxy)", line=dict(color='#3498db', width=1.2, dash='dot'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['Z_Calib'], name="Z-Score", line=dict(color='#f39c12', width=1.8), yaxis="y1"))
    fig.add_hrect(y0=7, y1=10, fillcolor="red", opacity=0.15, line_width=0)
    fig.add_hrect(y0=-0.5, y1=0.2, fillcolor="green", opacity=0.15, line_width=0)
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['MC'], name="Market Cap", line=dict(color='white'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['RC'], name="Realized Cap", line=dict(color='#3498db', dash='dot'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['Z_Calib'], name="Z-Score", line=dict(color='#f39c12'), yaxis="y1"))
    fig.add_hrect(y0=7, y1=10, fillcolor="red", opacity=0.15, annotation_text="Danger Zone")
    fig.add_hrect(y0=-0.5, y1=0.2, fillcolor="green", opacity=0.15, annotation_text="Buy Zone")
    fig.update_layout(template="plotly_dark", height=750, yaxis=dict(title="Z-Score", side="right", range=[-1.5, 11]), 
                      yaxis2=dict(title="Capitais (USD)", side="left", type="log", overlaying="y", showgrid=False), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: CICLOS PRESIDENCIAIS ---
elif aba == "Ciclos Presidenciais":
    st.header(f"🇺🇸 {asset_selected}: Presidential Cycle")
    ciclo = st.selectbox("Ciclo:", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"], index=0)
    fig = go.Figure()
    df_hist = df_raw[(df_raw['PresCycle'] == ciclo) & (df_raw['Year'] < ano_atual)]
    for yr in sorted(df_hist['Year'].unique()):
        df_yr = df_hist[df_hist['Year'] == yr]
        fig.add_trace(go.Scatter(x=df_yr['DayOfYear'], y=df_yr['ROI'], name=str(yr), line=dict(width=1.2), opacity=0.4))
    stats = df_hist.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name="Média", line=dict(color='white', dash='dash')))
    df_curr = df_raw[df_raw['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=str(ano_atual), line=dict(color='#00FFA3', width=3)))
    fig.update_layout(template="plotly_dark", height=700, yaxis_title="ROI YTD")
                      yaxis2=dict(type="log", overlaying="y", side="left", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis Semanais":
    st.header(f"📉 {asset_selected} Weekly Moving Averages")
    df = df_raw.copy().set_index('Date_Clean')
# --- ABA 4: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis":
    st.header("📉 Weekly Moving Averages")
    asset = st.selectbox("Ativo", ["Bitcoin (BTC)", "Ethereum (ETH)", "S&P 500"])
    ticker = "BTC-USD" if "Bitcoin" in asset else ("ETH-USD" if "Ethereum" in asset else "^GSPC")
    df = load_raw_data(ticker).set_index('Date_Clean')
    df_w = df['Price'].resample('W').last().to_frame()
    for p in [20, 50, 100, 200]:
        df_w[f'{p}W SMA'] = df_w['Price'].rolling(window=p).mean()
    for p in [20, 50, 100, 200]: df_w[f'{p} SMA'] = df_w['Price'].rolling(window=p).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name='Price', line=dict(color='white', width=1.5)))
    for p in [20, 50, 100, 200]:
        fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f'{p}W SMA'], name=f'{p}W SMA', opacity=0.7))
    fig.update_layout(template="plotly_dark", height=750, yaxis_type="log", yaxis_title="Price (Log Scale)")
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name='Price', line=dict(color='white')))
    for p in [20, 50, 100, 200]: fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f'{p} SMA'], name=f'{p} SMA', opacity=0.7))
    fig.update_layout(template="plotly_dark", height=750, yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)
