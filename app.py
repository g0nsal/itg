import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURAÇÃO E ESTILO ---
st.set_page_config(page_title="ITG Analytics", layout="wide")

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
        }
        h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- 2. CACHE DE DADOS ---
@st.cache_data(ttl=3600)
def load_raw_data(ticker, start_date="2010-07-17"):
    raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex): 
        raw.columns = raw.columns.get_level_values(0)
    
    df = raw[['Close']].reset_index()
    df.columns = ['Date', 'Price']
    df['Date_Clean'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[~((df['Date_Clean'].dt.month == 2) & (df['Date_Clean'].dt.day == 29))]

    df['Year'] = df['Date_Clean'].dt.year
    df['Month'] = df['Date_Clean'].dt.month
    df['Quarter'] = df['Date_Clean'].dt.quarter
    df['DayOfYear'] = df.groupby('Year').cumcount() + 1

    # Ciclos
    df['HalvCycle'] = df['Year'].apply(lambda y: {0:"Halving Year", 1:"Post-Halving Year", 2:"Bear Year", 3:"Pre-Halving Year"}.get(y % 4))
    df['PresCycle'] = df['Year'].apply(lambda y: {0:"Election Year", 1:"Post-Election Year", 2:"Midterm Year", 3:"Pre-Election Year"}.get(y % 4))

    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    
    datas_h = [(datetime(2023, 1, 1) + timedelta(days=i)).strftime('%d/%b') for i in range(366)]
    df['HoverDate'] = df['DayOfYear'].apply(lambda x: datas_h[min(int(x)-1, 364)])
    return df

# --- 3. SIDEBAR ---
ano_atual = 2026
with st.sidebar:
    st.title("🚀 ITG Analytics")
    st.markdown("---")
    st.info(f"📅 Ano Atual: {ano_atual}\n\n🇺🇸 Ciclo: **Midterm Year**\n\n₿ Ciclo: **Bear Year**")
    aba = st.radio("Selecione a Análise:", 
                   ["Sazonalidade (Heatmap)", "Ciclos de Mercado", "MVRV Z-Score", "Médias Móveis"])

# --- ABA 1: SAZONALIDADE ---
if aba == "Sazonalidade (Heatmap)":
    st.header("📅 Seasonality Returns")
    c1, c2, c3 = st.columns(3)
    asset_name = c1.selectbox("Ativo", ["Bitcoin (BTC)", "Ethereum (ETH)", "S&P 500"])
    view_mode = c2.selectbox("Frequência", ["Monthly Returns (%)", "Quarterly Returns (%)"])
    
    if asset_name == "S&P 500":
        filter_type = c3.selectbox("Filtrar por Ciclo Político", ["Todos os Anos", "Election Year", "Midterm Year", "Pre-Election Year", "Post-Election Year"])
        df_main = load_raw_data("^GSPC", start_date="1927-12-30")
        cycle_col = 'PresCycle'
    else:
        filter_type = c3.selectbox("Filtrar por Ciclo Halving", ["Todos os Anos", "Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"])
        ticker = "BTC-USD" if "Bitcoin" in asset_name else "ETH-USD"
        df_main = load_raw_data(ticker)
        cycle_col = 'HalvCycle'

    df_h = df_main.copy()
    if filter_type != "Todos os Anos":
        df_h = df_h[df_h[cycle_col] == filter_type]

    is_monthly = "Monthly" in view_mode
    group_key = 'Month' if is_monthly else 'Quarter'
    cols_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] if is_monthly else ['Q1','Q2','Q3','Q4']
    
    pivot_df = df_h.groupby(['Year', group_key])['Price'].last().unstack()
    all_prices = df_main.groupby(['Year', group_key])['Price'].last().unstack()
    returns_df = pivot_df.pct_change(axis=1) * 100
    
    last_col = 12 if is_monthly else 4
    for yr in returns_df.index:
        if yr-1 in all_prices.index:
            returns_df.at[yr, 1] = ((all_prices.at[yr, 1] / all_prices.at[yr-1, last_col]) - 1) * 100
    
    returns_df.columns = cols_names
    avg, med = returns_df.mean(), returns_df.median()
    years = [str(y) for y in returns_df.index.tolist()[::-1]]
    
    header_vals = ["<b>Year</b>"] + list(returns_df.columns)
    cell_vals = [years + ["<b>Average</b>", "<b>Median</b>"]]
    cell_colors = [["#1e293b"] * (len(years) + 2)]

    for col in returns_df.columns:
        vals = returns_df[col].tolist()[::-1] + [avg[col], med[col]]
        cell_vals.append([f"{v:+.2f}%" if pd.notnull(v) else "-" for v in vals])
        colors = []
        for i, v in enumerate(vals):
            if i >= len(vals)-2: colors.append("#334155")
            elif pd.isnull(v): colors.append("#0f172a")
            elif v > 0: colors.append("#10b981")
            else: colors.append("#ef4444")
        cell_colors.append(colors)

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_vals, fill_color='#334155', align='center', font=dict(color='white')),
        cells=dict(values=cell_vals, fill_color=cell_colors, align='center', font=dict(color='white'), height=30))])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=450 + (len(years) * 30), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 2: CICLOS DE MERCADO ---
elif aba == "Ciclos de Mercado":
    st.header("📈 Market Cycle ROI Comparison")
    c1, c2 = st.columns(2)
    asset_name = c1.selectbox("Ativo", ["Bitcoin (BTC)", "Ethereum (ETH)", "S&P 500"])
    
    if asset_name == "S&P 500":
        ciclo = c2.selectbox("Ciclo Político", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
        df_cycle = load_raw_data("^GSPC", start_date="1950-01-01")
        col_c = 'PresCycle'
    else:
        ciclo = c2.selectbox("Ciclo Halving", ["Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"])
        ticker = "BTC-USD" if "Bitcoin" in asset_name else "ETH-USD"
        df_cycle = load_raw_data(ticker)
        col_c = 'HalvCycle'

    fig = go.Figure()
    df_hist = df_cycle[(df_cycle[col_c] == ciclo) & (df_cycle['Year'] < ano_atual)]
    for yr in sorted(df_hist['Year'].unique()):
        df_yr = df_hist[df_hist['Year'] == yr]
        fig.add_trace(go.Scatter(x=df_yr['DayOfYear'], y=df_yr['ROI'], name=str(yr), line=dict(width=1), opacity=0.3))
    
    stats = df_hist.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name="Média", line=dict(color='white', dash='dash')))
    
    df_curr = df_cycle[df_cycle['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"2026", line=dict(color='#00FFA3', width=3)))
    
    fig.update_layout(template="plotly_dark", height=700, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 3: MVRV Z-SCORE ---
elif aba == "MVRV Z-Score":
    st.header("📈 Bitcoin MVRV Z-Score (On-Chain)")
    df_mvrv = load_raw_data("BTC-USD")
    supply = 19750000 
    df_mvrv['MC'] = df_mvrv['Price'] * supply
    df_mvrv['RC'] = (df_mvrv['Price'].rolling(365).mean()) * supply
    df_mvrv['Z'] = (df_mvrv['MC'] - df_mvrv['RC']) / (df_mvrv['MC'].rolling(365).std())
    df_mvrv['Z_Calib'] = (df_mvrv['Z'] * 2.5) + 0.5
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_mvrv['Date_Clean'], y=df_mvrv['MC'], name="Market Cap", line=dict(color='white'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_mvrv['Date_Clean'], y=df_mvrv['RC'], name="Realized Cap", line=dict(color='#3498db', dash='dot'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_mvrv['Date_Clean'], y=df_mvrv['Z_Calib'], name="Z-Score", line=dict(color='#f39c12'), yaxis="y1"))
    fig.add_hrect(y0=7, y1=10, fillcolor="red", opacity=0.15)
    fig.add_hrect(y0=-0.5, y1=0.2, fillcolor="green", opacity=0.15)
    fig.update_layout(template="plotly_dark", height=750, paper_bgcolor='rgba(0,0,0,0)', 
                      yaxis=dict(title="Z-Score", side="right", range=[-1.5, 11]), 
                      yaxis2=dict(type="log", overlaying="y", side="left", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 4: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis":
    st.header("📉 Weekly Moving Averages")
    asset_name = st.selectbox("Ativo", ["Bitcoin (BTC)", "Ethereum (ETH)", "S&P 500"])
    ticker = "BTC-USD" if "Bitcoin" in asset_name else ("ETH-USD" if "Ethereum" in asset_name else "^GSPC")
    df_ma = load_raw_data(ticker).set_index('Date_Clean')
    df_w = df_ma['Price'].resample('W').last().to_frame()
    for p in [20, 50, 100, 200]: 
        df_w[f'{p} SMA'] = df_w['Price'].rolling(window=p).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name='Price', line=dict(color='white')))
    for p in [20, 50, 100, 200]: 
        fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f'{p} SMA'], name=f'{p} SMA', opacity=0.7))
    fig.update_layout(template="plotly_dark", height=750, yaxis_type="log", paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
