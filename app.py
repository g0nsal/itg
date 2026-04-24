import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURAÇÃO ---
st.set_page_config(page_title="ITG Analytics", layout="wide")
st.markdown("<style>.main { background-color: #0e1117; }</style>", unsafe_allow_html=True)

# --- 2. CACHE DE DADOS ---
@st.cache_data(ttl=3600)
def load_data(asset_type):
    tickers = {"BTC": "BTC-USD", "S&P 500": "^GSPC"}
    raw = yf.download(tickers.get(asset_type, "BTC-USD"), start="2010-07-17", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw[['Close']].reset_index()
    df.columns = ['Date', 'Price']
    df['Date_Clean'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[~((df['Date_Clean'].dt.month == 2) & (df['Date_Clean'].dt.day == 29))]
    
    df['Year'] = df['Date_Clean'].dt.year
    df['DayOfYear'] = df.groupby('Year').cumcount() + 1
    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    df['HalvCycle'] = df['Year'].apply(lambda y: {0:"Halving Year", 1:"Post-Halving Year", 2:"Bear Year", 3:"Pre-Halving Year"}.get(y % 4))
    df['PresCycle'] = df['Year'].apply(lambda y: {0:"Election Year", 1:"Post-Election Year", 2:"Midterm Year", 3:"Pre-Election Year"}.get(y % 4))
    
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
                   ["Sazonalidade (Heatmap)", "Ciclos de Halving (BTC)", "Ciclos Presidenciais (ROI)", "MVRV Z-Score", "Médias Móveis Semanais"])

# --- ABA: SAZONALIDADE (HEATMAP PROFISSIONAL) ---
if aba == "Sazonalidade (Heatmap)":
    st.header("📅 Bitcoin Seasonality Returns", help="Retornos históricos. Verde para positivo, Vermelho para negativo.")
    
    df_raw = load_data("BTC")
    c1, c2 = st.columns(2)
    view_mode = c1.selectbox("Frequência", ["Monthly Returns (%)", "Quarterly Returns (%)"])
    cycle_filter = c2.selectbox("Filtrar por Ciclo de Halving", ["Todos os Anos", "Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"])
    
    df_h = df_raw.copy()
    if cycle_filter != "Todos os Anos":
        df_h = df_h[df_h['HalvCycle'] == cycle_filter]
        
    if "Monthly" in view_mode:
        df_h['Month'] = df_h['Date_Clean'].dt.month
        pivot_df = df_h.groupby(['Year', 'Month'])['Price'].last().unstack()
        pivot_df = pivot_df.pct_change(axis=1) * 100
        # Corrigir Janeiro comparando com Dezembro do ano anterior
        all_prices = df_raw.groupby(['Year', 'Month'])['Price'].last().unstack()
        for yr in pivot_df.index:
            if yr-1 in all_prices.index:
                pivot_df.at[yr, 1] = ((all_prices.at[yr, 1] / all_prices.at[yr-1, 12]) - 1) * 100
        cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_df.columns = cols
    else:
        df_h['Quarter'] = df_h['Date_Clean'].dt.quarter
        pivot_df = df_h.groupby(['Year', 'Quarter'])['Price'].last().unstack()
        pivot_df = pivot_df.pct_change(axis=1) * 100
        all_q_prices = df_raw.groupby(['Year', 'Quarter'])['Price'].last().unstack()
        for yr in pivot_df.index:
            if yr-1 in all_q_prices.index:
                pivot_df.at[yr, 1] = ((all_q_prices.at[yr, 1] / all_q_prices.at[yr-1, 4]) - 1) * 100
        cols = ['Q1', 'Q2', 'Q3', 'Q4']
        pivot_df.columns = cols

    # Cálculos Stats
    avg = pivot_df.mean()
    med = pivot_df.median()
    
    # Preparar dados para a Tabela Plotly (Estilo Bitbo)
    years = pivot_df.index.astype(str).tolist()[::-1]
    rows = [years + ["Average", "Median"]]
    
    header_values = ["<b>Year</b>"] + list(pivot_df.columns)
    
    cell_values = [rows[0]] # Primeira coluna é o ano
    cell_colors = [["#1e2127"] * len(rows[0])] # Cor da coluna de anos
    text_colors = [["white"] * len(rows[0])]

    for col in pivot_df.columns:
        col_data = pivot_df[col].tolist()[::-1]
        col_avg = avg[col]
        col_med = med[col]
        
        vals = col_data + [col_avg, col_med]
        formatted_vals = [f"{v:+.2f}%" if pd.notnull(v) else "-" for v in vals]
        cell_values.append(formatted_vals)
        
        # Lógica de Cores
        colors = []
        t_colors = []
        for i, v in enumerate(vals):
            if i >= len(vals) - 2: # Média e Mediana
                colors.append("#333a41")
                t_colors.append("white")
            elif pd.isnull(v):
                colors.append("#0e1117")
                t_colors.append("gray")
            elif v > 0:
                colors.append("#26a69a") # Verde Bitbo
                t_colors.append("white")
            else:
                colors.append("#ef5350") # Vermelho Bitbo
                t_colors.append("white")
        cell_colors.append(colors)
        text_colors.append(t_colors)

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values, fill_color='#1e2127', align='center', font=dict(color='white', size=14)),
        cells=dict(values=cell_values, fill_color=cell_colors, align='center', font=dict(color=text_colors, size=13), height=30)
    )])
    
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=35 * len(rows[0]) + 100)
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: CICLOS DE HALVING ---
elif aba == "Ciclos de Halving (BTC)":
    st.header(f"₿ BTC: Comparativo de Ciclos de Halving")
    fase = st.selectbox("Fase:", ["Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"], index=2)
    df = load_data("BTC")
    fig = go.Figure()
    df_hist = df[(df['HalvCycle'] == fase) & (df['Year'] < ano_atual)]
    for yr in sorted(df_hist['Year'].unique()):
        df_yr = df_hist[df_hist['Year'] == yr]
        fig.add_trace(go.Scatter(x=df_yr['DayOfYear'], y=df_yr['ROI'], name=str(yr), line=dict(width=1), opacity=0.3))
    stats = df_hist.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name="Média", line=dict(color='white', dash='dash')))
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}", line=dict(color='#00FFA3', width=2.5)))
    fig.update_layout(template="plotly_dark", height=700, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: MVRV Z-SCORE ---
elif aba == "MVRV Z-Score":
    st.header("📈 Bitcoin MVRV Z-Score")
    df = load_data("BTC")
    supply = 19750000 
    df['MC'] = df['Price'] * supply
    df['RC'] = (df['Price'].rolling(365).mean()) * supply
    df['Z'] = (df['MC'] - df['RC']) / (df['MC'].rolling(365).std())
    df['Z_Calib'] = (df['Z'] * 2.5) + 0.5
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['MC'], name="Market Cap", line=dict(color='white'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['RC'], name="Realized Cap", line=dict(color='rgba(173,216,230,0.6)', dash='dot'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df['Date_Clean'], y=df['Z_Calib'], name="Z-Score", line=dict(color='#f39c12'), yaxis="y1"))
    fig.add_hrect(y0=7, y1=10, fillcolor="red", opacity=0.15)
    fig.add_hrect(y0=-0.5, y1=0.2, fillcolor="green", opacity=0.15)
    fig.update_layout(template="plotly_dark", height=750, yaxis=dict(title="Z-Score", side="right", range=[-1.5, 11]), yaxis2=dict(type="log", overlaying="y", side="left", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: CICLOS PRESIDENCIAIS ---
elif aba == "Ciclos Presidenciais (ROI)":
    st.header("📊 ROI vs Presidential Cycles")
    asset = st.selectbox("Ativo", ["BTC", "S&P 500", "BTC/SPX Ratio"])
    cycle = st.selectbox("Ciclo", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
    df = load_data(asset)
    fig = go.Figure()
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}", line=dict(color='#00FFA3', width=2.5)))
    df_h = df[(df['PresCycle'] == cycle) & (df['Year'] < ano_atual)]
    stats = df_h.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name='Média', line=dict(color='white', dash='dash')))
    fig.update_layout(template="plotly_dark", height=700)
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis Semanais":
    st.header("📉 BTC Weekly Moving Averages")
    df = load_data("BTC").set_index('Date_Clean')
    df_w = df['Price'].resample('W').last().to_frame()
    for p in [20, 50, 100, 200]: df_w[f'{p}W SMA'] = df_w['Price'].rolling(window=p).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name='Price', line=dict(color='white')))
    for p in [20, 50, 100, 200]: fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f'{p}W SMA'], name=f'{p}W SMA', opacity=0.7))
    fig.update_layout(template="plotly_dark", height=750, yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)
