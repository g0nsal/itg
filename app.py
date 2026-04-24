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
    # Dicionário de ativos expandido
    tickers = {
        "Bitcoin (BTC)": "BTC-USD", 
        "Ethereum (ETH)": "ETH-USD", 
        "S&P 500": "^GSPC"
    }
    ticker = tickers.get(asset_type, "BTC-USD")
    
    raw = yf.download(ticker, start="2010-07-17", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    
    df = raw[['Close']].reset_index()
    df.columns = ['Date', 'Price']
    df['Date_Clean'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[~((df['Date_Clean'].dt.month == 2) & (df['Date_Clean'].dt.day == 29))]
    
    # Criar colunas temporais
    df['Year'] = df['Date_Clean'].dt.year
    df['Month'] = df['Date_Clean'].dt.month
    df['Quarter'] = df['Date_Clean'].dt.quarter
    df['DayOfYear'] = df.groupby('Year').cumcount() + 1
    
    # Cálculos de ROI e Ciclos
    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    df['HalvCycle'] = df['Year'].apply(lambda y: {0:"Halving Year", 1:"Post-Halving Year", 2:"Bear Year", 3:"Pre-Halving Year"}.get(y % 4))
    
    return df

# --- 3. SIDEBAR E SELEÇÃO DE ATIVO ---
ano_atual = 2026 
with st.sidebar:
    st.title("🚀 ITG Analytics")
    # NOVO: Seletor de Ativo Global
    asset_selected = st.selectbox("Ativo para Análise:", ["Bitcoin (BTC)", "Ethereum (ETH)", "S&P 500"])
    st.divider()
    st.info(f"📅 Ano Atual: {ano_atual}")
    aba = st.radio("Selecione a Análise:", 
                   ["Sazonalidade (Heatmap)", "Ciclos de Halving (BTC)", "MVRV Z-Score"])

# Carregar dados baseados na seleção
df_raw = load_data(asset_selected)

# --- ABA: SAZONALIDADE (HEATMAP DINÂMICO) ---
if aba == "Sazonalidade (Heatmap)":
    st.header(f"📅 {asset_selected} Seasonality Returns")
    
    c1, c2 = st.columns(2)
    view_mode = c1.selectbox("Frequência", ["Monthly Returns (%)", "Quarterly Returns (%)"])
    cycle_filter = c2.selectbox("Filtrar por Ciclo de Halving", ["Todos os Anos", "Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"])
    
    df_h = df_raw.copy()
    if cycle_filter != "Todos os Anos":
        years_to_keep = df_h[df_h['HalvCycle'] == cycle_filter]['Year'].unique()
        df_h = df_h[df_h['Year'].isin(years_to_keep)]

    # Lógica de Pivot e Retornos
    if "Monthly" in view_mode:
        pivot_df = df_h.groupby(['Year', 'Month'])['Price'].last().unstack()
        all_m_prices = df_raw.groupby(['Year', 'Month'])['Price'].last().unstack()
        returns_df = pivot_df.pct_change(axis=1) * 100
        # Ajuste de Janeiro vs Dezembro Anterior
        for yr in returns_df.index:
            if yr-1 in all_m_prices.index:
                returns_df.at[yr, 1] = ((all_m_prices.at[yr, 1] / all_m_prices.at[yr-1, 12]) - 1) * 100
        cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    else:
        pivot_df = df_h.groupby(['Year', 'Quarter'])['Price'].last().unstack()
        all_q_prices = df_raw.groupby(['Year', 'Quarter'])['Price'].last().unstack()
        returns_df = pivot_df.pct_change(axis=1) * 100
        for yr in returns_df.index:
            if yr-1 in all_q_prices.index:
                returns_df.at[yr, 1] = ((all_q_prices.at[yr, 1] / all_q_prices.at[yr-1, 4]) - 1) * 100
        cols = ['Q1', 'Q2', 'Q3', 'Q4']
    
    returns_df.columns = cols
    
    # Stats e Tabela (Estilo Bitbo Corrigido)
    avg, med = returns_df.mean(), returns_df.median()
    years = [str(y) for y in returns_df.index.tolist()[::-1]]
    header_vals = ["<b>Year</b>"] + list(returns_df.columns)
    
    cell_vals = [years + ["<b>Average</b>", "<b>Median</b>"]]
    cell_colors = [["#1e2127"] * (len(years) + 2)]

    for col in returns_df.columns:
        col_data = returns_df[col].tolist()[::-1]
        vals = col_data + [avg[col], med[col]]
        cell_vals.append([f"{v:+.2f}%" if pd.notnull(v) else "-" for v in vals])
        
        colors = []
        for i, v in enumerate(vals):
            if i >= len(vals) - 2: colors.append("#333a41") # Cinza para Stats
            elif pd.isnull(v): colors.append("#0e1117")
            elif v > 0: colors.append("#26a69a") # Verde Sólido
            else: colors.append("#ef5350") # Vermelho Sólido
        cell_colors.append(colors)

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_vals, fill_color='#1e2127', align='center', font=dict(color='white', size=14)),
        cells=dict(values=cell_vals, fill_color=cell_colors, align='center', font=dict(color='white', size=13), height=30)
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=450 + (len(years) * 30))
    st.plotly_chart(fig, use_container_width=True)
