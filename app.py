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
                   ["Sazonalidade (Heatmap)", "Ciclos de Halving (BTC)", "Ciclos Presidenciais (ROI)", "MVRV Z-Score", "Médias Móveis Semanais"])

# --- ABA 0: SAZONALIDADE (HEATMAP) ---
if aba == "Sazonalidade (Heatmap)":
    help_heat = "Heatmap de retornos percentuais.  \nAs linhas finais mostram a Média e Mediana histórica para identificar tendências sazonais (ex: Uptober)."
    st.header("📅 Bitcoin Seasonality Heatmap", help=help_heat)
    
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
        jan_prices = df_h.groupby(['Year', 'Month'])['Price'].last().unstack()[1]
        dec_prices_prev = df_h.groupby(['Year', 'Month'])['Price'].last().unstack()[12].shift(1)
        pivot_df[1] = ((jan_prices / dec_prices_prev) - 1) * 100
        pivot_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    else:
        df_h['Quarter'] = df_h['Date_Clean'].dt.quarter
        pivot_df = df_h.groupby(['Year', 'Quarter'])['Price'].last().unstack()
        pivot_df = pivot_df.pct_change(axis=1) * 100
        q1_prices = df_h.groupby(['Year', 'Quarter'])['Price'].last().unstack()[1]
        q4_prices_prev = df_h.groupby(['Year', 'Quarter'])['Price'].last().unstack()[4].shift(1)
        pivot_df[1] = ((q1_prices / q4_prices_prev) - 1) * 100
        pivot_df.columns = ['Q1', 'Q2', 'Q3', 'Q4']

    # CÁLCULO DE MÉDIA E MEDIANA
    stats_mean = pivot_df.mean().to_frame(name='AVERAGE').T
    stats_median = pivot_df.median().to_frame(name='MEDIAN').T
    
    # Unir à tabela principal
    pivot_df = pivot_df.sort_index(ascending=False)
    final_df = pd.concat([pivot_df, stats_mean, stats_median])

    fig = px.imshow(final_df, 
                    text_auto=".1f", 
                    color_continuous_scale=[[0, 'red'], [0.5, '#0e1117'], [1, 'green']],
                    color_continuous_midpoint=0,
                    aspect="auto")
    
    fig.update_layout(height=700, template="plotly_dark", coloraxis_showscale=False,
                      yaxis=dict(tickmode='array', tickvals=final_df.index, ticktext=final_df.index))
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 1: CICLOS DE HALVING ---
elif aba == "Ciclos de Halving (BTC)":
    help_h = "Comparativo de ROI YTD baseado nos ciclos de 4 anos do Bitcoin."
    st.header(f"₿ BTC: Comparativo de Ciclos de Halving", help=help_h)
    fase = st.selectbox("Comparar anos de:", ["Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"], index=2)
    df = load_data("BTC")
    limite_x = st.slider("Zoom X (Dias)", 30, 365, 365)
    fig = go.Figure()
    df_hist = df[(df['HalvCycle'] == fase) & (df['Year'] < ano_atual)]
    anos_na_fase = sorted(df_hist['Year'].unique())
    cores = px.colors.qualitative.Pastel
    for i, yr in enumerate(anos_na_fase):
        df_yr = df_hist[df_hist['Year'] == yr]
        fig.add_trace(go.Scatter(x=df_yr['DayOfYear'], y=df_yr['ROI'], name=f"Ano {yr}", line=dict(width=1.2, color=cores[i % len(cores)]), opacity=0.3))
    stats = df_hist.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name="Média Ciclos", line=dict(color='white', width=1.5, dash='dash')))
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}", line=dict(color='#00FFA3', width=2.5), customdata=df_curr['HoverDate'], hovertemplate="%{customdata}<br>ROI: %{y:.2f}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=700, xaxis_range=[0, limite_x], hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 2: CICLOS PRESIDENCIAIS ---
elif aba == "Ciclos Presidenciais (ROI)":
    st.header(f"📊 ROI vs Presidential Cycles")
    c1, c2 = st.columns(2)
    asset = c1.selectbox("Ativo", ["BTC", "S&P 500", "BTC/SPX Ratio"])
    cycle = c2.selectbox("Ciclo", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
    df = load_data(asset)
    fig = go.Figure()
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}", line=dict(color='#00FFA3' if asset != "S&P 500" else '#00FF00', width=2.5)))
    df_h = df[(df['PresCycle'] == cycle) & (df['Year'] < ano_atual)]
    stats = df_h.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name='Média Histórica', line=dict(color='white', dash='dash')))
    fig.update_layout(template="plotly_dark", height=700, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 3: MVRV Z-SCORE ---
elif aba == "MVRV Z-Score":
    st.header("📈 Bitcoin MVRV Z-Score")
    df = load_data("BTC")
    supply = 19750000 
    df['MC'] = df['Price'] * supply
    df['RC'] = (df['Price'].rolling(365).mean()) * supply
    df['Z'] = (df['MC'] - df['RC']) / (df['MC'].rolling(365).std())
    df['Z_Calib'] = (df['Z'] * 2.5) + 0.5 
    df_plot = df.dropna(subset=['Z_Calib'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['MC'], name="Market Cap", line=dict(color='white', width=1.5), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['RC'], name="Realized Cap", line=dict(color='rgba(173, 216, 230, 0.6)', width=1.2, dash='dot'), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Z_Calib'], name="Z-Score", line=dict(color='#f39c12', width=1.8), yaxis="y1"))
    fig.add_hrect(y0=7, y1=10, fillcolor="red", opacity=0.15, line_width=0)
    fig.add_hrect(y0=-0.5, y1=0.2, fillcolor="green", opacity=0.15, line_width=0)
    fig.update_layout(template="plotly_dark", height=750, yaxis=dict(title="Z-Score", side="right", range=[-1.5, 11]), yaxis2=dict(title="Market Cap (USD)", side="left", type="log", overlaying="y", showgrid=False), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 4: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis Semanais":
    st.header("📉 BTC Weekly Moving Averages")
    df = load_data("BTC")
    df.set_index('Date_Clean', inplace=True)
    df_w = df['Price'].resample('W').last().to_frame()
    for p in [20, 50, 100, 200]: df_w[f'{p}W SMA'] = df_w['Price'].rolling(window=p).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name='BTC Price', line=dict(color='white', width=2)))
    for p in [20, 50, 100, 200]: fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f'{p}W SMA'], name=f'{p}W SMA', opacity=0.7))
    fig.update_layout(template="plotly_dark", height=750, yaxis_type="log", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
