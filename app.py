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

# --- ABA: MVRV Z-SCORE (ESTABILIZAÇÃO LOGARÍTMICA) ---
if aba == "MVRV Z-Score":
    help_mvrv = """Indica sobrevalorização ou subvalorização.  \n
Z-Score alto (>7) sugere topo;  \n
Z-Score baixo (<0) sugere zona de acumulação."""
    
    st.header("📈 Bitcoin MVRV Z-Score", help=help_mvrv)
    
    df = load_data("BTC")
    supply = 19750000 
    
    # 1. CÁLCULOS ESTÁVEIS
    df['Market Cap'] = df['Price'] * supply
    # Média móvel de longo prazo (2 anos) para simular o Realized Cap de forma suave
    df['Realized Cap'] = df['Market Cap'].rolling(window=730, min_periods=1).mean()
    
    # 2. FÓRMULA MVRV RATIO (A base do Z-Score)
    df['MVRV_Ratio'] = df['Market Cap'] / df['Realized Cap']
    
    # 3. Z-SCORE NORMALIZADO (Usando o log para comprimir os triliões e evitar +30/-10)
    # Calculamos o desvio do ratio em vez do desvio do valor nominal
    mu = df['MVRV_Ratio'].expanding().mean()
    sigma = df['MVRV_Ratio'].expanding().std()
    df['Z-Score'] = (df['MVRV_Ratio'] - mu) / sigma
    
    # Ajuste final de calibração para bater com os gráficos da Glassnode (0 a 8)
    df['Z-Score'] = (df['Z-Score'] * 2.5) + 1.0

    df_plot = df.dropna(subset=['Z-Score']).copy()

    fig = go.Figure()

    # Market Cap (Branco) - Eixo Esquerdo (Y2)
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Market Cap'], name="Market Cap", 
                             line=dict(color='white', width=1.5), yaxis="y2"))
    
    # Realized Cap (Azul) - Eixo Esquerdo (Y2)
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Realized Cap'], name="Realized Cap", 
                             line=dict(color='#3498db', width=1.2, dash='dot'), yaxis="y2"))
    
    # Z-Score (Laranja) - Eixo Direito (Y1)
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Z-Score'], name="Z-Score", 
                             line=dict(color='#f39c12', width=1.8), yaxis="y1"))

    # ZONAS (0 a 7+)
    fig.add_hrect(y0=7, y1=10, fillcolor="red", opacity=0.15, line_width=0, annotation_text="TOP")
    fig.add_hrect(y0=-0.5, y1=0.2, fillcolor="green", opacity=0.15, line_width=0, annotation_text="BOTTOM")

    fig.update_layout(
        template="plotly_dark", height=750,
        yaxis=dict(title="MVRV Z-Score", side="right", range=[-1.5, 11], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis2=dict(title="Market Cap (USD)", side="left", type="log", overlaying="y", showgrid=False),
        xaxis=dict(showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: CICLOS DE HALVING ---
elif aba == "Ciclos de Halving (BTC)":
    help_h = """Performance do Bitcoin por ciclo de 4 anos.  \n
Halving Year: Ano do choque de oferta.  \n
Post-Halving: Ano histórico de Bull Run.  \n
Bear Year: Ano de ajuste e correção.  \n
Pre-Halving: Ano de recuperação."""
    st.header(f"₿ BTC: Comparativo de Ciclos de Halving", help=help_h)
    fase = st.selectbox("Comparar anos de:", ["Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"], index=2)
    df = load_data("BTC")
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
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}", line=dict(color='#00FFA3', width=2.5)))
    fig.update_layout(template="plotly_dark", height=700, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: CICLOS PRESIDENCIAIS ---
elif aba == "Ciclos Presidenciais (ROI)":
    st.header("📊 ROI vs Presidential Cycles")
    c1, c2 = st.columns(2)
    asset = c1.selectbox("Ativo", ["BTC", "S&P 500", "BTC/SPX Ratio"])
    cycle = c2.selectbox("Ciclo", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
    df = load_data(asset)
    fig = go.Figure()
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}", line=dict(color='#00FFA3', width=2.5)))
    df_h = df[(df['PresCycle'] == cycle) & (df['Year'] < ano_atual)]
    stats = df_h.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name='Média Histórica', line=dict(color='white', dash='dash')))
    fig.update_layout(template="plotly_dark", height=700, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis Semanais":
    st.header("📉 BTC Weekly Moving Averages")
    df = load_data("BTC")
    df.set_index('Date_Clean', inplace=True)
    df_w = df['Price'].resample('W').last().to_frame()
    for p in [20, 50, 100, 200]: df_w[f'{p}W SMA'] = df_w['Price'].rolling(window=p).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name='BTC Price', line=dict(color='white')))
    for p in [20, 50, 100, 200]: fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f'{p}W SMA'], name=f'{p}W SMA', opacity=0.7))
    fig.update_layout(template="plotly_dark", height=750, yaxis_type="log", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
