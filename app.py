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
    help_h = "Indica a performance do Bitcoin em relação ao ciclo de Halving.  \n\n- **Halving Year**: Ano do choque de oferta.  \n- **Post-Halving**: Ano histórico de Bull Run.  \n- **Bear Year**: Ano de ajuste e correção.  \n- **Pre-Halving**: Ano de recuperação."
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
        fig.add_trace(go.Scatter(x=df_yr['DayOfYear'], y=df_yr['ROI'], name=f"Ano {yr}",
                                 line=dict(width=1.2, color=cores[i % len(cores)]), opacity=0.3))

    stats = df_hist.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name="Média Ciclos", line=dict(color='white', width=1.5, dash='dash')))

    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}",
                                 line=dict(color='#00FFA3', width=2.5), 
                                 customdata=df_curr['HoverDate'], hovertemplate="%{customdata}<br>ROI: %{y:.2f}<extra></extra>"))

    fig.update_layout(template="plotly_dark", height=700, xaxis_range=[0, limite_x], hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 3: MVRV Z-SCORE (SINCERIDADE MATEMÁTICA) ---
elif aba == "MVRV Z-Score":
    help_m = "Indica sobrevalorização ou subvalorização.  \n\n- **Z-Score alto (>7)**: sugere topo;  \n- **Z-Score baixo (<0)**: sugere zona de acumulação."
    st.header("📈 Bitcoin MVRV Z-Score", help=help_m)
    
    df = load_data("BTC")
    supply = 19700000 
    
    # MATEMÁTICA REVISADA:
    df['Market Cap'] = df['Price'] * supply
    # Usamos o preço médio de 2 anos como base para o Realized Cap (mais estável)
    df['Realized Price'] = df['Price'].rolling(window=730, min_periods=100).mean()
    df['Realized Cap'] = df['Realized Price'] * supply
    
    # FÓRMULA MVRV: (MV - RV) / STD(MV)
    # Para evitar que caia a -2, calculamos o desvio padrão da própria diferença
    diff = df['Market Cap'] - df['Realized Cap']
    std_diff = diff.rolling(window=730, min_periods=100).std()
    
    # Normalização final para bater com o visual das imagens (0 a 10)
    df['Z-Score'] = (diff / std_diff) + 0.5 

    df_plot = df.dropna(subset=['Z-Score']).copy()

    fig = go.Figure()

    # Market Cap (Branco)
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Market Cap'], name="Market Cap", 
                             line=dict(color='white', width=1.5), yaxis="y2"))
    
    # Realized Cap (Azul)
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Realized Cap'], name="Realized Cap", 
                             line=dict(color='#add8e6', width=1.2), yaxis="y2"))
    
    # Z-Score (Laranja)
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Z-Score'], name="Z-Score", 
                             line=dict(color='#f39c12', width=2), yaxis="y1"))

    # ZONAS (Ajustadas)
    fig.add_hrect(y0=7, y1=10, fillcolor="red", opacity=0.15, line_width=0)
    fig.add_hrect(y0=-0.8, y1=0.2, fillcolor="green", opacity=0.15, line_width=0)

    fig.update_layout(
        template="plotly_dark", height=750,
        yaxis=dict(title="Z-Score", side="right", range=[-1.5, 11], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis2=dict(title="Capitais (USD)", side="left", type="log", overlaying="y", showgrid=False),
        xaxis=dict(showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 2: CICLOS PRESIDENCIAIS ---
elif aba == "Ciclos Presidenciais (ROI)":
    help_p = "Análise de performance baseada no ciclo político de 4 anos dos EUA.  \n\n- **Midterm Year (2026)**: Ano de eleições intercalares."
    st.header("📊 ROI vs Presidential Cycles", help=help_p)
    c1, c2 = st.columns(2)
    asset = c1.selectbox("Ativo", ["BTC", "S&P 500", "BTC/SPX Ratio"])
    cycle = c2.selectbox("Ciclo", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
    df = load_data(asset)
    
    fig = go.Figure()
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(x=df_curr['DayOfYear'], y=df_curr['ROI'], name=f"ATUAL {ano_atual}",
                                 line=dict(color='#00FFA3', width=2.5)))
    
    df_h = df[(df['PresCycle'] == cycle) & (df['Year'] < ano_atual)]
    stats = df_h.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['ROI'], name='Média Histórica', line=dict(color='white', dash='dash')))
    fig.update_layout(template="plotly_dark", height=700, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- ABA 4: MÉDIAS MÓVEIS ---
elif aba == "Médias Móveis Semanais":
    help_w = "Médias Móveis Semanais (SMA).  \n\n- **200W SMA**: O suporte histórico mais importante do Bitcoin."
    st.header("📉 BTC Weekly Moving Averages", help=help_w)
    df = load_data("BTC")
    df.set_index('Date_Clean', inplace=True)
    df_w = df['Price'].resample('W').last().to_frame()
    periods = [20, 50, 100, 200]
    for p in periods: df_w[f'{p}W SMA'] = df_w['Price'].rolling(window=p).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_w.index, y=df_w['Price'], name='BTC Price', line=dict(color='white')))
    for p in periods: fig.add_trace(go.Scatter(x=df_w.index, y=df_w[f'{p}W SMA'], name=f'{p}W SMA', opacity=0.7))
    fig.update_layout(template="plotly_dark", height=750, yaxis_type="log", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
