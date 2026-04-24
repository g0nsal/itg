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
    
    # Lógicas de Ciclo
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
    st.markdown(f"""
    **Fase do Mercado:**
    - 🇺🇸 Presidencial: `Midterm`
    - ₿ Halving: `Bear Year`
    """)
    st.write("---")
    aba = st.radio("Selecione a Análise:", 
                   ["Ciclos de Halving (BTC)", "Ciclos Presidenciais (ROI)", "MVRV Z-Score", "Médias Móveis Semanais"])

# --- ABA: CICLOS DE HALVING ---
if aba == "Ciclos de Halving (BTC)":
    st.header(f"₿ BTC: Comparativo de Ciclos de Halving")
    st.info("Aqui comparas o ano atual com os mesmos períodos de ciclos passados (ex: 2026 vs 2022, 2018, 2014).")
    
    fase = st.selectbox("Comparar anos de:", ["Halving Year", "Post-Halving Year", "Bear Year", "Pre-Halving Year"], index=2)
    
    df = load_data("BTC")
    limite_x = st.slider("Ver até que dia do ano?", 30, 365, 365)

    fig = go.Figure()
    
    # 1. Anos Históricos (Linhas Finas)
    df_hist = df[(df['HalvCycle'] == fase) & (df['Year'] < ano_atual)]
    anos_na_fase = sorted(df_hist['Year'].unique())
    
    cores = px.colors.qualitative.Pastel
    for i, yr in enumerate(anos_na_fase):
        df_yr = df_hist[df_hist['Year'] == yr]
        fig.add_trace(go.Scatter(
            x=df_yr['DayOfYear'], y=df_yr['ROI'],
            name=f"Ano {yr}",
            line=dict(width=1.5, color=cores[i % len(cores)]),
            opacity=0.4,
            customdata=df_yr['HoverDate'],
            hovertemplate=f"<b>Ano {yr}</b><br>%{{customdata}}<br>ROI: %{{y:.2f}}<extra></extra>"
        ))

    # 2. Média Histórica (Linha Tracejada)
    stats = df_hist.groupby('DayOfYear')['ROI'].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=stats['DayOfYear'], y=stats['ROI'],
        name="Média dos Ciclos",
        line=dict(color='white', width=2, dash='dash')
    ))

    # 3. Ano Atual (Linha de Destaque)
    df_curr = df[df['Year'] == ano_atual]
    if not df_curr.empty:
        fig.add_trace(go.Scatter(
            x=df_curr['DayOfYear'], y=df_curr['ROI'],
            name=f"ATUAL {ano_atual}",
            line=dict(color='#00FFA3', width=5),
            customdata=df_curr['HoverDate'],
            hovertemplate=f"<b>ATUAL {ano_atual}</b><br>%{{customdata}}<br>ROI: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_dark", height=700,
        xaxis=dict(title="Dia do Ano", range=[0, limite_x]),
        yaxis=dict(title="Multiplicador de ROI (Início do Ano = 1.0)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.add_hline(y=1.0, line_color="gray", line_dash="dot")
    st.plotly_chart(fig, use_container_width=True)

# --- (As outras abas continuam aqui de forma similar ao código anterior) ---
# Se precisares do código das outras abas completo no mesmo ficheiro, diz-me.
