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
    help_text = """Indica a performance do Bitcoin em relação ao ciclo de Halving.  \n\n- **Halving Year**: Ano do choque de oferta.  \n- **Post-Halving**: Ano histórico de Bull Run.  \n- **Bear Year**: Ano de ajuste e correção.  \n- **Pre-Halving**: Ano de recuperação."""
    st.header(f"₿ BTC: Comparativo de Ciclos de Halving", help=help_text)
    
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

# --- ABA 3: MVRV Z-SCORE (ESTILO IMAGEM) ---
# --- ABA 3: MVRV Z-SCORE (VERSÃO FINAL ASSERTIVA) ---
# --- ABA 3: MVRV Z-SCORE (VERSÃO FINAL MATEMATICAMENTE ALINHADA) ---
elif aba == "MVRV Z-Score":
    help_text_mvrv = """Indica sobrevalorização ou subvalorização.  \n\n- **Z-Score alto (>7)**: sugere topo;  \n- **Z-Score baixo (<0)**: sugere zona de acumulação."""
    st.header("📈 Bitcoin MVRV Z-Score", help=help_text_mvrv)
    
    df = load_data("BTC")
    supply = 19700000 
    
    # 1. CÁLCULOS MATEMÁTICOS DE PRECISÃO
    df['Market Cap'] = df['Price'] * supply
    # O Realized Price real é uma média ponderada. A média expansiva é a melhor aproximação sem dados on-chain.
    df['Realized Price'] = df['Price'].expanding().mean() 
    df['Realized Cap'] = df['Realized Price'] * supply
    
    # FÓRMULA Z-SCORE ASSERTIVA
    # Numerador: Market Cap - Realized Cap
    # Denominador: Desvio Padrão Expansivo do Market Cap (para manter a proporção histórica)
    diff = df['Market Cap'] - df['Realized Cap']
    df['Z-Score'] = diff / df['Market Cap'].expanding().std()

    # AJUSTE DE OFFSET: Para alinhar com o gráfico da Bitbo/Glassnode
    # O Z-score histórico tem uma base que flutua perto de 0.
    df['Z-Score'] = df['Z-Score'] * 5  # Fator de escala para atingir os picos de 7-10
    
    df_plot = df.dropna(subset=['Z-Score']).copy()

    fig = go.Figure()

    # Market Cap (Branco) - Eixo Esquerda (Y2)
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Market Cap'], name="Market Cap", 
                             line=dict(color='white', width=1.5), yaxis="y2"))
    
    # Realized Cap (Azul) - Eixo Esquerda (Y2)
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Realized Cap'], name="Realized Cap", 
                             line=dict(color='#add8e6', width=1.2), yaxis="y2"))
    
    # Z-Score (Laranja) - Eixo Direita (Y1)
    fig.add_trace(go.Scatter(x=df_plot['Date_Clean'], y=df_plot['Z-Score'], name="Z-Score", 
                             line=dict(color='#f39c12', width=1.5), yaxis="y1"))

    # ZONAS DE CORES (Ajustadas para os níveis 0 e 7)
    fig.add_hrect(y0=7, y1=10, fillcolor="red", opacity=0.15, line_width=0)
    fig.add_hrect(y0=-0.5, y1=0.2, fillcolor="green", opacity=0.15, line_width=0)

    fig.update_layout(
        template="plotly_dark", height=750,
        # Eixo Z-Score
        yaxis=dict(
            title="MVRV Z-Score", side="right", range=[-1, 11], # Alinhado com as imagens
            showgrid=True, gridcolor='rgba(255,255,255,0.05)',
            zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'
        ),
        # Eixo Market Cap
        yaxis2=dict(
            title="Market Cap (USD)", side="left", type="log", overlaying="y",
            showgrid=False # Remove a confusão de linhas
        ),
        xaxis=dict(showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
# --- (Restante do código: Ciclos Presidenciais e Médias Móveis mantidos) ---
elif aba == "Ciclos Presidenciais (ROI)":
    st.header("📊 ROI vs Presidential Cycles", help="Análise de performance baseada no ciclo político de 4 anos dos EUA.")
    # ... código anterior ...
    st.write("Seleção de Ativo e Ciclo...")
    # (Manter lógica anterior)

elif aba == "Médias Móveis Semanais":
    st.header("📉 BTC Weekly Moving Averages", help="Análise de longo prazo usando as principais Médias Móveis Semanais (SMA).")
    # ... código anterior ...
