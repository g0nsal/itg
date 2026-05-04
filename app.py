import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURAÇÃO E ESTILO DARK MODERNO ---
st.set_page_config(page_title="ITG Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        /* Forçar Dark Mode em todos os elementos do Streamlit */
        [data-testid="stAppViewContainer"], .main, .stApp {
            background-color: #0f172a !important; /* Slate 900 */
            color: #f8fafc !important;
        }
        
        /* Estilo da Sidebar */
        [data-testid="stSidebar"] {
            background-color: #1e293b !important; /* Slate 800 */
            border-right: 1px solid #334155;
        }

        /* Cartões de Estatísticas Estilo Stripe Dark */
        .stat-card {
            background-color: #1e293b;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #334155;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
        }

        /* Ajuste global de fontes */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Corrigir visibilidade de inputs e selects */
        .stSelectbox label, .stRadio label {
            color: #94a3b8 !important;
            font-weight: 600;
        }
        
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. CACHE DE DADOS ---
@st.cache_data(ttl=3600)
def load_raw_data(ticker, start_date="2010-07-17"):
    raw = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw[['Close']].reset_index()
    df.columns = ['Date', 'Price']
    df['Date_Clean'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['Year'] = df['Date_Clean'].dt.year
    df['Month'] = df['Date_Clean'].dt.month
    df['DayOfYear'] = df.groupby('Year').cumcount() + 1
    
    # Ciclos simplificados
    df['HalvCycle'] = df['Year'].apply(lambda y: {0:"Halving Year", 1:"Post-Halving Year", 2:"Bear Year", 3:"Pre-Halving Year"}.get(y % 4))
    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    return df

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#3b82f6; font-size: 22px;'>ITG ANALYTICS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size: 13px; margin-bottom: 20px;'>Intelligence Terminal</p>", unsafe_allow_html=True)
    
    aba = st.radio("MENU", ["Sazonalidade", "Ciclos de Mercado", "MVRV Z-Score"])
    
    st.markdown("---")
    st.markdown("""
        <div class="stat-card">
            <p style="color:#94a3b8; font-size:11px; font-weight:bold;">MARKET STATUS 2026</p>
            <p style="color:#22c55e; font-size:14px; margin:5px 0;">● Bear Year Phase</p>
            <p style="color:#f8fafc; font-size:12px;">Next Halving: 2028</p>
        </div>
    """, unsafe_allow_html=True)

# --- CONTEÚDO PRINCIPAL ---
st.markdown(f"<h2 style='margin-bottom:30px;'>{aba}</h2>", unsafe_allow_html=True)

if aba == "Sazonalidade":
    c1, c2 = st.columns(2)
    asset = c1.selectbox("Selecione o Ativo", ["Bitcoin (BTC)", "Ethereum (ETH)", "S&P 500"])
    
    # Simulação de tabela com Plotly para evitar o fundo branco padrão do Streamlit
    df = load_raw_data("BTC-USD" if "Bitcoin" in asset else "ETH-USD")
    
    # Criar Tabela Estilo Dark
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            fill_color='#1e293b',
            align='center',
            font=dict(color='white', size=12),
            line_color='#334155'
        ),
        cells=dict(
            values=[[2026, 2025, 2024, 2023], ["+5%", "-2%", "+10%", "-1%"]*3], # Exemplo simplificado
            fill_color='#0f172a',
            align='center',
            font=dict(color='#cbd5e1', size=12),
            line_color='#334155',
            height=30
        )
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

elif aba == "Ciclos de Mercado":
    df = load_raw_data("BTC-USD")
    
    fig = go.Figure()
    # Linha principal com brilho (Estilo Apple/Neon)
    fig.add_trace(go.Scatter(
        x=df['DayOfYear'].tail(365), 
        y=df['ROI'].tail(365),
        name="2026",
        line=dict(color='#3b82f6', width=3),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        xaxis=dict(showgrid=True, gridcolor='#1e293b', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#1e293b', zeroline=False),
        margin=dict(l=0, r=0, t=20, b=0),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
