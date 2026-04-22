import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="Crypto & SPX Cycle Analysis", layout="wide")

st.title("📊 Bitcoin & S&P 500: Cycle ROI Analysis")
st.markdown("Análise de Retorno sobre Investimento (ROI) baseada no Ciclo Presidencial de 4 anos.")

# --- 2. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("Configurações")
    asset_choice = st.selectbox("Selecione o Ativo", ["BTC", "S&P 500"])
    cycle_choice = st.selectbox("Selecione o Ciclo", ["Midterm Year", "Election Year", "Pre-Election Year", "Post-Election Year"])
    
    tickers = {"BTC": "BTC-USD", "S&P 500": "^GSPC"}
    hoje = datetime.now()
    ano_atual = hoje.year
    dias_decorridos = (hoje - datetime(ano_atual, 1, 1)).days
    zoom_default = min(max(dias_decorridos * 1.5, 120), 365)
    limite_x = st.slider("Ajustar Zoom (Dias)", 30, 365, int(zoom_default))

# --- 3. CARREGAMENTO DE DADOS COM CACHE ---
@st.cache_data(ttl=3600)
def load_data(ticker_name):
    df_raw = yf.download(tickers[ticker_name], start="2010-07-17", auto_adjust=True, progress=False)
    
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df = df_raw[['Close']].reset_index()
    df.columns = ['Date', 'Price']
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df[~((df['Date'].dt.month == 2) & (df['Date'].dt.day == 29))]
    
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df.groupby('Year').cumcount() + 1
    df['YearStartPrice'] = df.groupby('Year')['Price'].transform('first')
    df['ROI'] = df['Price'] / df['YearStartPrice']
    
    def get_cycle(year):
        return {0: "Election Year", 1: "Post-Election Year", 2: "Midterm Year", 3: "Pre-Election Year"}.get(year % 4)
    
    df['Cycle'] = df['Year'].apply(get_cycle)
    
    datas_hover = [(datetime(2023, 1, 1) + timedelta(days=i)).strftime('%d/%b') for i in range(366)]
    df['HoverDate'] = df['DayOfYear'].apply(lambda x: datas_hover[min(int(x)-1, 364)])
    return df

df = load_data(asset_choice)

# --- 4. CONSTRUÇÃO DO GRÁFICO ---
fig = go.Figure()
cores_lista = px.colors.qualitative.Bold

# A) Trace do Ano Atual
df_curr = df[df['Year'] == ano_atual]
if not df_curr.empty:
    fig.add_trace(go.Scatter(
        x=df_curr['DayOfYear'], y=df_curr['ROI'],
        customdata=df_curr['HoverDate'],
        hovertemplate="<b>%{customdata}</b><br>ROI: %{y:.3f}<extra></extra>",
        name=f"ATUAL {ano_atual}",
        line=dict(color='#FF0000' if asset_choice == "BTC" else '#00FF00', width=4)
    ))

# B) Histórico
df_hist = df[(df['Cycle'] == cycle_choice) & (df['Year'] < ano_atual)]
stats = df_hist.groupby('DayOfYear')['ROI'].agg(['mean', 'std']).reset_index()
stats['HoverDate'] = stats['DayOfYear'].apply(lambda x: (datetime(2023, 1, 1) + timedelta(days=int(x)-1)).strftime('%d/%b'))

fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean']+stats['std'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig.add_trace(go.Scatter(x=stats['DayOfYear'], y=stats['mean']-stats['std'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(150, 150, 150, 0.15)', name='1 SD Volatility'))
fig.add_trace(go.Scatter(
    x=stats['DayOfYear'], y=stats['mean'],
    customdata=stats['HoverDate'],
    hovertemplate="<b>%{customdata}</b> (Média Histórica)<br>ROI: %{y:.3f}<extra></extra>",
    name=f'Média {cycle_choice}', line=dict(color='white', dash='dash')
))

anos_no_ciclo = sorted(df_hist['Year'].unique())
for i, yr in enumerate(anos_no_ciclo):
    df_yr = df_hist[df_hist['Year'] == yr]
    fig.add_trace(go.Scatter(
        x=df_yr['DayOfYear'], y=df_yr['ROI'],
        customdata=df_yr['HoverDate'],
        hovertemplate=f"<b>%{{customdata}}</b> ({yr})<br>ROI: %{{y:.3f}}<extra></extra>",
        name=str(yr), line=dict(color=cores_lista[i%len(cores_lista)], width=1.2), opacity=0.3
    ))

# --- 5. ESCALA ---
vis_data = df[(df['Cycle'] == cycle_choice) & (df['DayOfYear'] <= limite_x)]
y_min, y_max = (vis_data['ROI'].min() * 0.9, vis_data['ROI'].max() * 1.1) if not vis_data.empty else (0.5, 2.0)

fig.update_layout(
    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title="Dias Decorridos", range=[0, limite_x], gridcolor='#222'),
    yaxis=dict(title="ROI Multiplier", range=[y_min, y_max], gridcolor='#222', zeroline=True, zerolinecolor='#444'),
    hovermode="x unified", height=750, margin=dict(t=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.add_hline(y=1.0, line_dash="solid", line_color="rgba(255,255,255,0.3)")

# 6. DISPLAY
st.plotly_chart(fig, use_container_width=True)
