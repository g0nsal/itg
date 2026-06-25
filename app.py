# --- ABA: RISK METRIC (BENJAMIN COWEN STYLE + DYNAMIC DCA CALCULATOR) ---
elif aba == "Risk Metric (DCA)":
    help_risk = """Métrica de risco de 0 a 1 inspirada no Into The Cryptoverse (Benjamin Cowen).  \n
Abaixo de 0.5: Zona de acumulação e compras estratégicas (DCA In).  \n
Acima de 0.5: Zona de distribuição e realização de lucros (DCA Out)."""
    st.header("📊 Asset Risk Metric & Dynamic DCA Simulator", help=help_risk)
    
    asset_name = st.selectbox("Selecione o Ativo para Análise de Risco", ["Bitcoin (BTC)", "Ethereum (ETH)"])
    df_risk = load_data(asset_name)
    
    if df_risk.empty:
        st.stop()
        
    # 1. Cálculos Matemáticos da Métrica de Risco
    df_risk['SMA_140'] = df_risk['Price'].rolling(140).mean()
    df_risk['Dev_SMA'] = np.log(df_risk['Price'] / df_risk['SMA_140'])
    
    GENESIS = pd.Timestamp("2009-01-03")
    df_risk['Time_Index'] = (df_risk['Date_Clean'] - GENESIS).dt.days + 1
    df_risk['Fair_Value_Log'] = np.log(df_risk['Time_Index']) * 1.8
    df_risk['Dev_Fair'] = np.log(df_risk['Price']) - df_risk['Fair_Value_Log']
    
    raw_risk = df_risk['Dev_SMA'].fillna(0) * 0.5 + df_risk['Dev_Fair'].fillna(0) * 0.5
    min_r = raw_risk.expanding().min()
    max_r = raw_risk.expanding().max()
    df_risk['Risk'] = (raw_risk - min_r) / (max_r - min_r)
    
    df_risk = df_risk.dropna(subset=['SMA_140']).copy()
    
    # Captura de dados atuais
    current_row = df_risk.iloc[-1]
    current_price = current_row['Price']
    current_risk = current_row['Risk']
    
    # Definição de zonas
    if current_risk >= 0.8: zone_desc, zone_color, mult = "Distribuição Máxima 🔴", "#ef4444", 0.0
    elif current_risk >= 0.6: zone_desc, zone_color, mult = "DCA Out Ativo (Vender) 🟠", "#f59e0b", 0.0
    elif current_risk >= 0.5: zone_desc, zone_color, mult = "Zona Neutra / Alerta 🟡", "#eab308", 0.5
    elif current_risk >= 0.4: zone_desc, zone_color, mult = "Zona Neutra / Acumulação ⚪", "#94a3b8", 1.0
    elif current_risk >= 0.2: zone_desc, zone_color, mult = "DCA In Ativo (Comprar) 🟢", "#10b981", 1.5
    else: zone_desc, zone_color, mult = "Acumulação Pesada 💎", "#059669", 2.5
        
    # Cards de Contexto Visual
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        st.markdown(f'<div class="stat-card">Preço Atual ({asset_name})<div class="stat-val">${current_price:,.2f}</div></div>', unsafe_allow_html=True)
    with cd2:
        st.markdown(f'<div class="stat-card">Métrica de Risco<div class="stat-val" style="color: {zone_color};">{current_risk:.4f}</div></div>', unsafe_allow_html=True)
    with cd3:
        st.markdown(f'<div class="stat-card">Estado do Mercado<div class="stat-val" style="color: {zone_color};">{zone_desc}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # --- NOVO: CALCULADORA DYNAMIC DCA ---
    st.subheader("🧮 Simulador de Compras Dinâmicas (Budget Allocation)")
    
    cc1, cc2 = st.columns(2)
    budget_total = cc1.number_input("Budget Inicial Total (€)", min_value=100, max_value=1000000, value=5000, step=500)
    total_parts = cc2.number_input("Número de Parcelas Desejadas (Semanas/Meses)", min_value=4, max_value=104, value=20, step=1)
    
    base_tranche = budget_total / total_parts
    dynamic_buy = base_tranche * mult
    
    st.markdown(f"""
    <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
        <h4>📋 Plano de Ação para este Período:</h4>
        <ul>
            <li>O teu investimento base padrão por parcela é de: <b>{base_tranche:,.2f} €</b></li>
            <li>Multiplicador atual aplicado pelo Risco (x{mult}): <b style="color:{zone_color};">{mult}x</b></li>
            <li><b>Valor a investir AGORA: <span style="font-size:20px; color:#00FFA3;">{dynamic_buy:,.2f} €</span></b></li>
        </ul>
        <p style="font-size: 13px; color: #94a3b8; margin-top: 10px;">
            💡 <i>Nota: Se o risco subir acima de 0.6, a calculadora sugere 0.00 € de compra, indicando que deves guardar o caixa ou começar a realizar lucros de forma fracionada.</i>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico Estatístico Principal
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_risk['Date_Clean'], y=df_risk['Price'], name="Preço (USD)",
                             line=dict(color='rgba(255,255,255,0.15)', width=1), yaxis="y2"))
    
    fig.add_trace(go.Scatter(x=df_risk['Date_Clean'], y=df_risk['Risk'], name="Risk Metric",
                             line=dict(color='#3b82f6', width=2), yaxis="y1"))
    
    fig.add_hrect(y0=0.0, y1=0.2, fillcolor="green", opacity=0.12, layer="below", line_width=0)
    fig.add_hrect(y0=0.2, y1=0.4, fillcolor="green", opacity=0.04, layer="below", line_width=0)
    fig.add_hrect(y0=0.6, y1=0.8, fillcolor="red", opacity=0.04, layer="below", line_width=0)
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor="red", opacity=0.12, layer="below", line_width=0)
    
    fig.update_layout(
        template="plotly_dark", height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Risk Level (0 - 1)", side="right", range=[0, 1], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis2=dict(type="log", overlaying="y", side="left", showgrid=False, title="Preço (USD)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
