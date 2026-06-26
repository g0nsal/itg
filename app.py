# --- ABA 3: RISK METRIC (TRUE COWEN MATRIX - CORRIGIDA) ---
elif aba == "Risk Metric (DCA)":
    help_risk = """Métrica de risco de 0 a 1 inspirada no Into The Cryptoverse (Benjamin Cowen).  \n
Abaixo de 0.3: Zona de Acumulação e Dynamic DCA In.  \n
Entre 0.3 e 0.6: Zona Cinzenta (Banda de Inação de Longo Prazo).  \n
Acima de 0.6: Distribuição Fracionada em 15 avos (DCA Out)."""
    st.header("📊 Into The Cryptoverse Risk Metric & Dynamic DCA", help=help_risk)
    
    asset_name = st.selectbox("Selecione o Ativo para Análise de Risco", ["Bitcoin (BTC)", "Ethereum (ETH)"])
    df_risk = load_data(asset_name)
    
    if df_risk.empty:
        st.stop()
        
    # 1. CÁLCULO DA ENGINE QUANT
    df_risk['SMA_140'] = df_risk['Price'].rolling(140).mean()
    df_risk['Dev_SMA'] = np.log(df_risk['Price'] / df_risk['SMA_140'])
    
    GENESIS = pd.Timestamp("2009-01-03")
    df_risk['Time_Index'] = (df_risk['Date_Clean'] - GENESIS).dt.days + 1
    df_risk['Fair_Value_Log'] = np.log(df_risk['Time_Index']) * 1.8
    df_risk['Dev_Fair'] = np.log(df_risk['Price']) - df_risk['Fair_Value_Log']
    
    raw_risk = df_risk['Dev_SMA'].fillna(0) * 0.4 + df_risk['Dev_Fair'].fillna(0) * 0.6
    min_r = raw_risk.expanding().min()
    max_r = raw_risk.expanding().max()
    df_risk['Risk'] = ((raw_risk - min_r) / (max_r - min_r)) * 0.9 + 0.05
    
    df_risk = df_risk.dropna(subset=['SMA_140']).copy()
    
    current_row = df_risk.iloc[-1]
    current_price = current_row['Price']
    current_risk = current_row['Risk']
    
    # 2. SISTEMA DE REGRAS REAIS DO BENJAMIM COWEN (BIFÁSICO: COMPRA / VENDA / HOLD)
    mode = "HOLD"
    fraction_text = "0/15"
    mult = 0.0
    
    if current_risk >= 0.9:
        zone_desc, zone_color, mode, fraction_text = "Distribuição Crítica (Vender 1/3) 🔴", "#ef4444", "SELL", "5/15"
    elif current_risk >= 0.8:
        zone_desc, zone_color, mode, fraction_text = "DCA Out Ativo (Forte Sobreaquecimento) 🔴", "#f87171", "SELL", "4/15"
    elif current_risk >= 0.7:
        zone_desc, zone_color, mode, fraction_text = "DCA Out Ativo (Distribuição) 🟠", "#f59e0b", "SELL", "3/15"
    elif current_risk >= 0.6:
        zone_desc, zone_color, mode, fraction_text = "DCA Out Inicial (Realização de Lucro) 🟠", "#fb923c", "SELL", "2/15"
    elif current_risk >= 0.3:
        zone_desc, zone_color, mode, mult = "Zona Cinzenta (Aguardar / No-Trade) ⚪", "#94a3b8", "HOLD", 0.0
    elif current_risk >= 0.2:
        zone_desc, zone_color, mode, mult = "DCA In Ativo (Comprar Baixo) 🟢", "#4ade80", "BUY", 1.0
    elif current_risk >= 0.1:
        zone_desc, zone_color, mode, mult = "DCA In Intenso (Comprar Forte) 🟢", "#10b981", "BUY", 2.5
    else:
        zone_desc, zone_color, mode, mult = "Capitulação / Fundo de Ciclo Extremo 💎", "#059669", "BUY", 4.0
        
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        st.markdown(f'<div class="stat-card">Preço Atual ({asset_name})<div class="stat-val">${current_price:,.2f}</div></div>', unsafe_allow_html=True)
    with cd2:
        st.markdown(f'<div class="stat-card">Métrica de Risco (ITC)<div class="stat-val" style="color: {zone_color};">{current_risk:.4f}</div></div>', unsafe_allow_html=True)
    with cd3:
        st.markdown(f'<div class="stat-card">Estratégia do Ciclo<div class="stat-val" style="color: {zone_color};">{zone_desc}</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 3. INTERFACE DINÂMICA E ADAPTATIVA (BIFÁSICA)
    st.subheader("🧮 Calculadora de DCA Dinâmico (Orçamento Estratégico)")
    
    if mode == "BUY":
        cc1, cc2 = st.columns(2)
        budget_total = cc1.number_input("Budget Total Disponível para Acumular (€)", min_value=100, max_value=1000000, value=5000, step=500)
        total_parts = cc2.number_input("Tranches Planeadas (Semanas/Meses)", min_value=4, max_value=104, value=20, step=1)
        
        base_tranche = budget_total / total_parts
        dynamic_buy = base_tranche * mult
        
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
            <h4 style="color:#4ade80;">🛒 Modo Acumulação Ativo (DCA In):</h4>
            <ul>
                <li>Investimento Semanal Base Padrão: <b>{base_tranche:,.2f} €</b></li>
                <li>Multiplicador de Risco Atual: <b>{mult}x</b></li>
                <li><b>Montante Líquido a Comprar HOJE: <span style="font-size:22px; color:#00FFA3;">{dynamic_buy:,.2f} €</span></b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif mode == "SELL":
        cc1, cc2 = st.columns(2)
        portfolio_value = cc1.number_input("Valor Atual do teu Portfólio neste Ativo (€)", min_value=100, max_value=10000000, value=10000, step=1000)
        
        # O Ben vende frações do portfólio total (ex: 3/15 significam 20% da posição atual)
        num, den = map(int, fraction_text.split('/'))
        amount_to_sell = portfolio_value * (num / den)
        
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
            <h4 style="color:#f87171;">💰 Modo Realização de Lucro Ativo (DCA Out):</h4>
            <ul>
                <li>Fração de Custódia a Desfazer (Regra dos 15avos): <b style="color:{zone_color};">{fraction_text}</b></li>
                <li><b>Montante Líquido a Vender/Garantir HOJE: <span style="font-size:22px; color:#ff4d4d;">{amount_to_sell:,.2f} €</span></b></li>
            </ul>
            <p style="font-size: 13px; color: #94a3b8; margin-top: 10px;">
                <i>⚠️ Nota: Esta liquidez gerada deve ser guardada em stablecoins ou fiat para ser reinstalada no mercado quando o risco regressar abaixo de 0.3 (estimado para a janela macro de Outubro).</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    else: # HOLD / ZONA CINZENTA
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 25px;">
            <h4>⚪ Modo de Inação Ativo (Grey Region):</h4>
            <p style="font-size: 18px; color: #94a3b8;">
                <b>Hold Estrito</b>. O risco atual não justifica compras nem vendas estratégicas.
            </p>
            <ul>
                <li>Montante a Comprar: <b>0.00 €</b></li>
                <li>Montante a Vender: <b>0.00 €</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    # Desenho do Gráfico Histórico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_risk['Date_Clean'], y=df_risk['Price'], name="Preço (USD)", line=dict(color='rgba(255,255,255,0.15)', width=1), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_risk['Date_Clean'], y=df_risk['Risk'], name="Risk Metric", line=dict(color='#3b82f6', width=2), yaxis="y1"))
    
    fig.add_hrect(y0=0.0, y1=0.3, fillcolor="green", opacity=0.08, layer="below", line_width=0, annotation_text="DCA Accumulation Zone (<0.3)", annotation_position="top left")
    fig.add_hrect(y0=0.3, y1=0.6, fillcolor="gray", opacity=0.12, layer="below", line_width=0, annotation_text="Grey Region / Hold (0.3 - 0.6)", annotation_position="top left")
    fig.add_hrect(y0=0.6, y1=1.0, fillcolor="red", opacity=0.08, layer="below", line_width=0, annotation_text="DCA Distribution Zone (>0.6)", annotation_position="top left")
    
    fig.update_layout(
        template="plotly_dark", height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Risk Level (0 - 1)", side="right", range=[0, 1], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis2=dict(type="log", overlaying="y", side="left", showgrid=False, title="Preço (USD)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
