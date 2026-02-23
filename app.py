import streamlit as st
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from main_controller import MainController
from trader import Trader

st.set_page_config(page_title="AI Pro Trade Bot", layout="wide", page_icon="🤑")

st.title("🤑 AI Algoritmik Trade Botu (Dual Mode)")

# --- KENAR ÇUBUĞU ---
with st.sidebar:
    st.header("⚙️ Konfigürasyon")
    
    # 1. Mod Seçimi
    trade_mode = st.radio("Çalışma Modu", ["PAPER (Sanal)", "REAL (Gerçek)"])
    
    # 2. Gerçek Mod İçin API Girişi
    api_key = None
    api_secret = None
    if trade_mode == "REAL (Gerçek)":
        st.warning("⚠️ DİKKAT: Gerçek para ile işlem yapılacaktır!")
        api_key = st.text_input("Binance API Key", type="password")
        api_secret = st.text_input("Binance Secret Key", type="password")
    
    symbol = st.selectbox("Parite", ["BTC/USDT", "ETH/USDT", "AVAX/USDT"])
    
    st.markdown("---")
    
    # Botu Başlat/Durdur
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
        
    start_btn = st.button("▶️ Botu Başlat" if not st.session_state.is_running else "⏹️ Botu Durdur")
    if start_btn:
        st.session_state.is_running = not st.session_state.is_running
        # Trader nesnesini her başlatmada yeniden oluştur (Ayarlar değişmiş olabilir)
        mode_code = 'REAL' if trade_mode == "REAL (Gerçek)" else 'PAPER'
        try:
            st.session_state.trader = Trader(mode=mode_code, api_key=api_key, api_secret=api_secret)
            st.toast(f"Bot {mode_code} modunda başlatıldı!", icon="🚀")
        except Exception as e:
            st.error(f"Başlatma Hatası: {e}")
            st.session_state.is_running = False

# --- ANA EKRAN ---
status_place = st.empty()
metric_place = st.empty()
chart_place = st.empty()
log_place = st.container()

if st.session_state.is_running:
    controller = MainController()
    trader = st.session_state.trader
    
    while st.session_state.is_running:
        try:
            with status_place.container():
                st.info(f"📡 {symbol} piyasası taranıyor... [{datetime.now().strftime('%H:%M:%S')}]")
            
            # 1. Analiz
            results = controller.run_analysis(symbol)
            if "error" in results:
                st.error(results["error"])
                time.sleep(5)
                continue

            current_price = results['current_price']
            signal = results['signal']
            timestamp = results['dataframe']['timestamp'].iloc[-1]
            
            # 2. İşlem
            is_traded, log_msg = trader.execute_trade(signal, symbol, current_price, timestamp)
            
            # 3. Bakiyeleri Güncelle
            usdt_bal, coin_bal = trader.get_balances(symbol)
            total_val = usdt_bal + (coin_bal * current_price)
            
            with metric_place.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("USDT Bakiye", f"${usdt_bal:.2f}")
                c2.metric("Coin Miktar", f"{coin_bal:.4f}")
                c3.metric("Toplam Portföy", f"${total_val:.2f}")
                
                sig_color = "green" if signal == "BUY" else "red" if signal == "SELL" else "gray"
                c4.markdown(f"### Sinyal: :{sig_color}[{signal}]")

            # 4. Grafik (Basitleştirilmiş canlı grafik)
            df = results['dataframe']
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name='Fiyat'), row=1, col=1)
            # ML Tahmin Çizgisi (Son 20 veri + 1 gelecek)
            fig.add_trace(go.Scatter(x=df['timestamp'].tail(20), y=[results['predicted_price']]*20, 
                                     name='ML Tahmini', line=dict(color='orange', dash='dot')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_14'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            chart_place.plotly_chart(fig, use_container_width=True)
            
            # 5. Loglar
            with log_place:
                if is_traded: st.toast(log_msg, icon="🔔")
                st.subheader("📜 İşlem Geçmişi")
                for log in reversed(trader.trade_history):
                    st.code(log)
            
            # Döngü Bekleme Süresi (Gerçek işlemde API limitleri için min 10-30sn önerilir)
            time.sleep(15)
            
        except Exception as e:
            st.error(f"Kritik Döngü Hatası: {e}")
            time.sleep(10)
else:
    status_place.warning("Bot durduruldu. Ayarları yapıp 'Botu Başlat' butonuna basın.")