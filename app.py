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

@st.cache_resource
def get_controller():
    return MainController()

st.set_page_config(page_title="AI Pro Trade Bot", layout="wide", page_icon="🤑")
st.title("🤑 AI Algoritmik Trade Botu (Dual Mode)")

with st.sidebar:
    st.header("⚙️ Konfigürasyon")
    trade_mode = st.radio("Çalışma Modu", ["PAPER (Sanal)", "REAL (Gerçek)"])
    
    api_key = None
    api_secret = None
    if trade_mode == "REAL (Gerçek)":
        st.warning("⚠️ DİKKAT: Gerçek para ile işlem yapılacaktır!")
        api_key = st.text_input("Binance API Key", type="password")
        api_secret = st.text_input("Binance Secret Key", type="password")
    
    symbol = st.selectbox("Parite", ["BTC/USDT", "ETH/USDT", "AVAX/USDT"])
    
    st.markdown("---")
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
        
    start_btn = st.button("▶️ Botu Başlat" if not st.session_state.is_running else "⏹️ Botu Durdur")
    if start_btn:
        st.session_state.is_running = not st.session_state.is_running
        mode_code = 'REAL' if trade_mode == "REAL (Gerçek)" else 'PAPER'
        try:
            st.session_state.trader = Trader(mode=mode_code, api_key=api_key, api_secret=api_secret)
            st.toast(f"Bot {mode_code} modunda başlatıldı!", icon="🚀")
        except Exception as e:
            st.error(f"Başlatma Hatası: {e}")
            st.session_state.is_running = False

status_place = st.empty()
metric_place = st.empty()
chart_place = st.empty()
log_place = st.container()

if st.session_state.is_running:
    controller = get_controller()
    trader = st.session_state.trader
    
    if trader is None:
        st.error("Lütfen botu durdurup tekrar başlatın.")
        st.stop()

    while st.session_state.is_running:
        try:
            with status_place.container():
                st.info(f"📡 {symbol} piyasası taranıyor... Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}")
            
            # Analiz
            results = controller.run_analysis(symbol)
            if "error" in results:
                st.error(results["error"])
                time.sleep(5)
                continue

            current_price = results['current_price']
            
            # DÜZELTME: Sinyal üretirken geçmiş işlemleri (trade_history) bota gönderiyoruz
            signal, confidence = controller.signal_generator.generate_signal(
                current_price, 
                results['predicted_price'], 
                results['sentiment_score'], 
                results['dataframe'],
                trader.trade_history
            )
            
            timestamp = results['dataframe']['timestamp'].iloc[-1]
            
            # İşlem Denemesi
            is_traded, log_msg = trader.execute_trade(signal, symbol, current_price, timestamp)
            
            usdt_bal, coin_bal = trader.get_balances(symbol)
            total_val = usdt_bal + (coin_bal * current_price)
            
            with metric_place.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("USDT Bakiye", f"${usdt_bal:.2f}")
                c2.metric("Coin Miktar", f"{coin_bal:.4f}")
                c3.metric("Toplam Portföy", f"${total_val:.2f}")
                
                sig_color = "green" if signal == "BUY" else "red" if signal == "SELL" else "gray"
                c4.markdown(f"### Sinyal: :{sig_color}[{signal}]")

            df = results['dataframe']
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name='Fiyat'), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['timestamp'].tail(20), y=[results['predicted_price']]*20, 
                                     name='ML Tahmini', line=dict(color='orange', dash='dot')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi_14'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            chart_place.plotly_chart(fig, use_container_width=True)
            
            with log_place:
                if is_traded: 
                    st.success(f"İŞLEM YAPILDI: {log_msg}")
                else:
                    if signal == "HOLD":
                        reason = "Sinyal Nötr (HOLD) / Yetersiz Güven Skoru"
                        icon = "⏳"
                    elif trader.in_position and signal == "BUY":
                        reason = "Zaten Alım Yapılmış"
                        icon = "🔒"
                    elif not trader.in_position and signal == "SELL":
                        reason = "Satılacak Coin Yok"
                        icon = "🚫"
                    else:
                        reason = "Bakiye Yetersiz"
                        icon = "⚠️"
                        
                    st.info(f"{icon} Durum: Beklemede... Sebep: {reason}")

                st.subheader("📜 İşlem Geçmişi")
                if not trader.trade_history:
                    st.text("Henüz işlem kaydı yok.")
                for log in reversed(trader.trade_history):
                    st.code(log)
            
            # DÜZELTME: Daha anlamlı işlemler için bekleme süresi 60 saniyeye çıkarıldı
            time.sleep(60)
            
        except Exception as e:
            st.error(f"Kritik Döngü Hatası: {e}")
            time.sleep(10)