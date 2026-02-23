import pandas as pd
import numpy as np
from data_collector import CryptoDataCollector
from news_scraper import NewsScraper
from technical_analysis import TechnicalAnalysis
from sentiment_analysis import SentimentAnalysis
from ml_models import LSTMModel, MLManager
from signal_generator import HybridSignalGenerator
from database_manager import DatabaseManager

class MainController:
    """
    Tüm sistemi koordine eden ana sınıf.
    """
    def __init__(self):
        self.db = DatabaseManager()
        self.collector = CryptoDataCollector()
        self.news_scraper = NewsScraper()
        self.sentiment_analyzer = SentimentAnalysis()
        self.signal_generator = HybridSignalGenerator()
        self.ml_manager = MLManager()
        
    def run_analysis(self, symbol='BTC/USDT'):
        results = {}
        
        # 1. Veri Toplama
        print("1. Veriler toplanıyor...")
        df = self.collector.fetch_ohlcv(symbol)
        news_list = self.news_scraper.fetch_news()
        
        if df is None or df.empty:
            return {"error": "Borsa verisi alınamadı."}
            
        # 2. Teknik Analiz
        print("2. Teknik analiz yapılıyor...")
        ta = TechnicalAnalysis(df)
        df_analyzed = ta.get_all_indicators()
        results['dataframe'] = df_analyzed
        
        # 3. Duygu Analizi
        print("3. Duygu analizi yapılıyor...")
        news_texts = [n['title'] for n in news_list]
        sentiment_score = self.sentiment_analyzer.aggregate_scores(news_texts)
        results['sentiment_score'] = sentiment_score
        
        # 4. ML Tahmini (LSTM)
        print("4. Fiyat tahmini yapılıyor...")
        X, y, scaler = self.ml_manager.prepare_data(df_analyzed)
        if len(X) > 0:
            lstm = LSTMModel(input_shape=(X.shape[1], X.shape[2]))
            
            # DÜZELTME: lstm.train(...) KALDIRILDI! Model burada sadece tahmin yapacak.
            # Öğrenme işini auto_learner.py yapıyor.
            
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            predicted_scaled = lstm.predict(last_sequence)
            predicted_price = scaler.inverse_transform([[predicted_scaled[0][0], 0]])[0][0]
            results['predicted_price'] = float(predicted_price)
        else:
            results['predicted_price'] = df_analyzed['close'].iloc[-1]
            
        # 5. Sinyal Üretimi
        print("5. Sinyal üretiliyor...")
        current_price = df_analyzed['close'].iloc[-1]
        
        # Trade geçmişini almak için (Geçici boş liste veriyoruz, asıl geçmiş app.py'de)
        signal, confidence = self.signal_generator.generate_signal(
            current_price, 
            results['predicted_price'], 
            sentiment_score, 
            df_analyzed
        )
        
        results['signal'] = signal
        results['confidence'] = confidence
        results['current_price'] = current_price
        
        return results