import pandas as pd
import numpy as np
import ccxt
import pandas_ta as ta  # Teknik analiz için (pip install pandas_ta gerekebilir)
from ml_models import LSTMModel, MLManager
from database_manager import DatabaseManager
import time

class ModelTrainer:
    def __init__(self, symbol='BTC/USDT', timeframe='1h', limit=1000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit # Ne kadar geçmiş veri çekilecek? (1000 mum ~ 40 gün)
        self.exchange = ccxt.binance()
        self.db = DatabaseManager()
        self.ml_manager = MLManager()

    def fetch_historical_data(self):
        """
        Borsadan geçmişe dönük büyük veri setini çeker.
        Binance API bir seferde max 1000 veri verir, döngüyle daha fazlası alınabilir.
        """
        print(f"📥 {self.symbol} için {self.limit} adet geçmiş veri çekiliyor...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            print(f"Veri çekme hatası: {e}")
            return pd.DataFrame()

    def add_features(self, df):
        """
        Veriyi zenginleştirir (Feature Engineering).
        """
        # Pandas-TA kütüphanesi veya manuel hesaplama ile indikatör ekle
        # Bizim TechnicalAnalysis sınıfımız da kullanılabilir ama eğitim için hızlı hesap lazım
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        df = pd.concat([df, macd], axis=1)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'])
        df = pd.concat([df, bb], axis=1)
        
        # Eksik verileri temizle (ilk hesaplamalar NaN döner)
        df.dropna(inplace=True)
        return df

    def train_initial_model(self):
        """
        Modeli sıfırdan, sağlam verilerle eğitir ve kaydeder.
        """
        # 1. Veriyi Getir
        df = self.fetch_historical_data()
        if df.empty:
            return

        # 2. İndikatörleri Ekle (Modelin zekasını artırır)
        # Not: Basitlik için şimdilik sadece close/volume kullanıyoruz, 
        # ama MLManager'ı tüm sütunları alacak şekilde güncelleyebiliriz.
        
        print(f"🧠 Eğitim başlıyor... Veri boyutu: {len(df)}")
        
        # 3. Veriyi Hazırla (X, y split)
        # lookback=60 (Son 60 saatlik veriye bakıp gelecek saati tahmin et)
        X, y, scaler = self.ml_manager.prepare_data(df, lookback=60)
        
        if len(X) == 0:
            print("Yetersiz veri!")
            return

        # 4. Modeli Başlat ve Eğit (Ağır Eğitim)
        # 50 Epoch: Modelin veriyi iyice öğrenmesini sağlar
        lstm = LSTMModel(input_shape=(X.shape[1], X.shape[2]))
        
        print("🏋️‍♂️ Model ağırlık kaldırıyor (50 Epoch)... Bu işlem biraz sürebilir.")
        lstm.train(X, y, epochs=50, batch_size=32)
        
        print("✅ Model başarıyla eğitildi ve 'data/lstm_model.keras' konumuna kaydedildi.")
        print("🤖 Artık botu (app.py) başlattığında bu 'akıllı' modeli kullanacak!")

if __name__ == "__main__":
    # Gerekli kütüphane uyarısı
    try:
        import pandas_ta
    except ImportError:
        print("Lütfen pandas_ta kütüphanesini yükleyin: pip install pandas_ta")
        exit()

    trainer = ModelTrainer(symbol='BTC/USDT', limit=2000) # Yaklaşık 3 aylık veri
    trainer.train_initial_model()