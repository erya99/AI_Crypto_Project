import pandas as pd
import numpy as np

class TechnicalAnalysis:
    """
    SDD Bölüm 5.3.2 uyarınca Teknik Analiz Sınıfı.
    OHLCV verilerini alıp teknik indikatörleri hesaplar.
    """
    
    def __init__(self, df):
        # Veri setini kopyalayarak işlem yapıyoruz, orijinali bozulmasın
        self.df = df.copy()
        
        # Eğer 'close' sütunu yoksa hata vermeli
        if 'close' not in self.df.columns:
            raise ValueError("DataFrame 'close' fiyat sütununu içermelidir.")

    def calculate_ema(self, period=14):
        """
        FR-06: Üstel Hareketli Ortalama (EMA) hesaplar.
        """
        return self.df['close'].ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, period=14):
        """
        FR-07: Göreceli Güç Endeksi (RSI) hesaplar.
        """
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, fast=12, slow=26, signal=9):
        """
        FR-08: MACD (Moving Average Convergence Divergence) hesaplar.
        Dönüş: (macd_line, signal_line, histogram)
        """
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """
        FR-09: Bollinger Bantlarını hesaplar.
        Dönüş: (upper_band, middle_band, lower_band)
        """
        middle_band = self.df['close'].rolling(window=period).mean()
        std = self.df['close'].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band

    def get_all_indicators(self):
        """
        Tüm indikatörleri hesaplayıp DataFrame'e ekler ve döner.
        """
        self.df['ema_12'] = self.calculate_ema(period=12)
        self.df['ema_26'] = self.calculate_ema(period=26)
        self.df['rsi_14'] = self.calculate_rsi(period=14)
        
        macd, signal, _ = self.calculate_macd()
        self.df['macd'] = macd
        self.df['macd_signal'] = signal
        
        upper, middle, lower = self.calculate_bollinger_bands()
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = middle
        self.df['bb_lower'] = lower
        
        # İlk hesaplamalarda oluşan NaN değerleri temizle
        return self.df.dropna()