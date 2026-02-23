import ccxt
import pandas as pd
from database_manager import DatabaseManager

class CryptoDataCollector:
    def __init__(self, exchange_name='binance'):
        # SDD [cite: 582] uyarınca borsa ismi parametrik
        self.exchange = getattr(ccxt, exchange_name)()
        self.db_manager = DatabaseManager()

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        """Borsadan OHLCV verisini çeker[cite: 586]."""
        try:
            print(f"{symbol} verisi çekiliyor...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Veriyi DataFrame'e çevir ve temizle [cite: 588]
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Timestamp'i okunabilir formata çevirmek isteyebiliriz ama
            # SDD şemasında INTEGER tutuluyor, o yüzden raw bırakıyoruz.
            
            # Veritabanına kaydet
            self.db_manager.insert_ohlcv(df, symbol)
            return df
            
        except Exception as e:
            print(f"Veri çekme hatası: {e}")
            return None

# Test etmek için (Main Controller'dan çağrılacak):
if __name__ == "__main__":
    collector = CryptoDataCollector()
    collector.fetch_ohlcv('BTC/USDT')
    collector.fetch_ohlcv('ETH/USDT')