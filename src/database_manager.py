import sqlite3
import pandas as pd
from datetime import datetime

class DatabaseManager:
    _instance = None

    def __new__(cls, db_path="data/crypto_system.db"):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.db_path = db_path
            cls._instance._initialize_tables()
        return cls._instance

    def connect(self):
        """Veritabanı bağlantısı oluşturur[cite: 623]."""
        return sqlite3.connect(self.db_path)

    def _initialize_tables(self):
        """SDD'de belirtilen tabloları oluşturur[cite: 668]."""
        conn = self.connect()
        cursor = conn.cursor()

        # OHLCV Veri Tablosu [cite: 634]
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                UNIQUE(symbol, timestamp)
            )
        ''')

        # Haber Veri Tablosu [cite: 641]
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                source TEXT,
                published_date INTEGER,
                sentiment_score REAL,
                UNIQUE(title, published_date)
            )
        ''')
        
        # Sinyal Tablosu [cite: 663]
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp INTEGER,
                signal TEXT,
                confidence REAL
            )
        ''')

        conn.commit()
        conn.close()
        print("Veritabanı ve tablolar hazır.")

    def insert_ohlcv(self, df, symbol):
        """Pandas DataFrame'i veritabanına kaydeder[cite: 626]."""
        conn = self.connect()
        try:
            # Timestamp ve Symbol sütunlarını ayarla
            data = df.copy()
            data['symbol'] = symbol
            # Sadece gerekli kolonları seç ve kaydet
            data[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].to_sql(
                'ohlcv_data', conn, if_exists='append', index=False, method='multi'
            )
            print(f"{symbol} için veriler kaydedildi.")
        except sqlite3.IntegrityError:
            print(f"{symbol} verileri zaten güncel.")
        except Exception as e:
            print(f"Hata: {e}")
        finally:
            conn.close()