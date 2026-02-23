import time
import schedule
from model_trainer import ModelTrainer
from datetime import datetime

class AutoLearner:
    """
    Modelin sürekli güncel kalmasını sağlayan Otomatik Öğrenme Modülü.
    """
    # VARSAYILAN PARİTE AVAX OLARAK DEĞİŞTİRİLDİ
    def __init__(self, symbol='AVAX/USDT'): 
        self.symbol = symbol
        self.trainer = ModelTrainer(symbol=symbol, limit=2000)

    def job(self):
        print(f"\n🧠 [AUTO-LEARN] Otomatik eğitim başladı: {datetime.now()}")
        try:
            df = self.trainer.fetch_historical_data()
            if df.empty:
                print("⚠️ Veri çekilemedi, eğitim atlandı.")
                return

            df = self.trainer.add_features(df)
            X, y, scaler = self.trainer.ml_manager.prepare_data(df, lookback=60, is_training=True)
            
            from ml_models import LSTMModel
            lstm = LSTMModel(input_shape=(X.shape[1], X.shape[2]))
            
            print(f"🏋️‍♂️ Model {self.symbol} piyasa verisiyle antrenman yapıyor...")
            lstm.train(X, y, epochs=5, batch_size=32)
            
            print(f"✅ {self.symbol} Modeli başarıyla güncellendi ve kaydedildi!")
            
        except Exception as e:
            print(f"❌ Eğitim hatası: {e}")

    def start(self, interval_minutes=60):
        print(f"🕒 Otomatik Öğrenme Modülü Başlatıldı. ({interval_minutes} dakikada bir eğitilecek)")
        self.job()
        schedule.every(interval_minutes).minutes.do(self.job)
        
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    # SADECE AVAX İÇİN ÇALIŞTIRILIYOR
    learner = AutoLearner(symbol='AVAX/USDT')
    learner.start(interval_minutes=60)