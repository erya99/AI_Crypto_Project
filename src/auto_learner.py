import time
import schedule  # pip install schedule
from model_trainer import ModelTrainer
from datetime import datetime

class AutoLearner:
    """
    Modelin sürekli güncel kalmasını sağlayan Otomatik Öğrenme Modülü.
    Periyodik olarak verileri çeker, modeli eğitir ve kaydeder.
    """
    def __init__(self, symbol='BTC/USDT'):
        self.symbol = symbol
        self.trainer = ModelTrainer(symbol=symbol, limit=2000) # Son 2000 mumu baz al

    def job(self):
        print(f"\n🧠 [AUTO-LEARN] Otomatik eğitim başladı: {datetime.now()}")
        try:
            # 1. En güncel veriyi çek
            df = self.trainer.fetch_historical_data()
            if df.empty:
                print("⚠️ Veri çekilemedi, eğitim atlandı.")
                return

            # 2. İndikatörleri ekle
            df = self.trainer.add_features(df)

            # 3. Modeli "Fine-Tune" et (Sadece son bilgilerle güncelle)
            # Epoch sayısını düşük tutuyoruz (5) ki geçmişi unutmasın ama yeniyi öğrensin.
            X, y, scaler = self.trainer.ml_manager.prepare_data(df, lookback=60)
            
            # Kayıtlı modeli çağırıp üzerine eğitim yapacak (Transfer Learning)
            from ml_models import LSTMModel
            lstm = LSTMModel(input_shape=(X.shape[1], X.shape[2]))
            
            print("🏋️‍♂️ Model güncel piyasa verisiyle antrenman yapıyor...")
            lstm.train(X, y, epochs=5, batch_size=32)
            
            print("✅ Model başarıyla güncellendi ve kaydedildi!")
            
        except Exception as e:
            print(f"❌ Eğitim hatası: {e}")

    def start(self, interval_minutes=60):
        """
        Belirtilen dakikada bir eğitimi tetikler.
        """
        print(f"🕒 Otomatik Öğrenme Modülü Başlatıldı. ({interval_minutes} dakikada bir eğitilecek)")
        
        # İlk açılışta bir kez eğit
        self.job()
        
        # Zamanlayıcıyı kur
        schedule.every(interval_minutes).minutes.do(self.job)
        
        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    # Bu dosyayı ayrı bir terminalde çalıştıracaksın
    learner = AutoLearner()
    learner.start(interval_minutes=60) # Her 1 saatte bir modeli güncelle