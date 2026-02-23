import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os

class BaseMLModel:
    """
    SDD Bölüm 5.3.4: Tüm ML modelleri için temel sınıf (Strategy Pattern).
    """
    def train(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
        
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return mean_absolute_error(y, predictions)

class LinearRegressionModel(BaseMLModel):
    """
    FR-15: Basit fiyat tahmini için Doğrusal Regresyon.
    """
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

class RandomForestModel(BaseMLModel):
    """
    FR-16: Karmaşık ilişkileri yakalamak için Random Forest.
    """
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

class LSTMModel(BaseMLModel):
    """
    FR-17: Zaman serisi tahmini için LSTM (Derin Öğrenme).
    Modeli kaydeder ve yükler (Save/Load), böylece her seferinde eğitim yapmaz.
    """
    def __init__(self, input_shape):
        self.model_path = "data/lstm_model.keras" # Modelin kaydedileceği yer
        self.input_shape = input_shape
        
        # Eğer kayıtlı model varsa onu yükle, yoksa yeni oluştur
        if os.path.exists(self.model_path):
            try:
                # print("💾 Kayıtlı model yükleniyor...") # Konsolu kirletmemek için kapalı
                self.model = tf.keras.models.load_model(self.model_path)
            except Exception as e:
                print(f"Model yükleme hatası, yeniden oluşturuluyor: {e}")
                self._build_model()
        else:
            self._build_model()

    def _build_model(self):
        """Model mimarisini oluşturur"""
        self.model = tf.keras.models.Sequential()
        # Input katmanı ayrı eklendi (Keras 3.0 uyarısını çözer)
        self.model.add(tf.keras.layers.Input(shape=self.input_shape))
        self.model.add(tf.keras.layers.LSTM(50, return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(50, return_sequences=False))
        self.model.add(tf.keras.layers.Dense(25))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
    def train(self, X, y, epochs=5, batch_size=32):
        # Modeli eğit
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        # Eğitilen modeli kaydet
        try:
            self.model.save(self.model_path)
            # print("💾 Model güncellendi ve kaydedildi.")
        except Exception as e:
            print(f"Model kayıt hatası: {e}")
        
    def predict(self, X):
        return self.model.predict(X)

class MLManager:
    """
    Veriyi hazırlayıp modelleri yöneten yardımcı sınıf.
    """
    def prepare_data(self, df, feature_cols=['close', 'volume'], target_col='close', lookback=60):
        """
        Veriyi ML formatına çevirir. LSTM için 3D array oluşturur.
        """
        if len(df) < lookback:
            return np.array([]), np.array([]), None

        data = df[feature_cols].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0]) # Hedef: close price
            
        return np.array(X), np.array(y), scaler

# Test Bloğu
if __name__ == "__main__":
    # Rastgele veri ile test
    df_test = pd.DataFrame({
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000
    })
    
    manager = MLManager()
    X, y, scaler = manager.prepare_data(df_test, lookback=10)
    
    if len(X) > 0:
        # LSTM Testi
        lstm = LSTMModel(input_shape=(X.shape[1], X.shape[2]))
        try:
            lstm.train(X, y, epochs=1)
            print("LSTM Eğitimi ve Kaydı Başarılı!")
        except Exception as e:
            print(f"Hata: {e}")