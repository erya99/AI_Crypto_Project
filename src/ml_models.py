import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import joblib # YENİ: Scaler'ı kaydedip yüklemek için eklendi

class BaseMLModel:
    """
    Tüm ML modelleri için temel sınıf (Strategy Pattern).
    """
    def train(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
        
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return mean_absolute_error(y, predictions)

class LinearRegressionModel(BaseMLModel):
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

class RandomForestModel(BaseMLModel):
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

class LSTMModel(BaseMLModel):
    """
    Zaman serisi tahmini için LSTM (Derin Öğrenme).
    """
    def __init__(self, input_shape):
        self.model_path = "data/lstm_model.keras" 
        self.input_shape = input_shape
        
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except Exception as e:
                print(f"Model yükleme hatası, yeniden oluşturuluyor: {e}")
                self._build_model()
        else:
            self._build_model()

    def _build_model(self):
        """Model mimarisini oluşturur"""
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(shape=self.input_shape))
        self.model.add(tf.keras.layers.LSTM(50, return_sequences=True))
        
        # 3. DÜZELTME (OVERFITTING): Dropout eklendi (%20 unutma oranı)
        self.model.add(tf.keras.layers.Dropout(0.2)) 
        
        self.model.add(tf.keras.layers.LSTM(50, return_sequences=False))
        
        # 3. DÜZELTME (OVERFITTING): İkinci Dropout eklendi
        self.model.add(tf.keras.layers.Dropout(0.2))
        
        self.model.add(tf.keras.layers.Dense(25))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
    def train(self, X, y, epochs=5, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        try:
            self.model.save(self.model_path)
        except Exception as e:
            print(f"Model kayıt hatası: {e}")
        
    def predict(self, X):
        return self.model.predict(X)

class MLManager:
    """
    Veriyi hazırlayıp modelleri yöneten yardımcı sınıf.
    """
    def __init__(self):
        self.scaler_path = "data/scaler.save"

    def prepare_data(self, df, feature_cols=['close', 'volume'], target_col='close', lookback=60, is_training=True):
        """
        is_training=True ise modeli eğitmek için X,y üretir ve scaler kaydeder.
        is_training=False ise canlı trade için sadece son 60 mumu verir ve kayıtlı scaler'ı yükler.
        """
        if len(df) < lookback:
            return np.array([]), np.array([]), None

        data = df[feature_cols].values
        
        # 2. DÜZELTME (TRAINING-SERVING SKEW): Scaler kaydetme ve yükleme
        if is_training:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            os.makedirs("data", exist_ok=True)
            joblib.dump(scaler, self.scaler_path) # Eğitilen scaler'ı kaydet
        else:
            if os.path.exists(self.scaler_path):
                scaler = joblib.load(self.scaler_path) # Canlıda aynı scaler'ı yükle
                scaled_data = scaler.transform(data)   # fit_transform DEĞİL, transform yap!
            else:
                print("⚠️ Scaler dosyası bulunamadı, fallback yapılıyor.")
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)
        
        # 1. DÜZELTME (LOOKAHEAD BIAS): X ve y'yi doğru ayırma
        if is_training:
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i])
                y.append(scaled_data[i, 0]) 
            return np.array(X), np.array(y), scaler
        else:
            # Canlı sistem (Inference): Bize sadece GELECEĞİ (T+1) tahmin etmek için
            # EN SON lookback kadar veri (örneğin son 60 mum) lazım.
            X_latest = scaled_data[-lookback:]
            return np.array([X_latest]), None, scaler