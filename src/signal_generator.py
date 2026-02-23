import pandas as pd

class HybridSignalGenerator:
    """
    SDD Bölüm 5.3.6: Hibrit Sinyal Üretici Sınıfı.
    Teknik, Duygu ve ML verilerini birleştirir.
    Ek Özellik: Geçmiş işlem başarısına göre dinamik eşik ayarı (Feedback Loop).
    """
    def __init__(self, tech_weight=0.4, sentiment_weight=0.2, ml_weight=0.4):
        # [cite_start]SDD'de belirtilen ağırlıklar [cite: 611-613]
        self.tech_weight = tech_weight
        self.sentiment_weight = sentiment_weight
        self.ml_weight = ml_weight

    def adjust_thresholds_based_on_history(self, trade_history):
        """
        Geçmiş işlemlere bakarak risk iştahını ayarlar (Adaptive Learning).
        Eğer son işlem zararsa, alım eşiğini zorlaştırır (Daha güvenli mod).
        """
        # Varsayılan "Güvenli" eşik
        base_threshold = 0.15 
        
        if not trade_history:
            return base_threshold
            
        last_trade = trade_history[-1]
        
        # Son işlem ZARAR ise (🔻 emojisi varsa)
        if "🔻" in last_trade: 
            # Eşiği yükselt (0.25), böylece sadece çok güçlü sinyallerde işlem yapar
            return 0.25 
        
        # Son işlem KÂR ise (💰 emojisi varsa)
        elif "💰" in last_trade: 
            # Standart eşiğe dön veya biraz daha agresif ol (0.10)
            return 0.10
            
        return base_threshold

    def generate_signal(self, current_price, predicted_price, sentiment_score, tech_indicators, trade_history=[]):
        """
        [cite_start]FR-20, FR-21: Girdileri birleştirip AL/SAT/TUT sinyali üretir. [cite: 284-286]
        """
        
        # 1. Teknik Analiz Skoru (-1 ile 1 arası)
        tech_score = 0
        rsi = tech_indicators['rsi_14'].iloc[-1]
        
        # RSI Mantığı: 30 altı AL, 70 üstü SAT
        if rsi < 30: tech_score += 0.5
        elif rsi > 70: tech_score -= 0.5
        
        # MACD Kesişimi
        macd = tech_indicators['macd'].iloc[-1]
        macd_signal = tech_indicators['macd_signal'].iloc[-1]
        
        if macd > macd_signal:
            tech_score += 0.5
        else:
            tech_score -= 0.5
            
        # 2. ML Tahmin Skoru (-1, 0, 1)
        ml_score = 0
        price_diff_ratio = (predicted_price - current_price) / current_price
        
        # Fiyat farkı %0.5'ten büyükse sinyal üret (Daha duyarlı olması için %1 yerine %0.5 seçildi)
        if price_diff_ratio > 0.005: ml_score = 1
        elif price_diff_ratio < -0.005: ml_score = -1
        
        # 3. Duygu Skoru (-1 ile 1 arası)
        sent_score = sentiment_score
        
        # [cite_start]4. Ağırlıklı Toplam Skor [cite: 284]
        final_score = (tech_score * self.tech_weight) + \
                      (sent_score * self.sentiment_weight) + \
                      (ml_score * self.ml_weight)
                      
        # 5. Dinamik Karar Mekanizması
        # Geçmişteki hatalardan ders alarak eşiği belirle
        threshold = self.adjust_thresholds_based_on_history(trade_history)
        
        signal = "HOLD"
        confidence = abs(final_score) # Güven skoru mutlak değerdir
        
        if final_score > threshold:
            signal = "BUY"
        elif final_score < -threshold:
            signal = "SELL"
            
        return signal, final_score