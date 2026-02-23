import pandas as pd

class HybridSignalGenerator:
    """
    SDD Bölüm 5.3.6: Hibrit Sinyal Üretici Sınıfı.
    Teknik, Duygu ve ML verilerini birleştirir.
    Geçmiş işlem başarısına göre dinamik eşik ayarı (Feedback Loop) içerir.
    """
    def __init__(self, tech_weight=0.4, sentiment_weight=0.2, ml_weight=0.4):
        self.tech_weight = tech_weight
        self.sentiment_weight = sentiment_weight
        self.ml_weight = ml_weight

    def adjust_thresholds_based_on_history(self, trade_history):
        """
        Geçmiş işlemlere bakarak risk iştahını ayarlar.
        Eşikleri yükselttik (Bot artık çok emin olmadan işlem yapmayacak).
        """
        base_threshold = 0.20  # Eskiden 0.15'ti. Artık daha zor sinyal üretecek.
        
        if not trade_history:
            return base_threshold
            
        last_trade = trade_history[-1]
        
        if "🔻" in last_trade: 
            return 0.30 # Zarar edildiyse çok daha zor işlem yap (Defansif Mod)
        elif "💰" in last_trade: 
            return 0.15 # Kâr edildiyse biraz daha rahat işlem yapabilir
            
        return base_threshold

    def generate_signal(self, current_price, predicted_price, sentiment_score, tech_indicators, trade_history=[]):
        """
        Girdileri birleştirip AL/SAT/TUT sinyali üretir.
        """
        
        # 1. Teknik Analiz Skoru
        tech_score = 0
        rsi = tech_indicators['rsi_14'].iloc[-1]
        
        if rsi < 30: tech_score += 0.5
        elif rsi > 70: tech_score -= 0.5
        
        macd = tech_indicators['macd'].iloc[-1]
        macd_signal = tech_indicators['macd_signal'].iloc[-1]
        
        if macd > macd_signal:
            tech_score += 0.5
        else:
            tech_score -= 0.5
            
        # 2. ML Tahmin Skoru
        ml_score = 0
        price_diff_ratio = (predicted_price - current_price) / current_price
        
        if price_diff_ratio > 0.005: ml_score = 1
        elif price_diff_ratio < -0.005: ml_score = -1
        
        # 3. Duygu Skoru
        sent_score = sentiment_score
        
        # 4. Ağırlıklı Toplam Skor
        final_score = (tech_score * self.tech_weight) + \
                      (sent_score * self.sentiment_weight) + \
                      (ml_score * self.ml_weight)
                      
        # 5. Dinamik Karar Mekanizması
        threshold = self.adjust_thresholds_based_on_history(trade_history)
        
        signal = "HOLD"
        confidence = abs(final_score)
        
        if final_score > threshold:
            signal = "BUY"
        elif final_score < -threshold:
            signal = "SELL"
            
        return signal, final_score