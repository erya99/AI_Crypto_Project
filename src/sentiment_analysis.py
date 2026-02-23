from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalysis:
    """
    SDD Bölüm 5.3.3 uyarınca Duygu Analizi Sınıfı.
    VADER algoritması kullanarak metinleri analiz eder. Kripto sözlüğü eklenmiştir.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Kripto Jargonunu VADER'a öğretiyoruz (ÖZEL EKLENTİ)
        crypto_lexicon = {
            'bullish': 2.0, 'bearish': -2.0, 
            'moon': 2.0, 'rekt': -2.0, 
            'dump': -2.0, 'pump': 2.0, 
            'ath': 1.5, 'hack': -2.0, 
            'scam': -2.0, 'fud': -1.5,
            'surge': 1.5, 'plunge': -1.5
        }
        self.analyzer.lexicon.update(crypto_lexicon)

    def analyze_text(self, text):
        """
        Tek bir metin için duygu skoru üretir.
        Dönüş: -1 (Çok Negatif) ile +1 (Çok Pozitif) arası float değer.
        """
        if not text:
            return 0.0
        
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']

    def aggregate_scores(self, news_list):
        """
        Birden fazla haberin ortalama duygu skorunu hesaplar.
        """
        if not news_list:
            return 0.0
            
        total_score = 0
        count = 0
        
        for news in news_list:
            score = self.analyze_text(news)
            total_score += score
            count += 1
            
        if count == 0:
            return 0.0
            
        return total_score / count