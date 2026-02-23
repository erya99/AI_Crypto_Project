from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalysis:
    """
    SDD Bölüm 5.3.3 uyarınca Duygu Analizi Sınıfı.
    VADER algoritması kullanarak metinleri analiz eder.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        """
        FR-11, FR-12: Tek bir metin için duygu skoru üretir.
        Dönüş: -1 (Çok Negatif) ile +1 (Çok Pozitif) arası float değer (compound score).
        """
        if not text:
            return 0.0
        
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']

    def aggregate_scores(self, news_list):
        """
        FR-13: Birden fazla haberin ortalama duygu skorunu hesaplar.
        news_list: Haber metinlerinden oluşan bir liste ['Haber 1', 'Haber 2'...]
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

# Basit bir test bloğu
if __name__ == "__main__":
    sa = SentimentAnalysis()
    test_news = "Bitcoin hits a new all-time high as investors are very excited!"
    print(f"Test Skoru: {sa.analyze_text(test_news)}")