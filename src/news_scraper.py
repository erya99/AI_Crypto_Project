import requests
from bs4 import BeautifulSoup
from database_manager import DatabaseManager
from datetime import datetime

class NewsScraper:
    """
    SRS Bölüm 4.1.1 uyarınca Haber Toplama Modülü.
    Kripto para haber sitelerinden başlıkları çeker.
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        # Örnek olarak CoinDesk veya genel bir RSS akışı hedeflenebilir.
        # Burada basitlik adına genel bir yapı kuruyoruz.
        self.sources = [
            {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
            {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss"}
        ]

    def fetch_news(self):
        """
        FR-02: Tanımlı kaynaklardan güncel haberleri çeker.
        """
        all_news = []
        
        for source in self.sources:
            print(f"{source['name']} taranıyor...")
            try:
                response = requests.get(source['url'], timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml') # RSS için xml parser
                    items = soup.find_all('item')
                    
                    for item in items[:10]: # Her kaynaktan son 10 haber
                        news_item = {
                            'title': item.title.text,
                            'content': item.description.text if item.description else "",
                            'source': source['name'],
                            'published_date': int(datetime.now().timestamp()) # Basitleştirilmiş tarih
                        }
                        all_news.append(news_item)
                        self._save_to_db(news_item)
            except Exception as e:
                print(f"{source['name']} hatası: {e}")
                
        return all_news

    def _save_to_db(self, news_item):
        """
        FR-04: Haberi veritabanına kaydeder.
        """
        conn = self.db_manager.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO news_data (title, content, source, published_date, sentiment_score)
                VALUES (?, ?, ?, ?, 0)
            ''', (news_item['title'], news_item['content'], news_item['source'], news_item['published_date']))
            conn.commit()
        except Exception as e:
            print(f"DB Kayıt Hatası: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    scraper = NewsScraper()
    news = scraper.fetch_news()
    print(f"Toplam {len(news)} haber çekildi.")