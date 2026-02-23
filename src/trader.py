import ccxt
from datetime import datetime

class Trader:
    """
    Hem Sanal (Paper) hem de Gerçek (Real) ticareti yöneten hibrit sınıf.
    """
    def __init__(self, mode='PAPER', exchange_id='binance', api_key=None, api_secret=None, paper_balance=10000):
        self.mode = mode
        self.in_position = False
        self.trade_history = []
        
        # --- PAPER MODE AYARLARI ---
        self.paper_usdt = paper_balance
        self.paper_crypto = 0
        self.paper_entry_price = 0

        # --- REAL MODE AYARLARI ---
        self.exchange = None
        if self.mode == 'REAL':
            if not api_key or not api_secret:
                raise ValueError("Gerçek işlem için API Key ve Secret gereklidir!")
            
            # CCXT ile Borsa Bağlantısı
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'} # Spot piyasa
            })
            print("🔌 Borsa bağlantısı kuruldu (REAL MODE).")

    def get_balances(self, symbol):
        """
        Mevcut USDT ve Coin bakiyesini getirir.
        """
        base_currency = symbol.split('/')[0] # BTC
        quote_currency = symbol.split('/')[1] # USDT

        if self.mode == 'PAPER':
            return self.paper_usdt, self.paper_crypto
        
        elif self.mode == 'REAL':
            try:
                balance = self.exchange.fetch_balance()
                usdt_free = balance.get(quote_currency, {}).get('free', 0)
                coin_free = balance.get(base_currency, {}).get('free', 0)
                return usdt_free, coin_free
            except Exception as e:
                print(f"Bakiye hatası: {e}")
                return 0, 0

    def execute_trade(self, signal, symbol, current_price, timestamp):
        """
        Sinyale göre (AL/SAT) işlem yapar.
        """
        date_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M')
        usdt_bal, coin_bal = self.get_balances(symbol)
        
        # --- ALIM (BUY) ---
        if signal == "BUY" and not self.in_position:
            # Bakiyenin %99'u ile al (Komisyon payı bırak)
            amount_usdt = usdt_bal * 0.99 
            
            if amount_usdt < 10: # Binance min işlem limiti genellikle 10$
                return False, "❌ Yetersiz Bakiye (Min 10$)"

            if self.mode == 'PAPER':
                amount_coin = amount_usdt / current_price
                self.paper_crypto = amount_coin
                self.paper_usdt = 0
                self.paper_entry_price = current_price
                log = f"🔵 [SANAL] ALIM: {current_price}$ fiyatından alındı."
            
            elif self.mode == 'REAL':
                try:
                    # Piyasa emri ile al (Market Buy)
                    # amount_coin hesaplaması yerine create_market_buy_order cost parametresi (bazı borsalar desteklemez)
                    # O yüzden coin miktarını hesaplayıp gönderiyoruz
                    amount_coin = amount_usdt / current_price
                    order = self.exchange.create_market_buy_order(symbol, amount_coin)
                    log = f"🟢 [GERÇEK] ALIM EMRİ GİRİLDİ: {amount_coin:.4f} adet."
                except Exception as e:
                    return False, f"Borsa Hatası: {e}"

            self.in_position = True
            self.trade_history.append(log)
            return True, log

        # --- SATIŞ (SELL) ---
        elif signal == "SELL" and self.in_position:
            if coin_bal <= 0 and self.mode == 'REAL':
                self.in_position = False
                return False, "Satılacak coin yok."

            if self.mode == 'PAPER':
                new_balance = self.paper_crypto * current_price
                profit = new_balance - (self.paper_crypto * self.paper_entry_price)
                self.paper_usdt = new_balance
                self.paper_crypto = 0
                
                emoji = "💰" if profit > 0 else "🔻"
                log = f"{emoji} [SANAL] SATIŞ: P/L: {profit:.2f}$"

            elif self.mode == 'REAL':
                try:
                    # Tüm coini sat
                    order = self.exchange.create_market_sell_order(symbol, coin_bal)
                    log = f"🔴 [GERÇEK] SATIŞ EMRİ GİRİLDİ."
                except Exception as e:
                    return False, f"Borsa Hatası: {e}"

            self.in_position = False
            self.trade_history.append(log)
            return True, log

        return False, "İşlem yapılmadı."