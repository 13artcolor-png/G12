# -*- coding: utf-8 -*-
"""
G12 - Whale Tracker
Surveille les gros mouvements institutionnels
"""

import requests
from datetime import datetime
from typing import List, Dict, Optional

class WhaleTracker:
    """Detecte les baleines (Whales) via Binance et APIs publiques"""
    
    def __init__(self):
        self.binance_url = "https://fapi.binance.com/fapi/v1/aggTrades"
        self.symbol = "BTCUSDT"
        self.whale_threshold_usd = 100000  # 100k USD = Whale trade
        self.cache = []
        self.last_update = None

    def get_recent_whales(self, limit: int = 10) -> List[Dict]:
        """Recupere les trades massifs recents sur Binance"""
        try:
            # Recuperer les derniers trades aggregates
            response = requests.get(self.binance_url, params={
                "symbol": self.symbol,
                "limit": 1000  # Scanner les 1000 derniers
            }, timeout=5)
            
            if response.status_code != 200:
                return []
                
            trades = response.json()
            whales = []
            
            for t in trades:
                qty = float(t.get('q', 0))
                price = float(t.get('p', 0))
                value = qty * price
                
                if value >= self.whale_threshold_usd:
                    whales.append({
                        "price": price,
                        "qty": qty,
                        "value_usd": round(value, 2),
                        "side": "SELL" if t.get('m') else "BUY",
                        "time": datetime.fromtimestamp(t.get('T', 0) / 1000).isoformat()
                    })
            
            # Trier par valeur et limiter
            whales.sort(key=lambda x: x["value_usd"], reverse=True)
            self.cache = whales[:limit]
            self.last_update = datetime.now()
            return self.cache

        except Exception as e:
            print(f"[Whale] Erreur: {e}")
            return []

    def get_whale_bias(self) -> Dict:
        """Calcule le sentiment des baleines sur les 1000 derniers trades"""
        whales = self.get_recent_whales(limit=50)
        if not whales:
            return {"bias": "neutral", "whale_power": 0}
            
        buy_vol = sum(w["value_usd"] for w in whales if w["side"] == "BUY")
        sell_vol = sum(w["value_usd"] for w in whales if w["side"] == "SELL")
        
        total = buy_vol + sell_vol
        if total == 0:
            return {"bias": "neutral", "whale_power": 0}
            
        imb = (buy_vol - sell_vol) / total
        
        return {
            "bias": "bullish" if imb > 0.2 else "bearish" if imb < -0.2 else "neutral",
            "whale_buy_vol": round(buy_vol, 0),
            "whale_sell_vol": round(sell_vol, 0),
            "imbalance": round(imb, 2)
        }

# Singleton
_whale_instance = None

def get_whale_tracker() -> WhaleTracker:
    global _whale_instance
    if _whale_instance is None:
        _whale_instance = WhaleTracker()
    return _whale_instance
