# -*- coding: utf-8 -*-
"""
G12 - Donnees Binance Futures
Funding rate, Open Interest, Liquidations
"""

import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time


class BinanceData:
    """Recupere les donnees Binance Futures pour BTCUSD"""

    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.symbol = "BTCUSDT"
        self.cache = {}
        self.cache_duration = 10  # secondes

    def _get_cached(self, key: str) -> Optional[Dict]:
        """Recupere depuis le cache si valide"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_duration:
                return data
        return None

    def _set_cache(self, key: str, data: Dict):
        """Stocke dans le cache"""
        self.cache[key] = (data, time.time())

    def get_funding_rate(self) -> Optional[Dict]:
        """Recupere le funding rate actuel"""
        cached = self._get_cached("funding")
        if cached:
            return cached

        try:
            # Funding rate actuel
            url = f"{self.base_url}/fapi/v1/premiumIndex"
            response = requests.get(url, params={"symbol": self.symbol}, timeout=5)

            if response.status_code != 200:
                return None

            data = response.json()

            result = {
                "symbol": self.symbol,
                "funding_rate": float(data.get("lastFundingRate", 0)) * 100,  # En pourcentage
                "mark_price": float(data.get("markPrice", 0)),
                "index_price": float(data.get("indexPrice", 0)),
                "next_funding_time": datetime.fromtimestamp(
                    data.get("nextFundingTime", 0) / 1000
                ).isoformat() if data.get("nextFundingTime") else None,
                "timestamp": datetime.now().isoformat()
            }

            self._set_cache("funding", result)
            return result

        except Exception as e:
            print(f"[Binance] Erreur funding rate: {e}")
            return None

    def get_open_interest(self) -> Optional[Dict]:
        """Recupere l'Open Interest"""
        cached = self._get_cached("oi")
        if cached:
            return cached

        try:
            url = f"{self.base_url}/fapi/v1/openInterest"
            response = requests.get(url, params={"symbol": self.symbol}, timeout=5)

            if response.status_code != 200:
                return None

            data = response.json()

            # Recuperer aussi l'historique pour calculer le changement
            hist_url = f"{self.base_url}/futures/data/openInterestHist"
            hist_response = requests.get(hist_url, params={
                "symbol": self.symbol,
                "period": "5m",
                "limit": 12  # 1 heure
            }, timeout=5)

            change_1h = None
            if hist_response.status_code == 200:
                hist_data = hist_response.json()
                if len(hist_data) >= 2:
                    old_oi = float(hist_data[0].get("sumOpenInterest", 0))
                    new_oi = float(hist_data[-1].get("sumOpenInterest", 0))
                    if old_oi > 0:
                        change_1h = ((new_oi - old_oi) / old_oi) * 100

            result = {
                "symbol": self.symbol,
                "open_interest": float(data.get("openInterest", 0)),
                "open_interest_usd": float(data.get("openInterest", 0)) * float(data.get("openInterest", 0)),
                "change_1h_pct": round(change_1h, 2) if change_1h else None,
                "timestamp": datetime.now().isoformat()
            }

            self._set_cache("oi", result)
            return result

        except Exception as e:
            print(f"[Binance] Erreur open interest: {e}")
            return None

    def get_long_short_ratio(self) -> Optional[Dict]:
        """Recupere le ratio Long/Short des top traders"""
        cached = self._get_cached("ls_ratio")
        if cached:
            return cached

        try:
            # Ratio des top traders (positions)
            url = f"{self.base_url}/futures/data/topLongShortPositionRatio"
            response = requests.get(url, params={
                "symbol": self.symbol,
                "period": "5m",
                "limit": 1
            }, timeout=5)

            if response.status_code != 200:
                return None

            data = response.json()
            if not data:
                return None

            latest = data[-1]

            result = {
                "symbol": self.symbol,
                "long_ratio": float(latest.get("longAccount", 0)) * 100,
                "short_ratio": float(latest.get("shortAccount", 0)) * 100,
                "long_short_ratio": float(latest.get("longShortRatio", 1)),
                "timestamp": datetime.now().isoformat()
            }

            self._set_cache("ls_ratio", result)
            return result

        except Exception as e:
            print(f"[Binance] Erreur long/short ratio: {e}")
            return None

    def get_fibo3s(self, hours: int = 24) -> Optional[Dict]:
        """Recupere les liquidations recentes (approximation via trades forces)"""
        cached = self._get_cached("liquidations")
        if cached:
            return cached

        try:
            # Binance ne fournit pas directement les liquidations
            # On utilise l'endpoint forceOrders
            url = f"{self.base_url}/fapi/v1/forceOrders"

            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)

            response = requests.get(url, params={
                "symbol": self.symbol,
                "startTime": start_time,
                "endTime": end_time,
                "limit": 100
            }, timeout=5)

            # Note: Cet endpoint peut ne pas etre accessible sans API key
            if response.status_code != 200:
                # Fallback: estimer via le funding rate
                return {
                    "symbol": self.symbol,
                    "liquidations_24h_usd": None,
                    "long_fibo3s": None,
                    "short_fibo3s": None,
                    "note": "Donnees non disponibles sans API key",
                    "timestamp": datetime.now().isoformat()
                }

            data = response.json()

            total_long = 0
            total_short = 0

            for order in data:
                qty = float(order.get("origQty", 0))
                price = float(order.get("avgPrice", 0))
                value = qty * price

                if order.get("side") == "SELL":  # Long liquide
                    total_long += value
                else:
                    total_short += value

            result = {
                "symbol": self.symbol,
                "liquidations_24h_usd": total_long + total_short,
                "long_fibo3s_usd": total_long,
                "short_fibo3s_usd": total_short,
                "count": len(data),
                "timestamp": datetime.now().isoformat()
            }

            self._set_cache("liquidations", result)
            return result

        except Exception as e:
            print(f"[Binance] Erreur liquidations: {e}")
            return None

    def get_orderbook_imbalance(self, depth: int = 20) -> Optional[Dict]:
        """Calcule le desequilibre du carnet d'ordres"""
        try:
            url = f"{self.base_url}/fapi/v1/depth"
            response = requests.get(url, params={
                "symbol": self.symbol,
                "limit": depth
            }, timeout=5)

            if response.status_code != 200:
                return None

            data = response.json()

            # Somme des bids et asks
            bid_volume = sum(float(b[1]) for b in data.get("bids", []))
            ask_volume = sum(float(a[1]) for a in data.get("asks", []))

            total = bid_volume + ask_volume
            if total == 0:
                return None

            # Imbalance: positif = plus de bids (bullish), negatif = plus de asks (bearish)
            imbalance = ((bid_volume - ask_volume) / total) * 100

            return {
                "symbol": self.symbol,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "imbalance_pct": round(imbalance, 2),
                "bias": "bullish" if imbalance > 5 else "bearish" if imbalance < -5 else "neutral",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"[Binance] Erreur orderbook: {e}")
            return None

    def get_all_data(self) -> Dict:
        """Recupere toutes les donnees Binance"""
        return {
            "funding": self.get_funding_rate(),
            "open_interest": self.get_open_interest(),
            "long_short_ratio": self.get_long_short_ratio(),
            "liquidations": self.get_fibo3s(),
            "orderbook": self.get_orderbook_imbalance()
        }


# Singleton
_binance_instance = None

def get_binance() -> BinanceData:
    """Retourne l'instance Binance singleton"""
    global _binance_instance
    if _binance_instance is None:
        _binance_instance = BinanceData()
    return _binance_instance
