# -*- coding: utf-8 -*-
"""
G12 - Moteur Macro
Recupere DXY et S&P 500 pour correlation BTC
"""

import yfinance as yf
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

class MacroEngine:
    """Analyse les correlations macro (DXY, S&P 500)"""
    
    def __init__(self):
        self.symbols = {
            "dxy": "DX-Y.NYB",  # Dollar Index
            "sp500": "^GSPC"    # S&P 500
        }
        self.cache = {}
        self.cache_duration = 3600  # 1 heure (macro bouge lentement)
        self.last_update = None

    def get_macro_data(self) -> Dict:
        """Recupere les dernieres valeurs et variations"""
        now = datetime.now()
        if self.last_update and (now - self.last_update).total_seconds() < self.cache_duration:
            return self.cache

        data = {}
        try:
            for name, ticker in self.symbols.items():
                t = yf.Ticker(ticker)
                # Periode plus large pour les weekends
                hist = t.history(period="5d")
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    # Si le marché est fermé (aujourd'hui = weekend), on prend les 2 derniers ouverts
                    if pd.isna(current) or current == 0:
                         current = hist['Close'].dropna().iloc[-1]
                         prev = hist['Close'].dropna().iloc[-2]
                    
                    change = ((current - prev) / prev) * 100
                    data[name] = {
                        "value": round(current, 2),
                        "change_24h": round(change, 2)
                    }
                else:
                    data[name] = {"value": None, "change_24h": None}

            # Sentiment Macro
            # DXY UP = Bearish pour BTC
            # SP500 UP = Bullish pour BTC (Risk-on)
            dxy_change = data.get("dxy", {}).get("change_24h", 0) or 0
            sp500_change = data.get("sp500", {}).get("change_24h", 0) or 0
            
            macro_bias = "neutral"
            if dxy_change > 0.1 and sp500_change < -0.1:
                macro_bias = "bearish"
            elif dxy_change < -0.1 and sp500_change > 0.1:
                macro_bias = "bullish"
            
            data["macro_bias"] = macro_bias
            data["timestamp"] = now.isoformat()
            
            self.cache = data
            self.last_update = now
            return data

        except Exception as e:
            print(f"[Macro] Erreur yfinance: {e}")
            return {"error": str(e)}

# Singleton
_macro_instance = None

def get_macro_engine() -> MacroEngine:
    global _macro_instance
    if _macro_instance is None:
        _macro_instance = MacroEngine()
    return _macro_instance
