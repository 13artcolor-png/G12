# -*- coding: utf-8 -*-
"""
G12 - Filtre News
Detecte les annonces a haut impact (CPI, FOMC, etc.)
"""

from datetime import datetime
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Ajouter le chemin pour les imports
sys.path.append(str(Path(__file__).parent.parent))
from data.sentiment import get_sentiment

class NewsFilter:
    """Analyse les news pour detecter les evenements a haut risque"""
    
    HIGH_IMPACT_KEYWORDS = [
        "CPI", "FOMC", "FED", "Interest Rate", "Payrolls", "NFP", 
        "Inflation", "Jackson Hole", "Powell", "Lagarde", "ECB"
    ]

    def __init__(self):
        self.sentiment = get_sentiment()

    def get_high_impact_events(self) -> List[Dict]:
        """Scanne les news recentes pour trouver des evenements a haut impact"""
        news = self.sentiment.get_news(max_items=50)
        if not news:
            return []
            
        high_impact = []
        for item in news:
            title = item.get("title", "")
            matches = [word for word in self.HIGH_IMPACT_KEYWORDS if word.lower() in title.lower()]
            
            if matches:
                high_impact.append({
                    "title": title,
                    "source": item.get("source"),
                    "detected_keywords": matches,
                    "published": item.get("published"),
                    "impact": "HIGH"
                })
        
        return high_impact

    def should_pause_trading(self) -> bool:
        """Verifie si on doit mettre en pause le trading (annonce tres recente)"""
        events = self.get_high_impact_events()
        if not events:
            return False
            
        now = datetime.now()
        for event in events:
            pub_date_str = event.get("published")
            if pub_date_str:
                try:
                    pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    # Si l'annonce date de moins de 30 minutes
                    if (now - pub_date.replace(tzinfo=None)).total_seconds() < 1800:
                        return True
                except (ValueError, TypeError):
                    pass
        return False

# Singleton
_news_filter_instance = None

def get_news_filter() -> NewsFilter:
    global _news_filter_instance
    if _news_filter_instance is None:
        _news_filter_instance = NewsFilter()
    return _news_filter_instance
