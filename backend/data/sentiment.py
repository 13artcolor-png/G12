# -*- coding: utf-8 -*-
"""
G12 - Donnees Sentiment
Fear & Greed Index, News RSS
"""

import requests
import feedparser
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import re
import json


class SentimentData:
    """Recupere les donnees de sentiment pour BTC"""

    def __init__(self):
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.news_feeds = [
            {
                "name": "CoinTelegraph",
                "url": "https://cointelegraph.com/rss",
                "category": "crypto"
            },
            {
                "name": "Bitcoin Magazine",
                "url": "https://bitcoinmagazine.com/feed",
                "category": "bitcoin"
            }
        ]
        self.coingecko_url = "https://api.coingecko.com/api/v3/global"
        self.cache = {}
        self.cache_duration = 60  # secondes
        self.cache_duration_ai = 900  # 15 minutes

    def _get_cached(self, key: str, duration: Optional[int] = None) -> Optional[Dict]:
        """Recupere depuis le cache si valide"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            valid_duration = duration or self.cache_duration
            if time.time() - timestamp < valid_duration:
                return data
        return None

    def _set_cache(self, key: str, data: Dict):
        """Stocke dans le cache"""
        self.cache[key] = (data, time.time())

    def get_fear_greed_index(self) -> Optional[Dict]:
        """Recupere le Fear & Greed Index"""
        cached = self._get_cached("fear_greed")
        if cached:
            return cached

        try:
            response = requests.get(self.fear_greed_url, timeout=5)

            if response.status_code != 200:
                return None

            data = response.json()

            if "data" not in data or len(data["data"]) == 0:
                return None

            current = data["data"][0]

            # Interpreter le niveau
            value = int(current.get("value", 50))
            if value <= 25:
                interpretation = "Extreme Fear"
                signal = "bullish"  # Contre-tendance
            elif value <= 40:
                interpretation = "Fear"
                signal = "slightly_bullish"
            elif value <= 60:
                interpretation = "Neutral"
                signal = "neutral"
            elif value <= 75:
                interpretation = "Greed"
                signal = "slightly_bearish"
            else:
                interpretation = "Extreme Greed"
                signal = "bearish"  # Contre-tendance

            result = {
                "value": value,
                "label": current.get("value_classification", "Unknown"),
                "interpretation": interpretation,
                "signal": signal,
                "timestamp": datetime.now().isoformat(),
                "update_time": current.get("timestamp")
            }

            self._set_cache("fear_greed", result)
            return result

        except Exception as e:
            print(f"[Sentiment] Erreur Fear & Greed: {e}")
            return None

    def get_news(self, max_items: int = 10) -> Optional[List[Dict]]:
        """Recupere les dernieres news crypto"""
        cached = self._get_cached("news")
        if cached:
            return cached

        all_news = []

        for feed_config in self.news_feeds:
            try:
                feed = feedparser.parse(feed_config["url"])

                for entry in feed.entries[:5]:  # 5 par source
                    # Parser la date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6]).isoformat()

                    # Analyser le sentiment du titre
                    sentiment = self._analyze_title_sentiment(entry.title)

                    all_news.append({
                        "title": entry.title,
                        "source": feed_config["name"],
                        "category": feed_config["category"],
                        "link": entry.link,
                        "published": pub_date,
                        "sentiment": sentiment
                    })

            except Exception as e:
                print(f"[Sentiment] Erreur RSS {feed_config['name']}: {e}")

        # Trier par date
        all_news.sort(key=lambda x: x.get("published") or "", reverse=True)

        result = all_news[:max_items]
        self._set_cache("news", result)
        return result

    def _analyze_title_sentiment(self, title: str) -> Dict:
        """Analyse basique du sentiment d'un titre"""
        title_lower = title.lower()

        # Mots positifs
        bullish_words = [
            "surge", "soar", "rally", "bull", "gain", "rise", "up", "high",
            "record", "ath", "breakout", "pump", "moon", "bullish", "buy",
            "adoption", "institutional", "etf approved", "halving"
        ]

        # Mots negatifs
        bearish_words = [
            "crash", "drop", "fall", "bear", "loss", "down", "low", "dump",
            "plunge", "sell", "fear", "panic", "hack", "scam", "ban",
            "regulation", "sec", "lawsuit", "bearish", "correction"
        ]

        # Compter
        bullish_count = sum(1 for word in bullish_words if word in title_lower)
        bearish_count = sum(1 for word in bearish_words if word in title_lower)

        # Determiner le sentiment
        if bullish_count > bearish_count:
            return {"score": bullish_count - bearish_count, "label": "bullish"}
        elif bearish_count > bullish_count:
            return {"score": bearish_count - bullish_count, "label": "bearish"}
        else:
            return {"score": 0, "label": "neutral"}

    def get_news_sentiment_aggregate(self) -> Optional[Dict]:
        """Calcule le sentiment agrege des news"""
        news = self.get_news(max_items=20)

        if not news:
            return None

        bullish = 0
        bearish = 0
        neutral = 0

        for article in news:
            sentiment = article.get("sentiment", {})
            label = sentiment.get("label", "neutral")

            if label == "bullish":
                bullish += 1
            elif label == "bearish":
                bearish += 1
            else:
                neutral += 1

        total = bullish + bearish + neutral
        if total == 0:
            return None

        # Score de -100 (tres bearish) a +100 (tres bullish)
        score = ((bullish - bearish) / total) * 100

        return {
            "score": round(score, 1),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "total_articles": total,
            "bias": "bullish" if score > 20 else "bearish" if score < -20 else "neutral",
            "timestamp": datetime.now().isoformat()
        }

    def get_btc_dominance(self) -> Optional[float]:
        """Recupere la dominance du BTC depuis CoinGecko"""
        cached = self._get_cached("btc_dominance")
        if cached:
            return cached.get("dominance")

        try:
            response = requests.get(self.coingecko_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                dominance = data.get("data", {}).get("market_cap_percentage", {}).get("btc", 0)
                if dominance > 0:
                    self._set_cache("btc_dominance", {"dominance": round(dominance, 2)})
                    return round(dominance, 2)
        except Exception as e:
            print(f"[Sentiment] Erreur CoinGecko: {e}")
        return None

    def get_ai_news_analysis(self) -> Optional[Dict]:
        """Analyse la psychologie de masse via l'IA sur les derniers titres"""
        news = self.get_news(max_items=15)
        if not news:
            return None

        titles = "\n".join([f"- {a['title']}" for a in news])
        
        prompt = f"""
        En tant qu'expert en psychologie des marchés crypto, analyse ces titres de news récents :
        {titles}
        
        Détermine :
        1. Le score de sentiment global (-100 pour panique totale, +100 pour euphorie extrême).
        2. Le biais dominant (bullish, bearish, neutral).
        3. La présence de FUD (peur/incertitude) ou FOMO.
        
        Réponds UNIQUEMENT au format JSON comme ceci :
        {{"score": 25, "bias": "bullish", "psychology": "FOMO léger", "reason": "courte explication"}}
        """

        try:
            # Importer dynamiquement pour eviter les cycles
            from agents.simple_agent import SimpleAgent
            # Utiliser un agent temporaire pour l'appel AI
            dummy_agent = SimpleAgent("news_analyzer")
            response_text = dummy_agent.call_ai(prompt)
            
            if not response_text:
                return None
            
            # Extraire le JSON (parfois l'IA ajoute du texte avant/apres)
            match = re.search(r'({.*})', response_text, re.DOTALL)
            if match:
                result = json.loads(match.group(1))
                self._set_cache("ai_news_sentiment", result)
                return result
        except Exception as e:
            print(f"[Sentiment] Erreur analyse IA: {e}")
        return None

    def get_all_sentiment(self) -> Dict:
        """Recupere toutes les donnees de sentiment incluant l'IA"""
        fear_greed = self.get_fear_greed_index()
        news = self.get_news()
        
        # Priorite a l'analyse IA si disponible (cache de 15 min)
        ai_sentiment = self._get_cached("ai_news_sentiment", duration=self.cache_duration_ai)
        if not ai_sentiment:
            ai_sentiment = self.get_ai_news_analysis()

        news_sentiment = self.get_news_sentiment_aggregate()

        # Score global combine
        global_score = 50

        if fear_greed:
            global_score = fear_greed["value"]

        # Utiliser le score IA pour l'ajustement si present
        if ai_sentiment:
            ai_adjustment = ai_sentiment["score"] / 5 # +-20
            global_score = max(0, min(100, global_score + ai_adjustment))
        elif news_sentiment:
            news_adjustment = news_sentiment["score"] / 5
            global_score = max(0, min(100, global_score + news_adjustment))

        btc_dom = self.get_btc_dominance()

        return {
            "fear_greed": fear_greed,
            "news": news,
            "news_sentiment": news_sentiment,
            "ai_sentiment": ai_sentiment,
            "btc_dominance": btc_dom,
            "global_score": round(global_score, 1),
            "global_bias": "bullish" if global_score > 60 else "bearish" if global_score < 40 else "neutral"
        }


# Singleton
_sentiment_instance = None

def get_sentiment() -> SentimentData:
    """Retourne l'instance Sentiment singleton"""
    global _sentiment_instance
    if _sentiment_instance is None:
        _sentiment_instance = SentimentData()
    return _sentiment_instance
