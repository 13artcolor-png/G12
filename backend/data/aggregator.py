# -*- coding: utf-8 -*-
"""
G12 - Aggregateur de donnees
Combine toutes les sources en un contexte unifie
"""

from datetime import datetime, time as dt_time
from typing import Optional, Dict, List
from pathlib import Path
import sys
import numpy as np
import json
sys.path.append('..')

from data.binance_data import get_binance
from data.sentiment import get_sentiment
from core.mt5_connector import get_mt5
from config import SESSIONS
import time

# Import du detecteur de patterns institutionnels
try:
    from institutional_patterns import InstitutionalPatternDetector, format_for_ai_prompt
    INSTITUTIONAL_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_AVAILABLE = False
    print("[Aggregator] Module institutional_patterns non disponible")

from data.news_filter import get_news_filter
from data.macro_engine import get_macro_engine
from data.whale_tracker import get_whale_tracker

# Chemin du fichier de configuration des poids
DATABASE_DIR = Path(__file__).parent.parent / "database"
WEIGHTS_CONFIG_FILE = DATABASE_DIR / "analysis_weights_config.json"


class DataAggregator:
    """Agregge toutes les donnees en un contexte unifie pour les agents"""

    def __init__(self):
        # Utiliser le compte momentum pour les donnees de prix (tous les comptes ont le meme prix)
        self.mt5 = get_mt5("fibo1")
        self.binance = get_binance()
        self.sentiment = get_sentiment()
        self.news_filter = get_news_filter()
        self.macro_engine = get_macro_engine()
        self.whale_tracker = get_whale_tracker()
        self.last_context = None

        # Charger les poids d'analyse (modifiables par Strategist)
        self.analysis_weights = self._load_analysis_weights()

        # Detecteur de patterns institutionnels
        if INSTITUTIONAL_AVAILABLE:
            self.pattern_detector = InstitutionalPatternDetector(swing_lookback=5, min_swing_size=0.0003)
            print("[Aggregator] Detecteur de patterns institutionnels active")
        else:
            self.pattern_detector = None

    def _load_analysis_weights(self) -> Dict:
        """Charge les poids de ponderation pour l'analyse consensus"""
        try:
            if WEIGHTS_CONFIG_FILE.exists():
                with open(WEIGHTS_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    weights = config.get('weights', {})
                    print(f"[Aggregator] Poids d'analyse charges: {weights}")
                    return weights
        except Exception as e:
            print(f"[Aggregator] Erreur chargement poids analyse: {e}")

        # Poids par defaut si erreur
        default_weights = {
            "price_momentum": 20,
            "macro": 25,
            "whales": 20,
            "sentiment": 15,
            "futures": 15,
            "btc_dominance": 5
        }
        print(f"[Aggregator] Utilisation poids par defaut: {default_weights}")
        return default_weights

    def get_session_status(self) -> Dict:
        """Détermine l'état de toutes les sessions (Paris time)"""
        now = datetime.now()
        current_time = now.time()
        
        status = {
            "active_session": None,
            "sessions": {}
        }

        for session_id, session in SESSIONS.items():
            start = dt_time.fromisoformat(session["start"])
            end = dt_time.fromisoformat(session["end"])

            is_active = False
            if start <= end:
                if start <= current_time <= end:
                    is_active = True
            else:
                if current_time >= start or current_time <= end:
                    is_active = True
            
            status["sessions"][session_id] = {
                "name": session["name"],
                "active": is_active,
                "volatility": session["volatility"]
            }
            
            if is_active:
                status["active_session"] = session_id
                # Copie des infos pour la compatibilité
                status["id"] = session_id
                status["name"] = session["name"]
                status["volatility"] = session["volatility"]

        if not status["active_session"]:
            status["id"] = "unknown"
            status["name"] = "Unknown"
            status["volatility"] = "medium"

        return status

    def get_current_session(self) -> Dict:
        """Determine la session actuelle (Paris time)"""
        return self.get_session_status()

    def get_price_data(self) -> Optional[Dict]:
        """Recupere les donnees de prix MT5"""
        symbol_info = self.mt5.get_symbol_info()
        if not symbol_info:
            return None

        # Calculer les momentums
        momentum_1m = self.mt5.calculate_momentum("1m", 5)
        momentum_5m = self.mt5.calculate_momentum("5m", 5)
        momentum_15m = self.mt5.calculate_momentum("15m", 5)
        momentum_1h = self.mt5.calculate_momentum("1h", 5)

        # Volatilite
        volatility = self.mt5.calculate_volatility("1h", 24)

        # Derniere bougie
        candles_1m = self.mt5.get_candles("1m", 1)
        last_candle = candles_1m[-1] if candles_1m else None

        return {
            "price": (symbol_info["bid"] + symbol_info["ask"]) / 2,
            "bid": symbol_info["bid"],
            "ask": symbol_info["ask"],
            "spread_points": symbol_info["spread_points"],
            "momentum": {
                "1m": momentum_1m,
                "5m": momentum_5m,
                "15m": momentum_15m,
                "1h": momentum_1h
            },
            "volatility_pct": volatility,
            "last_candle": last_candle,
            "timestamp": symbol_info["time"]
        }

    def get_institutional_analysis(self, timeframe: str = "15m", candle_count: int = 100) -> Optional[Dict]:
        """
        Analyse les patterns institutionnels sur le timeframe specifie

        Args:
            timeframe: "1m", "5m", "15m", "1h"
            candle_count: Nombre de bougies a analyser

        Returns:
            Dict avec patterns detectes, structure de marche, zones de liquidite
        """
        if not self.pattern_detector:
            return None

        try:
            # Recuperer les bougies
            candles = self.mt5.get_candles(timeframe, candle_count)
            if not candles or len(candles) < 20:
                return None

            # Convertir en arrays numpy
            highs = np.array([c["high"] for c in candles])
            lows = np.array([c["low"] for c in candles])
            closes = np.array([c["close"] for c in candles])

            # Analyser les patterns
            analysis = self.pattern_detector.analyze(highs, lows, closes)

            # Ajouter le timeframe utilise
            analysis["timeframe"] = timeframe
            analysis["candle_count"] = len(candles)

            return analysis

        except Exception as e:
            print(f"[Aggregator] Erreur analyse institutionnelle: {e}")
            return None

    def get_account_data(self) -> Optional[Dict]:
        """Recupere les donnees de TOUS les comptes MT5 (multi-agents)"""
        # Agregger les donnees de tous les comptes
        all_positions = []
        total_balance = 0
        total_equity = 0
        total_margin_free = 0
        total_profit = 0
        accounts_data = {}
        floating_pnl_total = 0

        for agent_id in ["fibo1", "fibo2", "fibo3"]:
            try:
                agent_mt5 = get_mt5(agent_id)
                connected = agent_mt5.connect()
                if connected:
                    # Delai minimal pour la synchronisation MT5
                    time.sleep(0.2)

                    account = agent_mt5.get_account_info()
                    if account:
                        total_balance += account.get("balance", 0)
                        total_equity += account.get("equity", 0)
                        total_margin_free += account.get("margin_free", 0)
                        total_profit += account.get("profit", 0)
                        accounts_data[agent_id] = account

                    # Recuperer les positions de ce compte
                    positions = agent_mt5.get_positions()
                    agent_floating_pnl = sum(pos.get("profit", 0) for pos in positions)
                    floating_pnl_total += agent_floating_pnl
                    print(f"[Aggregator] {agent_id}: {len(positions)} positions, P&L flottant: {agent_floating_pnl:.2f} EUR")

                    for pos in positions:
                        pos["_agent_id"] = agent_id  # Marquer l'agent proprietaire
                        all_positions.append(pos)
                else:
                    print(f"[Aggregator] WARN: {agent_id} - connexion MT5 echouee")
            except Exception as e:
                print(f"[Aggregator] Erreur recuperation compte {agent_id}: {e}")
                import traceback
                traceback.print_exc()

        if not accounts_data:
            return None

        # P&L flottant = somme des profits des positions (comme affiche dans MT5)
        print(f"[Aggregator] TOTAL: {len(all_positions)} positions")
        print(f"[Aggregator] P&L flottant: {floating_pnl_total:.2f} EUR")
        print(f"[Aggregator] Balance: {total_balance:.2f} EUR, Equity: {total_equity:.2f} EUR")

        return {
            "balance": total_balance,
            "equity": total_equity,
            "margin_free": total_margin_free,
            "profit": total_profit,
            "floating_pnl": floating_pnl_total,  # P&L flottant = somme des profits positions (comme MT5)
            "positions": all_positions,
            "position_count": len(all_positions),
            "accounts": accounts_data  # Details par compte
        }

    def get_full_context(self) -> Dict:
        """Construit le contexte complet pour les agents"""
        now = datetime.now()

        # Donnees MT5
        price_data = self.get_price_data()
        account_data = self.get_account_data()

        # Donnees Binance (fallback dict vide si API indisponible)
        binance_data = self.binance.get_all_data() or {}

        # Donnees Sentiment (fallback dict vide si API indisponible)
        sentiment_data = self.sentiment.get_all_sentiment() or {}

        # Session actuelle
        session = self.get_current_session()

        # Analyse institutionnelle (patterns, structure, liquidite)
        institutional_data = self.get_institutional_analysis("15m", 100)

        # Construire le contexte
        context = {
            "timestamp": now.isoformat(),
            "symbol": "BTCUSD",

            # Session
            "session": session,

            # Prix et technique
            "price": price_data,

            # Compte
            "account": account_data,

            # Binance Futures
            "futures": {
                "funding_rate": binance_data["funding"]["funding_rate"] if binance_data.get("funding") else None,
                "funding_signal": self._interpret_funding(binance_data.get("funding")),
                "open_interest": binance_data["open_interest"]["open_interest"] if binance_data.get("open_interest") else None,
                "oi_change_1h": binance_data["open_interest"]["change_1h_pct"] if binance_data.get("open_interest") else None,
                "long_short_ratio": binance_data["long_short_ratio"]["long_short_ratio"] if binance_data.get("long_short_ratio") else None,
                "orderbook_imbalance": binance_data["orderbook"]["imbalance_pct"] if binance_data.get("orderbook") else None,
                "orderbook_bias": binance_data["orderbook"]["bias"] if binance_data.get("orderbook") else None
            },

            # Sentiment
            "sentiment": {
                "fear_greed_index": sentiment_data["fear_greed"]["value"] if sentiment_data.get("fear_greed") else None,
                "fear_greed_label": sentiment_data["fear_greed"]["label"] if sentiment_data.get("fear_greed") else None,
                "news_score": sentiment_data["news_sentiment"]["score"] if sentiment_data.get("news_sentiment") else None,
                "news_bias": sentiment_data["news_sentiment"]["bias"] if sentiment_data.get("news_sentiment") else None,
                "global_score": sentiment_data.get("global_score"),
                "global_bias": sentiment_data.get("global_bias")
            },

            # News recentes
            "recent_news": sentiment_data.get("news", [])[:5],
            
            # News a haut impact
            "high_impact_news": self.news_filter.get_high_impact_events(),
            "pause_recommended": self.news_filter.should_pause_trading(),
            
            # Données Macro
            "macro": self.macro_engine.get_macro_data(),
            
            # Whale Tracker
            "whales": self.whale_tracker.get_whale_bias(),
            "whale_list": self.whale_tracker.cache,

            # Analyse institutionnelle (Price Action)
            "institutional": institutional_data,

            # Analyse globale
            "analysis": self._generate_analysis(price_data, binance_data, sentiment_data, session, institutional_data)
        }

        self.last_context = context
        return context

    def _interpret_funding(self, funding_data: Optional[Dict]) -> Optional[str]:
        """Interprete le funding rate"""
        if not funding_data:
            return None

        rate = funding_data.get("funding_rate", 0)

        if rate > 0.05:
            return "very_bullish_crowd"  # Beaucoup de longs = risque de squeeze
        elif rate > 0.01:
            return "bullish_crowd"
        elif rate < -0.05:
            return "very_bearish_crowd"  # Beaucoup de shorts = risque de squeeze
        elif rate < -0.01:
            return "bearish_crowd"
        else:
            return "neutral"

    def _generate_analysis(self, price_data: Optional[Dict], binance_data: Dict,
                          sentiment_data: Dict, session: Dict,
                          institutional_data: Optional[Dict] = None) -> Dict:
        """Genere une analyse globale basee sur un consensus pondere Multi-AI

        Utilise les poids dynamiques depuis analysis_weights_config.json
        Ces poids peuvent etre modifies par le Strategist pour optimiser l'analyse
        """
        scores = {"bullish": 0, "bearish": 0}
        reasons = []

        # Charger les poids dynamiques
        w = self.analysis_weights

        # 1. PRICE MOMENTUM - PRIORITE MAXIMALE
        # Analyse le momentum reel du prix (Mom 1m, 3m, 5m)
        weight_momentum = w.get('price_momentum', 20)
        if price_data:
            mom_1m = price_data.get('fibo1', {}).get('1m', 0) or 0
            mom_3m = price_data.get('fibo1', {}).get('3m', 0) or 0

            # Moyenne ponderee: 1m (60%), 3m (40%)
            avg_momentum = (mom_1m * 0.6) + (mom_3m * 0.4)

            # Seuils de momentum
            if avg_momentum > 0.05:  # +0.05% = bullish fort
                scores['bullish'] += weight_momentum
                reasons.append(f"Price Momentum BULLISH (1m: {mom_1m:+.2f}%, 3m: {mom_3m:+.2f}%)")
            elif avg_momentum > 0.01:  # +0.01% = bullish modere
                scores['bullish'] += int(weight_momentum * 0.75)
                reasons.append(f"Price Momentum Bullish (1m: {mom_1m:+.2f}%, 3m: {mom_3m:+.2f}%)")
            elif avg_momentum < -0.05:  # -0.05% = bearish fort
                scores['bearish'] += weight_momentum
                reasons.append(f"Price Momentum BEARISH (1m: {mom_1m:+.2f}%, 3m: {mom_3m:+.2f}%)")
            elif avg_momentum < -0.01:  # -0.01% = bearish modere
                scores['bearish'] += int(weight_momentum * 0.75)
                reasons.append(f"Price Momentum Bearish (1m: {mom_1m:+.2f}%, 3m: {mom_3m:+.2f}%)")
            else:
                # Neutral - ne pas ajouter de reason
                pass

        # 2. MACRO ENGINE
        weight_macro = w.get('macro', 25)
        macro = self.macro_engine.get_macro_data()
        macro_bias = macro.get('bias', 'neutral')
        if macro_bias == 'bullish':
            scores['bullish'] += weight_macro
            reasons.append(f"Macro Bullish (DXY/SPX correlation: {macro.get('correlation', 0):.2f})")
        elif macro_bias == 'bearish':
            scores['bearish'] += weight_macro
            reasons.append(f"Macro Bearish (DXY/SPX correlation: {macro.get('correlation', 0):.2f})")

        # 3. WHALE TRACKER
        weight_whales = w.get('whales', 20)
        whales = self.whale_tracker.get_whale_bias()
        whale_bias = whales.get('bias', 'neutral')
        if whale_bias == 'bullish':
            scores['bullish'] += weight_whales
            reasons.append(f"Whales Bullish: {whales.get('reason')}")
        elif whale_bias == 'bearish':
            scores['bearish'] += weight_whales
            reasons.append(f"Whales Bearish: {whales.get('reason')}")

        # 4. SENTIMENT
        weight_sentiment = w.get('sentiment', 15)
        global_bias = sentiment_data.get('global_bias', 'neutral')
        if global_bias == 'bullish':
            scores['bullish'] += weight_sentiment
            reasons.append(f"Sentiment Bullish (Fear&Greed: {sentiment_data.get('fear_greed_index')})")
        elif global_bias == 'bearish':
            scores['bearish'] += weight_sentiment
            reasons.append(f"Sentiment Bearish (Fear&Greed: {sentiment_data.get('fear_greed_index')})")

        # 5. FUTURES & ORDERBOOK
        weight_futures = w.get('futures', 15)
        # Logic combinee funding + orderbook imbalance
        funding_rate = binance_data.get("funding", {}).get("funding_rate", 0)
        imbalance = binance_data.get("orderbook", {}).get("imbalance_pct", 0)

        # Bullish signal
        if funding_rate < 0 or imbalance > 10:
            scores['bullish'] += weight_futures
            reasons.append(f"Futures/Orderbook Bullish (Funding: {funding_rate:.4f}%, Imb: {imbalance:.1f}%)")
        # Bearish signal
        elif funding_rate > 0.05 or imbalance < -10:
            scores['bearish'] += weight_futures
            reasons.append(f"Futures/Orderbook Bearish (Funding: {funding_rate:.4f}%, Imb: {imbalance:.1f}%)")

        # 6. BTC DOMINANCE
        weight_btc_dom = w.get('btc_dominance', 5)
        btc_dom = sentiment_data.get('btc_dominance', 50)
        if btc_dom and btc_dom > 55:
            if macro_bias == 'bullish' or global_bias == 'bullish':
                scores['bullish'] += weight_btc_dom
                reasons.append(f"BTC Dominance Bullish ({btc_dom}%) - Strong market absorption")
            else:
                scores['bearish'] += weight_btc_dom  # Dominance high in red market = BTC bleed
                reasons.append(f"BTC Dominance Bearish context ({btc_dom}%)")

        # Calcul du biais final
        total_bullish = scores['bullish']
        total_bearish = scores['bearish']
        
        if total_bullish > total_bearish + 10:
            bias = "bullish"
            confidence = total_bullish
        elif total_bearish > total_bullish + 10:
            bias = "bearish"
            confidence = total_bearish
        else:
            bias = "neutral"
            confidence = 50

        # Normalisation confiance (max 100)
        confidence = min(100, confidence)

        return {
            "bias": bias,
            "confidence": round(confidence, 1),
            "signals": scores,
            "reasons": reasons,
            "session_volatility": session.get("volatility", "medium"),
            "recommended_action": self._recommend_action(bias, confidence, session)
        }

    def _recommend_action(self, bias: str, confidence: float, session: Dict) -> str:
        """Recommande une action basee sur l'analyse"""
        volatility = session.get("volatility", "medium")

        if confidence < 40:
            return "HOLD - Signaux mixtes, attendre confirmation"

        if volatility == "low" and bias != "neutral":
            return f"CAUTION - Volatilite basse, signaux {bias} mais prudence"

        if confidence >= 70:
            if bias == "bullish":
                return "STRONG BUY - Signaux fortement bullish"
            elif bias == "bearish":
                return "STRONG SELL - Signaux fortement bearish"

        if confidence >= 50:
            if bias == "bullish":
                return "BUY - Signaux moderement bullish"
            elif bias == "bearish":
                return "SELL - Signaux moderement bearish"

        return "HOLD - Pas de signal clair"

    def reload_analysis_weights(self):
        """Recharge les poids d'analyse depuis le fichier (appele par Strategist apres modification)"""
        self.analysis_weights = self._load_analysis_weights()
        print("[Aggregator] Poids d'analyse recharges")

    def format_context_for_prompt(self, context: Dict = None) -> str:
        """Formate le contexte pour l'envoyer aux agents IA"""
        if context is None:
            context = self.get_full_context()

        price = context.get("price", {})
        futures = context.get("futures", {})
        sentiment = context.get("sentiment", {})
        analysis = context.get("analysis", {})
        session = context.get("session", {})
        account = context.get("account", {})
        institutional = context.get("institutional", {})

        prompt = f"""
=== CONTEXTE BTCUSD ===
Timestamp: {context.get('timestamp')}
Session: {session.get('name')} (Volatilite: {session.get('volatility')})

=== PRIX & TECHNIQUE ===
Prix: ${price.get('price', 'N/A'):,.2f}
Spread: {price.get('spread_points', 'N/A')} points
Momentum 1min: {price.get('momentum', {}).get('1m', 'N/A')}%
Momentum 5min: {price.get('momentum', {}).get('5m', 'N/A')}%
Momentum 15min: {price.get('momentum', {}).get('15m', 'N/A')}%
Momentum 1h: {price.get('momentum', {}).get('1h', 'N/A')}%
Volatilite: {price.get('volatility_pct', 'N/A')}%

=== BINANCE FUTURES ===
Funding Rate: {futures.get('funding_rate', 'N/A')}%
Signal Funding: {futures.get('funding_signal', 'N/A')}
Open Interest Change 1h: {futures.get('oi_change_1h', 'N/A')}%
Long/Short Ratio: {futures.get('long_short_ratio', 'N/A')}
Orderbook Imbalance: {futures.get('orderbook_imbalance', 'N/A')}%
Orderbook Bias: {futures.get('orderbook_bias', 'N/A')}

=== SENTIMENT ===
Fear & Greed Index: {sentiment.get('fear_greed_index', 'N/A')} ({sentiment.get('fear_greed_label', 'N/A')})
News Score: {sentiment.get('news_score', 'N/A')}
Global Bias: {sentiment.get('global_bias', 'N/A')}

=== COMPTE ===
Balance: {account.get('balance', 'N/A')} EUR
Equity: {account.get('equity', 'N/A')} EUR
Positions ouvertes: {account.get('position_count', 0)}
"""

        # Ajouter l'analyse institutionnelle si disponible
        if institutional and not institutional.get("error"):
            ms = institutional.get("market_structure", {})
            patterns = institutional.get("patterns_detected", [])
            rec = institutional.get("recommendation", {})
            liquidity = institutional.get("liquidity_zones", [])

            prompt += f"""
=== ANALYSE INSTITUTIONNELLE (Price Action) ===
Timeframe: {institutional.get('timeframe', 'N/A')}
Structure: {ms.get('trend', 'N/A')} (HH:{ms.get('hh_count', 0)} HL:{ms.get('hl_count', 0)} LH:{ms.get('lh_count', 0)} LL:{ms.get('ll_count', 0)})
"""
            if patterns:
                prompt += "Patterns detectes:\n"
                for p in patterns:
                    prompt += f"  - {p.get('type')}: {p.get('description')} (confiance: {p.get('confidence', 0)*100:.0f}%)\n"
                    prompt += f"    Entry: {p.get('entry_zone', [0,0])[0]:.2f}-{p.get('entry_zone', [0,0])[1]:.2f} | SL: {p.get('stop_loss', 0):.2f} | TP: {p.get('take_profit', 0):.2f}\n"
            else:
                prompt += "Patterns detectes: Aucun\n"

            if liquidity:
                prompt += "Zones de liquidite proches:\n"
                for lz in liquidity[:3]:
                    prompt += f"  - {lz.get('type')} @ {lz.get('level', 0):.2f} ({lz.get('distance_pct', 0):+.2f}%)\n"

            if rec.get("action"):
                prompt += f"Recommandation institutionnelle: {rec.get('action')} (confiance: {rec.get('confidence', 0)*100:.0f}%)\n"
                if rec.get("risk_reward"):
                    prompt += f"Risk/Reward: {rec.get('risk_reward', 0):.2f}\n"

        prompt += f"""
=== ANALYSE GLOBALE ===
Bias: {analysis.get('bias', 'N/A')}
Confiance: {analysis.get('confidence', 0)}%
Raisons: {', '.join(analysis.get('reasons', []))}
Recommandation: {analysis.get('recommended_action', 'N/A')}
"""
        return prompt


# Singleton
_aggregator_instance = None

def get_aggregator() -> DataAggregator:
    """Retourne l'instance DataAggregator singleton"""
    global _aggregator_instance
    if _aggregator_instance is None:
        _aggregator_instance = DataAggregator()
    return _aggregator_instance
