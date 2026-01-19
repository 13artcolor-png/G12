# -*- coding: utf-8 -*-
"""
G12 - Agent FIBO2
Trade sur les niveaux de Fibonacci + patterns ICT/SMC institutionnels
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from agents.base_agent import BaseAgent


class Fibo2Agent(BaseAgent):
    """Agent qui trade sur les niveaux Fibonacci avec confirmation ICT/SMC"""

    # Niveaux Fibonacci standard
    FIBO2_LEVELS = {
        "0.236": 0.236,
        "0.382": 0.382,
        "0.5": 0.5,
        "0.618": 0.618,
        "0.786": 0.786
    }

    def __init__(self):
        super().__init__("fibo2")

    def _get_higher_timeframe(self) -> str:
        """Retourne la timeframe superieure pour le calcul Fibo"""
        # PRIORITE: Utiliser higher_timeframe depuis la config si defini
        higher_tf = self.config.get("higher_timeframe")
        
        if higher_tf:
            # Convertir le format MT5 (M15, M30, H1) vers format Binance (15m, 30m, 1h)
            tf_conversion = {
                "M1": "1m",
                "M5": "5m",
                "M15": "15m",
                "M30": "30m",
                "H1": "1h",
                "H4": "4h",
                "D1": "1d"
            }
            return tf_conversion.get(higher_tf, higher_tf.lower())
        
        # FALLBACK: Mapping automatique si higher_timeframe non defini
        signal_tf = self.config.get("signal_timeframe", "M15")
        tf_map = {
            "M1": "15m",   # M1 -> M15
            "M5": "1h",    # M5 -> H1
            "M15": "4h",   # M15 -> H4
            "M30": "4h",   # M30 -> H4
            "H1": "1d",    # H1 -> D1
            "H4": "1d",    # H4 -> D1
        }
        return tf_map.get(signal_tf, "4h")

    def _calculate_fibo_levels(self, swing_high: float, swing_low: float,
                                is_uptrend: bool) -> Dict[str, float]:
        """
        Calcule les niveaux de Fibonacci entre le swing high et swing low

        Args:
            swing_high: Point haut du swing
            swing_low: Point bas du swing
            is_uptrend: True si tendance haussiere (retracement depuis le haut)

        Returns:
            Dict avec les niveaux Fibo calcules
        """
        range_size = swing_high - swing_low

        levels = {}
        for name, ratio in self.FIBO2_LEVELS.items():
            if is_uptrend:
                # Retracement depuis le haut: on part du high et on descend
                levels[name] = swing_high - (range_size * ratio)
            else:
                # Extension depuis le bas: on part du low et on monte
                levels[name] = swing_low + (range_size * ratio)

        levels["swing_high"] = swing_high
        levels["swing_low"] = swing_low
        levels["range"] = range_size
        levels["is_uptrend"] = is_uptrend

        return levels

    def _get_swing_points(self, candles: List[Dict]) -> Tuple[float, float, bool]:
        """
        Identifie les points swing haut et bas sur les bougies

        Returns:
            (swing_high, swing_low, is_uptrend)
        """
        if not candles or len(candles) < 10:
            return None, None, None

        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        # Trouver le plus haut et le plus bas
        swing_high = max(highs)
        swing_low = min(lows)

        swing_high_idx = highs.index(swing_high)
        swing_low_idx = lows.index(swing_low)

        # Determiner la tendance: si le high est plus recent, on est en uptrend
        is_uptrend = swing_high_idx > swing_low_idx

        return swing_high, swing_low, is_uptrend

    def _get_institutional_analysis(self, candles: List[Dict]) -> Optional[Dict]:
        """Analyse les patterns institutionnels avec institutional_patterns.py"""
        try:
            import sys
            sys.path.insert(0, '..')
            from institutional_patterns import InstitutionalPatternDetector, format_for_ai_prompt

            if not candles or len(candles) < 20:
                return None

            highs = np.array([c["high"] for c in candles])
            lows = np.array([c["low"] for c in candles])
            closes = np.array([c["close"] for c in candles])

            detector = InstitutionalPatternDetector(swing_lookback=3)
            analysis = detector.analyze(highs, lows, closes)

            return analysis

        except Exception as e:
            print(f"[FIBO2] Erreur analyse institutionnelle: {e}")
            return None

    def _check_price_near_fibo_level(self, current_price: float,
                                      fibo_levels: Dict,
                                      tolerance_pct: float = 0.3) -> Tuple[bool, str, str]:
        """
        Verifie si le prix est proche d'un niveau Fibonacci

        Returns:
            (is_near, level_name, signal_type)
        """
        target_level = self.config.get("fibo_level", "0.618")

        if target_level not in fibo_levels:
            return False, "", ""

        level_price = fibo_levels[target_level]
        tolerance = current_price * (tolerance_pct / 100)

        if abs(current_price - level_price) <= tolerance:
            is_uptrend = fibo_levels.get("is_uptrend", True)

            # En uptrend avec retracement: on achete au niveau Fibo
            # En downtrend avec rebond: on vend au niveau Fibo
            if is_uptrend:
                signal = "BUY"  # Acheter sur le retracement en uptrend
            else:
                signal = "SELL"  # Vendre sur le rebond en downtrend

            return True, target_level, signal

        return False, "", ""

    def should_consider_trade(self, context: Dict) -> Tuple[bool, str]:
        """Verifie si les conditions Fibonacci + ICT sont reunies"""
        price_data = context.get("price") or {}

        # Verifier le spread
        from agents.base_agent import load_spread_runtime_config
        spread_config = load_spread_runtime_config()
        if spread_config.get("spread_check_enabled", True):
            spread = price_data.get("spread_points", 999)
            max_spread = spread_config.get("max_spread_points", 2000)
            if spread > max_spread:
                return False, f"Spread trop eleve: {spread} > {max_spread} points"

        current_price = price_data.get("price", 0)
        if current_price == 0:
            return False, "Prix non disponible"

        # Recuperer les bougies de la timeframe superieure
        try:
            from core.mt5_connector import get_mt5
            mt5 = get_mt5("fibo2")
            higher_tf = self._get_higher_timeframe()
            candles = mt5.get_candles(higher_tf, count=50)

            if not candles or len(candles) < 20:
                return False, f"Pas assez de bougies sur {higher_tf}"

        except Exception as e:
            return False, f"Erreur recuperation bougies: {e}"

        # Calculer les swings et niveaux Fibo
        swing_high, swing_low, is_uptrend = self._get_swing_points(candles)

        if swing_high is None or swing_low is None:
            return False, "Impossible de detecter les swings"

        fibo_levels = self._calculate_fibo_levels(swing_high, swing_low, is_uptrend)

        # Stocker pour le prompt
        context["fibo_levels"] = fibo_levels
        context["higher_tf"] = higher_tf

        # Verifier si prix proche d'un niveau Fibo
        tolerance = self.config.get("fibo_tolerance_pct", 0.3)
        is_near_fibo, level_name, signal = self._check_price_near_fibo_level(
            current_price, fibo_levels, tolerance
        )

        if not is_near_fibo:
            target = self.config.get("fibo_level", "0.618")
            target_price = fibo_levels.get(target, 0)
            distance = abs(current_price - target_price) / current_price * 100
            return False, f"Prix loin du niveau Fibo {target} ({target_price:.2f}), distance={distance:.2f}%"

        # Analyse institutionnelle (ICT/SMC patterns)
        inst_analysis = self._get_institutional_analysis(candles)
        if inst_analysis:
            context["institutional"] = inst_analysis

            # Bonus: pattern institutionnel confirme le signal?
            rec = inst_analysis.get("recommendation", {})
            inst_action = rec.get("action", "HOLD")
            inst_confidence = rec.get("confidence", 0)

            if inst_action != "HOLD" and inst_confidence >= 0.6:
                if inst_action == signal:
                    return True, f"{signal} sur Fibo {level_name} + confirmation ICT ({rec.get('reason', '')})"

        # Meme sans confirmation ICT, on peut trader si proche du niveau Fibo
        trend_str = "Uptrend (retracement)" if is_uptrend else "Downtrend (rebond)"
        return True, f"{signal} sur Fibo {level_name} ({trend_str})"

    def get_opener_prompt(self, context: Dict) -> str:
        """Prompt d'ouverture specifique a Fibonacci + ICT"""
        price_data = context.get("price") or {}
        momentum = price_data.get("momentum") or {}
        futures = context.get("futures") or {}
        sentiment = context.get("sentiment") or {}
        session = context.get("session") or {}

        fibo_levels = context.get("fibo_levels", {})
        higher_tf = context.get("higher_tf", "4h")
        institutional = context.get("institutional", {})

        # Construire la section Fibonacci
        fibo_section = ""
        if fibo_levels:
            target_level = self.config.get("fibo_level", "0.618")
            is_uptrend = fibo_levels.get("is_uptrend", True)

            fibo_section = f"""
NIVEAUX FIBO2NACCI ({higher_tf}):
- Swing High: ${fibo_levels.get('swing_high', 0):,.2f}
- Swing Low: ${fibo_levels.get('swing_low', 0):,.2f}
- Range: ${fibo_levels.get('range', 0):,.2f}
- Tendance: {"HAUSSIERE (retracement)" if is_uptrend else "BAISSIERE (rebond)"}

Niveaux de retracement:
- 0.236: ${fibo_levels.get('0.236', 0):,.2f}
- 0.382: ${fibo_levels.get('0.382', 0):,.2f}
- 0.5: ${fibo_levels.get('0.5', 0):,.2f}
- 0.618: ${fibo_levels.get('0.618', 0):,.2f} {"<-- NIVEAU CIBLE" if target_level == "0.618" else ""}
- 0.786: ${fibo_levels.get('0.786', 0):,.2f}

NIVEAU DE DECISION CONFIGURE: {target_level}
"""

        # Construire la section institutionnelle
        inst_section = ""
        if institutional and "recommendation" in institutional:
            rec = institutional["recommendation"]
            patterns = institutional.get("patterns_detected", [])
            structure = institutional.get("market_structure", {})

            pattern_list = [f"{p['type']} ({p['confidence']*100:.0f}%)" for p in patterns[:3]]

            inst_section = f"""
ANALYSE INSTITUTIONNELLE (ICT/SMC):
- Structure: {structure.get('trend', 'NEUTRAL')}
- Patterns: {', '.join(pattern_list) if pattern_list else 'Aucun'}
- Recommandation ICT: {rec.get('action', 'HOLD')} ({rec.get('confidence', 0)*100:.0f}%)
- Raison: {rec.get('reason', 'N/A')}
"""

        prompt = f"""
=== ANALYSE FIBO2 + ICT BTCUSD ===

PRIX ACTUEL: ${price_data.get('price', 0):,.2f}
SPREAD: {price_data.get('spread_points', 0)} points

MOMENTUM:
- 5 minutes: {momentum.get('5m', 0):.3f}%
- 15 minutes: {momentum.get('15m', 0):.3f}%
- 1 heure: {momentum.get('1h', 0):.3f}%

VOLATILITE: {price_data.get('volatility_pct', 0):.2f}%
{fibo_section}
{inst_section}
INDICATEURS COMPLEMENTAIRES:
- Funding Rate: {futures.get('funding_rate', 'N/A')}%
- Long/Short Ratio: {futures.get('long_short_ratio', 'N/A')}
- Fear & Greed: {sentiment.get('fear_greed_index', 'N/A')} ({sentiment.get('fear_greed_label', 'N/A')})

SESSION: {session.get('name', 'N/A')} (Volatilite: {session.get('volatility', 'N/A')})

=== TA MISSION ===
Tu es l'agent FIBO2. Tu trades sur les niveaux de Fibonacci avec confirmation ICT/SMC.

LOGIQUE FIBO2NACCI + ICT:
1. Prix atteint niveau Fibo en UPTREND -> BUY (retracement = opportunite d'achat)
2. Prix atteint niveau Fibo en DOWNTREND -> SELL (rebond = opportunite de vente)
3. Pattern ICT confirme le signal -> Confiance elevee
4. Pas de niveau Fibo proche ou contradiction ICT -> HOLD

ZONES CLE:
- 0.618 (Golden Ratio): Zone de retracement ideale
- 0.5: Zone psychologique
- 0.382: Retracement faible (tendance forte)
- 0.786: Retracement profond (risque plus eleve)

DECIDE: BUY, SELL, ou HOLD?
Format: ACTION: [BUY/SELL/HOLD] | RAISON: [explication courte]
"""
        return prompt

    def get_closer_prompt(self, context: Dict, position: Dict) -> str:
        """Prompt de fermeture specifique a Fibonacci"""
        price_data = context.get("price") or {}
        momentum = price_data.get("fibo1", {})
        fibo_levels = context.get("fibo_levels", {})

        direction = position.get("type", "BUY")
        entry_price = position.get("price_open", 0)
        current_price = position.get("price_current", 0)
        profit = position.get("profit", 0)

        # Calculer le temps depuis ouverture
        position_time = position.get("time", 0)
        age_seconds = 0
        if position_time:
            from datetime import datetime, timezone
            try:
                age_seconds = (datetime.now(timezone.utc) - datetime.fromtimestamp(position_time, timezone.utc)).total_seconds()
            except (ValueError, TypeError, OSError):
                age_seconds = 0

        # Determiner les objectifs Fibo
        swing_high = fibo_levels.get("swing_high", 0)
        swing_low = fibo_levels.get("swing_low", 0)

        if direction == "BUY":
            target = swing_high  # En BUY, objectif = retour au swing high
            stop = swing_low  # Stop sous le swing low
        else:
            target = swing_low  # En SELL, objectif = retour au swing low
            stop = swing_high  # Stop au-dessus du swing high

        prompt = f"""
=== GESTION POSITION FIBO2 ===

POSITION:
- Direction: {direction}
- Prix entree: ${entry_price:,.2f}
- Prix actuel: ${current_price:,.2f}
- Profit: {profit:.2f} EUR
- Duree position: {age_seconds:.0f} secondes

OBJECTIFS FIBO2NACCI:
- Swing High: ${swing_high:,.2f}
- Swing Low: ${swing_low:,.2f}
- Target ({direction}): ${target:,.2f}
- Stop zone: ${stop:,.2f}

MOMENTUM ACTUEL:
- 5 minutes: {momentum.get('5m', 0):.3f}%
- 15 minutes: {momentum.get('15m', 0):.3f}%

=== LOGIQUE DE GESTION ===
FERME SI:
1. Prix atteint le target (swing high pour BUY, swing low pour SELL)
2. Prix casse le niveau de stop
3. Momentum inverse fort (plus de -0.5% contre ta position)

GARDE SI:
- Le prix evolue vers le target
- Le momentum est favorable ou neutre

Format: ACTION: [KEEP/CLOSE] | RAISON: [explication]
"""
        return prompt


def get_fibo2_agent() -> Fibo2Agent:
    """Factory pour l'agent fibo2"""
    return Fibo2Agent()
