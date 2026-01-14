"""
GENIUS10 - Institutional Price Action Detector
Detecte les patterns institutionnels (QM, Stop Hunt, Liquidity, etc.)
Base sur ReadTheMarket / Mansor Sapari (CMS)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class TrendDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class PatternType(Enum):
    QM_BULLISH = "QM_BULLISH"           # Quasimodo haussier
    QM_BEARISH = "QM_BEARISH"           # Quasimodo baissier
    STOP_HUNT_BULL = "STOP_HUNT_BULL"   # Stop hunt puis hausse
    STOP_HUNT_BEAR = "STOP_HUNT_BEAR"   # Stop hunt puis baisse
    COMPRESSION = "COMPRESSION"          # Compression avant explosion
    THREE_DRIVE_TOP = "THREE_DRIVE_TOP" # 3 drive au sommet
    THREE_DRIVE_BOT = "THREE_DRIVE_BOT" # 3 drive au creux
    FLAG_BULL = "FLAG_BULL"             # Flag haussier
    FLAG_BEAR = "FLAG_BEAR"             # Flag baissier
    LIQUIDITY_GRAB = "LIQUIDITY_GRAB"   # Prise de liquidite

@dataclass
class SwingPoint:
    """Point de swing (sommet ou creux)"""
    index: int
    price: float
    is_high: bool  # True = sommet, False = creux

@dataclass
class Pattern:
    """Pattern detecte"""
    type: PatternType
    confidence: float  # 0.0 - 1.0
    entry_zone: Tuple[float, float]  # (min, max) zone d'entree
    stop_loss: float
    take_profit: float
    description: str

class InstitutionalPatternDetector:
    """
    Detecteur de patterns institutionnels
    Analyse les prix pour identifier les setups de trading
    """

    def __init__(self, swing_lookback: int = 5, min_swing_size: float = 0.0005):
        """
        Args:
            swing_lookback: Nombre de bougies pour confirmer un swing
            min_swing_size: Taille minimum d'un swing (en % du prix)
        """
        self.swing_lookback = swing_lookback
        self.min_swing_size = min_swing_size

    def find_swing_points(self, highs: np.ndarray, lows: np.ndarray,
                          closes: np.ndarray) -> List[SwingPoint]:
        """
        Trouve tous les points de swing (sommets et creux)
        """
        swings = []
        n = len(highs)
        lookback = self.swing_lookback

        for i in range(lookback, n - lookback):
            # Detecter swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                swings.append(SwingPoint(i, highs[i], True))

            # Detecter swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                swings.append(SwingPoint(i, lows[i], False))

        # Trier par index
        swings.sort(key=lambda x: x.index)
        return swings

    def get_market_structure(self, swings: List[SwingPoint]) -> Dict:
        """
        Analyse la structure de marche (HH, HL, LH, LL)
        """
        if len(swings) < 4:
            return {"trend": TrendDirection.NEUTRAL, "structure": []}

        structure = []
        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if not s.is_high]

        # Analyser les sommets
        for i in range(1, len(highs)):
            if highs[i].price > highs[i-1].price:
                structure.append(("HH", highs[i]))  # Higher High
            else:
                structure.append(("LH", highs[i]))  # Lower High

        # Analyser les creux
        for i in range(1, len(lows)):
            if lows[i].price > lows[i-1].price:
                structure.append(("HL", lows[i]))  # Higher Low
            else:
                structure.append(("LL", lows[i]))  # Lower Low

        # Determiner la tendance
        recent = structure[-4:] if len(structure) >= 4 else structure
        hh_count = sum(1 for s in recent if s[0] == "HH")
        hl_count = sum(1 for s in recent if s[0] == "HL")
        lh_count = sum(1 for s in recent if s[0] == "LH")
        ll_count = sum(1 for s in recent if s[0] == "LL")

        if hh_count + hl_count > lh_count + ll_count:
            trend = TrendDirection.BULLISH
        elif lh_count + ll_count > hh_count + hl_count:
            trend = TrendDirection.BEARISH
        else:
            trend = TrendDirection.NEUTRAL

        return {
            "trend": trend,
            "structure": structure,
            "hh_count": hh_count,
            "hl_count": hl_count,
            "lh_count": lh_count,
            "ll_count": ll_count
        }

    def detect_quasimodo(self, swings: List[SwingPoint],
                         current_price: float) -> Optional[Pattern]:
        """
        Detecte le pattern Quasimodo (QM)

        QM Bullish:  H -> HH -> L (casse) -> LL -> Retour au niveau QML
        QM Bearish:  L -> LL -> H (casse) -> HH -> Retour au niveau QML
        """
        if len(swings) < 5:
            return None

        # Prendre les 5 derniers swings
        recent = swings[-5:]

        # QM Bullish: chercher L, H, LL, HH, avec prix actuel proche du QML
        # Pattern: L -> H -> LL -> HH -> retour vers H (QML)
        highs = [s for s in recent if s.is_high]
        lows = [s for s in recent if not s.is_high]

        if len(highs) >= 2 and len(lows) >= 2:
            # QM Bullish
            if (lows[-1].price < lows[-2].price and  # LL
                highs[-1].price > highs[-2].price):   # HH

                qml_level = highs[-2].price  # Le "cou" du QM
                tolerance = abs(highs[-1].price - lows[-1].price) * 0.1

                if abs(current_price - qml_level) <= tolerance:
                    return Pattern(
                        type=PatternType.QM_BULLISH,
                        confidence=0.8,
                        entry_zone=(qml_level - tolerance, qml_level + tolerance),
                        stop_loss=lows[-1].price - tolerance,
                        take_profit=highs[-1].price,
                        description=f"QM Bullish - QML a {qml_level:.5f}, SL sous LL"
                    )

            # QM Bearish
            if (highs[-1].price > highs[-2].price and  # HH
                lows[-1].price < lows[-2].price):       # LL dans la structure

                qml_level = lows[-2].price  # Le "cou" du QM
                tolerance = abs(highs[-1].price - lows[-1].price) * 0.1

                if abs(current_price - qml_level) <= tolerance:
                    return Pattern(
                        type=PatternType.QM_BEARISH,
                        confidence=0.8,
                        entry_zone=(qml_level - tolerance, qml_level + tolerance),
                        stop_loss=highs[-1].price + tolerance,
                        take_profit=lows[-1].price,
                        description=f"QM Bearish - QML a {qml_level:.5f}, SL au-dessus HH"
                    )

        return None

    def detect_stop_hunt(self, highs: np.ndarray, lows: np.ndarray,
                         closes: np.ndarray, swings: List[SwingPoint]) -> Optional[Pattern]:
        """
        Detecte les Stop Hunts (chasse aux stop-loss)

        Stop Hunt = Prix qui casse un niveau cle puis revient violemment
        """
        if len(swings) < 3 or len(closes) < 10:
            return None

        current_price = closes[-1]
        prev_close = closes[-2]

        # Trouver les niveaux de support/resistance recents
        recent_highs = [s for s in swings[-6:] if s.is_high]
        recent_lows = [s for s in swings[-6:] if not s.is_high]

        if not recent_highs or not recent_lows:
            return None

        last_resistance = max(s.price for s in recent_highs)
        last_support = min(s.price for s in recent_lows)

        # Detecter Stop Hunt Bullish (fausse cassure du support)
        recent_low = min(lows[-5:])
        if (recent_low < last_support and  # Cassure du support
            current_price > last_support and  # Retour au-dessus
            prev_close < last_support):  # Etait en dessous

            return Pattern(
                type=PatternType.STOP_HUNT_BULL,
                confidence=0.75,
                entry_zone=(last_support, last_support * 1.002),
                stop_loss=recent_low * 0.998,
                take_profit=last_resistance,
                description=f"Stop Hunt Bullish - Faux breakout sous {last_support:.5f}"
            )

        # Detecter Stop Hunt Bearish (fausse cassure de la resistance)
        recent_high = max(highs[-5:])
        if (recent_high > last_resistance and  # Cassure de la resistance
            current_price < last_resistance and  # Retour en dessous
            prev_close > last_resistance):  # Etait au-dessus

            return Pattern(
                type=PatternType.STOP_HUNT_BEAR,
                confidence=0.75,
                entry_zone=(last_resistance * 0.998, last_resistance),
                stop_loss=recent_high * 1.002,
                take_profit=last_support,
                description=f"Stop Hunt Bearish - Faux breakout au-dessus {last_resistance:.5f}"
            )

        return None

    def detect_compression(self, highs: np.ndarray, lows: np.ndarray,
                           period: int = 20) -> Optional[Pattern]:
        """
        Detecte les compressions (prix qui se resserre)
        Precede souvent un mouvement explosif
        """
        if len(highs) < period:
            return None

        # Calculer l'ATR sur differentes periodes
        ranges = highs[-period:] - lows[-period:]
        recent_range = np.mean(ranges[-5:])
        older_range = np.mean(ranges[-period:-5])

        # Compression = range recent < 60% du range precedent
        if recent_range < older_range * 0.6:
            current_price = (highs[-1] + lows[-1]) / 2
            compression_high = max(highs[-5:])
            compression_low = min(lows[-5:])

            return Pattern(
                type=PatternType.COMPRESSION,
                confidence=0.7,
                entry_zone=(compression_low, compression_high),
                stop_loss=compression_low - (compression_high - compression_low) * 0.5,
                take_profit=compression_high + (compression_high - compression_low) * 2,
                description=f"Compression detectee - Range reduit de {(1 - recent_range/older_range)*100:.0f}%"
            )

        return None

    def detect_three_drive(self, swings: List[SwingPoint],
                           current_price: float) -> Optional[Pattern]:
        """
        Detecte le pattern 3 Drive (3 poussees = epuisement)
        """
        if len(swings) < 6:
            return None

        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if not s.is_high]

        # 3 Drive Top (3 sommets de plus en plus hauts = epuisement haussier)
        if len(highs) >= 3:
            last_3_highs = highs[-3:]
            if (last_3_highs[0].price < last_3_highs[1].price < last_3_highs[2].price):
                # Verifier que les increments diminuent (epuisement)
                inc1 = last_3_highs[1].price - last_3_highs[0].price
                inc2 = last_3_highs[2].price - last_3_highs[1].price

                if inc2 < inc1 * 0.8:  # Le 3eme drive est plus faible
                    return Pattern(
                        type=PatternType.THREE_DRIVE_TOP,
                        confidence=0.7,
                        entry_zone=(last_3_highs[2].price * 0.998, last_3_highs[2].price),
                        stop_loss=last_3_highs[2].price * 1.005,
                        take_profit=last_3_highs[0].price,
                        description="3 Drive Top - Epuisement haussier, retournement probable"
                    )

        # 3 Drive Bottom (3 creux de plus en plus bas = epuisement baissier)
        if len(lows) >= 3:
            last_3_lows = lows[-3:]
            if (last_3_lows[0].price > last_3_lows[1].price > last_3_lows[2].price):
                # Verifier que les increments diminuent
                dec1 = last_3_lows[0].price - last_3_lows[1].price
                dec2 = last_3_lows[1].price - last_3_lows[2].price

                if dec2 < dec1 * 0.8:  # Le 3eme drive est plus faible
                    return Pattern(
                        type=PatternType.THREE_DRIVE_BOT,
                        confidence=0.7,
                        entry_zone=(last_3_lows[2].price, last_3_lows[2].price * 1.002),
                        stop_loss=last_3_lows[2].price * 0.995,
                        take_profit=last_3_lows[0].price,
                        description="3 Drive Bottom - Epuisement baissier, retournement probable"
                    )

        return None

    def find_liquidity_zones(self, swings: List[SwingPoint],
                             current_price: float) -> List[Dict]:
        """
        Identifie les zones de liquidite (ou sont les stop-loss)
        """
        zones = []

        # Les stops sont generalement places:
        # - Juste sous les swing lows (pour les longs)
        # - Juste au-dessus des swing highs (pour les shorts)

        for swing in swings[-10:]:
            if swing.is_high:
                zone = {
                    "type": "SELL_STOPS",
                    "level": swing.price,
                    "liquidity_above": swing.price * 1.002,
                    "distance_pct": (swing.price - current_price) / current_price * 100,
                    "description": f"Liquidite SHORT au-dessus de {swing.price:.5f}"
                }
            else:
                zone = {
                    "type": "BUY_STOPS",
                    "level": swing.price,
                    "liquidity_below": swing.price * 0.998,
                    "distance_pct": (current_price - swing.price) / current_price * 100,
                    "description": f"Liquidite LONG en-dessous de {swing.price:.5f}"
                }
            zones.append(zone)

        return zones

    def analyze(self, highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray) -> Dict:
        """
        Analyse complete des patterns institutionnels

        Returns:
            Dict avec tous les patterns detectes et la structure de marche
        """
        if len(highs) < 20:
            return {"error": "Pas assez de donnees (minimum 20 bougies)"}

        current_price = closes[-1]

        # Trouver les swings
        swings = self.find_swing_points(highs, lows, closes)

        if len(swings) < 4:
            return {"error": "Pas assez de swings detectes"}

        # Analyser la structure
        structure = self.get_market_structure(swings)

        # Detecter les patterns
        patterns = []

        # Quasimodo
        qm = self.detect_quasimodo(swings, current_price)
        if qm:
            patterns.append(qm)

        # Stop Hunt
        stop_hunt = self.detect_stop_hunt(highs, lows, closes, swings)
        if stop_hunt:
            patterns.append(stop_hunt)

        # Compression
        compression = self.detect_compression(highs, lows)
        if compression:
            patterns.append(compression)

        # 3 Drive
        three_drive = self.detect_three_drive(swings, current_price)
        if three_drive:
            patterns.append(three_drive)

        # Zones de liquidite
        liquidity_zones = self.find_liquidity_zones(swings, current_price)

        # Generer la recommandation
        recommendation = self._generate_recommendation(patterns, structure, liquidity_zones, current_price)

        return {
            "current_price": current_price,
            "market_structure": {
                "trend": structure["trend"].value,
                "hh_count": structure["hh_count"],
                "hl_count": structure["hl_count"],
                "lh_count": structure["lh_count"],
                "ll_count": structure["ll_count"]
            },
            "patterns_detected": [
                {
                    "type": p.type.value,
                    "confidence": p.confidence,
                    "entry_zone": p.entry_zone,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "description": p.description
                }
                for p in patterns
            ],
            "liquidity_zones": liquidity_zones[:5],  # Top 5 plus proches
            "recommendation": recommendation,
            "swing_points": [
                {"index": s.index, "price": s.price, "type": "HIGH" if s.is_high else "LOW"}
                for s in swings[-10:]
            ]
        }

    def _generate_recommendation(self, patterns: List[Pattern],
                                  structure: Dict,
                                  liquidity_zones: List[Dict],
                                  current_price: float) -> Dict:
        """
        Genere une recommandation basee sur l'analyse
        """
        if not patterns:
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reason": "Aucun pattern institutionnel detecte"
            }

        # Prendre le pattern avec la plus haute confiance
        best_pattern = max(patterns, key=lambda p: p.confidence)

        # Determiner l'action
        if best_pattern.type in [PatternType.QM_BULLISH, PatternType.STOP_HUNT_BULL,
                                  PatternType.THREE_DRIVE_BOT]:
            action = "BUY"
        elif best_pattern.type in [PatternType.QM_BEARISH, PatternType.STOP_HUNT_BEAR,
                                    PatternType.THREE_DRIVE_TOP]:
            action = "SELL"
        elif best_pattern.type == PatternType.COMPRESSION:
            # Pour compression, suivre la tendance
            action = "BUY" if structure["trend"] == TrendDirection.BULLISH else "SELL"
        else:
            action = "HOLD"

        # Verifier la coherence avec la structure
        trend = structure["trend"]
        if action == "BUY" and trend == TrendDirection.BEARISH:
            confidence = best_pattern.confidence * 0.7  # Reduire si contre-tendance
            reason = f"{best_pattern.description} (ATTENTION: contre-tendance)"
        elif action == "SELL" and trend == TrendDirection.BULLISH:
            confidence = best_pattern.confidence * 0.7
            reason = f"{best_pattern.description} (ATTENTION: contre-tendance)"
        else:
            confidence = best_pattern.confidence
            reason = best_pattern.description

        return {
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "entry_zone": best_pattern.entry_zone,
            "stop_loss": best_pattern.stop_loss,
            "take_profit": best_pattern.take_profit,
            "risk_reward": abs(best_pattern.take_profit - current_price) / abs(current_price - best_pattern.stop_loss) if best_pattern.stop_loss != current_price else 0
        }


def format_for_ai_prompt(analysis: Dict) -> str:
    """
    Formate l'analyse pour l'inclure dans un prompt IA
    """
    if "error" in analysis:
        return f"[INSTITUTIONAL ANALYSIS] Erreur: {analysis['error']}"

    output = []
    output.append("=" * 50)
    output.append("ANALYSE INSTITUTIONNELLE (Price Action)")
    output.append("=" * 50)

    # Structure de marche
    ms = analysis["market_structure"]
    output.append(f"\n[STRUCTURE] Tendance: {ms['trend']}")
    output.append(f"  - Higher Highs: {ms['hh_count']} | Higher Lows: {ms['hl_count']}")
    output.append(f"  - Lower Highs: {ms['lh_count']} | Lower Lows: {ms['ll_count']}")

    # Patterns detectes
    output.append(f"\n[PATTERNS] {len(analysis['patterns_detected'])} pattern(s) detecte(s):")
    for p in analysis["patterns_detected"]:
        output.append(f"  * {p['type']} (confiance: {p['confidence']*100:.0f}%)")
        output.append(f"    {p['description']}")
        output.append(f"    Entry: {p['entry_zone'][0]:.5f} - {p['entry_zone'][1]:.5f}")
        output.append(f"    SL: {p['stop_loss']:.5f} | TP: {p['take_profit']:.5f}")

    # Zones de liquidite
    output.append(f"\n[LIQUIDITE] Zones proches:")
    for lz in analysis["liquidity_zones"][:3]:
        output.append(f"  * {lz['type']} @ {lz['level']:.5f} ({lz['distance_pct']:+.2f}%)")

    # Recommandation
    rec = analysis["recommendation"]
    output.append(f"\n[RECOMMANDATION]")
    output.append(f"  Action: {rec['action']} (confiance: {rec['confidence']*100:.0f}%)")
    output.append(f"  Raison: {rec['reason']}")
    if rec['action'] != 'HOLD':
        output.append(f"  Risk/Reward: {rec.get('risk_reward', 0):.2f}")

    output.append("=" * 50)

    return "\n".join(output)


# Test
if __name__ == "__main__":
    # Donnees de test simulees
    np.random.seed(42)
    n = 100

    # Simuler un mouvement avec QM pattern
    base = 100.0
    trend = np.cumsum(np.random.randn(n) * 0.5) + base
    noise = np.random.randn(n) * 0.2

    closes = trend + noise
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)

    # Analyser
    detector = InstitutionalPatternDetector(swing_lookback=3)
    analysis = detector.analyze(highs, lows, closes)

    # Afficher
    print(format_for_ai_prompt(analysis))
