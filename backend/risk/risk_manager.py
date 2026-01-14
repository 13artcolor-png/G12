# -*- coding: utf-8 -*-
"""
G12 - Gestionnaire de risque
Protege le capital avec des regles strictes
"""

from datetime import datetime, date
from typing import Optional, Dict, List
import json
from pathlib import Path
import sys
sys.path.append('..')

from config import RISK_CONFIG, SPREAD_CONFIG, DATABASE_DIR


def load_runtime_config(filename: str, default: dict) -> dict:
    """Charge la config runtime depuis un fichier JSON (prioritaire sur config.py)"""
    filepath = DATABASE_DIR / filename
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                print(f"[Risk] Config runtime chargee: {filename}")
                return loaded
    except Exception as e:
        print(f"[Risk] Erreur chargement {filename}: {e}")
    return default


class RiskManager:
    """Gere le risque pour G12"""

    def __init__(self):
        # Charger la config runtime (prioritaire) ou fallback sur config.py
        self.config = load_runtime_config("risk_runtime_config.json", RISK_CONFIG)
        self.spread_config = load_runtime_config("spread_runtime_config.json", SPREAD_CONFIG)
        self.state_file = DATABASE_DIR / "state.json"
        print(f"[Risk] max_positions_total = {self.config.get('max_positions_total')}")

        # Etat en memoire
        self.trading_halted = False
        self.halt_reason = None
        self.initial_balance = None
        self.daily_start_balance = None
        self.daily_start_date = None

    def reload_config(self):
        """Recharge la config depuis les fichiers runtime"""
        self.config = load_runtime_config("risk_runtime_config.json", RISK_CONFIG)
        self.spread_config = load_runtime_config("spread_runtime_config.json", SPREAD_CONFIG)
        print(f"[Risk] Config rechargee: max_positions_total = {self.config.get('max_positions_total')}")

    def load_state(self) -> Dict:
        """Charge l'etat depuis le fichier"""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_state(self, state: Dict):
        """Sauvegarde l'etat dans le fichier"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[Risk] Erreur sauvegarde state: {e}")

    def update_state(self, updates: Dict):
        """Met a jour partiellement l'etat"""
        state = self.load_state()
        state.update(updates)
        self.save_state(state)

    def set_initial_balance(self, balance: float):
        """Definit la balance initiale de reference"""
        self.initial_balance = balance

        # Initialiser la balance journaliere si nouveau jour
        today = date.today()
        if self.daily_start_date != today:
            self.daily_start_date = today
            self.daily_start_balance = balance

    def check_spread(self, spread_points: int) -> tuple[bool, str]:
        """Verifie si le spread est acceptable"""
        max_spread = self.spread_config.get("max_spread_points", 50)

        if spread_points > max_spread:
            return False, f"Spread trop eleve: {spread_points} > {max_spread}"

        return True, "Spread OK"

    def check_drawdown(self, current_equity: float) -> tuple[bool, str]:
        """Verifie le drawdown"""
        if self.initial_balance is None:
            return True, "Balance initiale non definie"

        max_dd = self.config.get("max_drawdown_pct", 10)
        current_dd = ((self.initial_balance - current_equity) / self.initial_balance) * 100

        if current_dd > max_dd:
            self.halt_trading(f"Drawdown {current_dd:.1f}% > {max_dd}%")
            return False, f"Drawdown depasse: {current_dd:.1f}% > {max_dd}%"

        return True, f"Drawdown OK: {current_dd:.1f}%"

    def check_daily_loss(self, current_equity: float) -> tuple[bool, str]:
        """Verifie la perte journaliere"""
        if self.daily_start_balance is None:
            return True, "Balance journaliere non definie"

        max_daily_loss = self.config.get("max_daily_loss_pct", 5)
        daily_loss = ((self.daily_start_balance - current_equity) / self.daily_start_balance) * 100

        if daily_loss > max_daily_loss:
            self.halt_trading(f"Perte journaliere {daily_loss:.1f}% > {max_daily_loss}%")
            return False, f"Perte journaliere depassee: {daily_loss:.1f}%"

        return True, f"Perte journaliere OK: {daily_loss:.1f}%"

    def check_position_count(self, current_positions: int) -> tuple[bool, str]:
        """Verifie le nombre de positions"""
        max_positions = self.config.get("max_positions_total", 3)

        if current_positions >= max_positions:
            return False, f"Max positions atteint: {current_positions}/{max_positions}"

        return True, f"Positions OK: {current_positions}/{max_positions}"

    def check_emergency_close(self, position_profit_pct: float, position_profit_eur: float) -> tuple[bool, str]:
        """Verifie si une fermeture d'urgence est necessaire"""
        emergency_pct = self.config.get("emergency_close_pct", 15)
        min_loss_eur = 3.0  # Minimum 3 EUR de perte

        if position_profit_pct < -emergency_pct and position_profit_eur < -min_loss_eur:
            return True, f"EMERGENCY: Perte {position_profit_pct:.1f}% > {emergency_pct}%"

        return False, "Pas d'urgence"

    def check_tp_mechanical(self, profit_eur: float) -> tuple[bool, str]:
        """Verifie si le TP mecanique est atteint"""
        tp_eur = self.spread_config.get("tp_eur", 3.0)

        if profit_eur >= tp_eur:
            return True, f"TP mecanique atteint: {profit_eur:.2f} >= {tp_eur} EUR"

        return False, f"TP non atteint: {profit_eur:.2f} < {tp_eur} EUR"

    def check_forced_tp(self, profit_pct: float) -> tuple[bool, str]:
        """Verifie si le TP force (%) est atteint"""
        forced_tp = self.spread_config.get("forced_tp_pct", 0.5)

        if profit_pct >= forced_tp:
            return True, f"TP force atteint: {profit_pct:.2f}% >= {forced_tp}%"

        return False, f"TP force non atteint: {profit_pct:.2f}%"

    def can_open_position(self, equity: float, spread_points: int, positions: List[Dict], context: Dict = None) -> tuple[bool, str]:
        """Verification complete avant ouverture"""
        # Trading halte?
        if self.trading_halted:
            return False, f"Trading halte: {self.halt_reason}"

        # Verification si trading desactive manuellement (ex: via Telegram /stop)
        if self.config.get("trading_enabled", True) is False:
            return False, "Trading desactive manuellement (config)"

        # News protection
        if self.config.get("news_protection_enabled", True) and context and context.get("pause_recommended"):
            return False, "Pause trading recommandee (Annonce news high impact)"

        # Spread
        ok, msg = self.check_spread(spread_points)
        if not ok:
            return False, msg

        # Drawdown
        ok, msg = self.check_drawdown(equity)
        if not ok:
            return False, msg

        # Perte journaliere
        ok, msg = self.check_daily_loss(equity)
        if not ok:
            return False, msg

        # Nombre de positions
        ok, msg = self.check_position_count(len(positions))
        if not ok:
            return False, msg

        return True, "Toutes les verifications OK"

    def should_close_position(self, position: Dict, equity: float) -> tuple[bool, str]:
        """Determine si une position doit etre fermee (mecaniquement)"""
        profit_eur = position.get("profit", 0)
        price_open = position.get("price_open", 0)
        price_current = position.get("price_current", 0)
        volume = position.get("volume", 0)

        # Calculer le profit en pourcentage
        if price_open > 0:
            if position.get("type") == "BUY":
                profit_pct = ((price_current - price_open) / price_open) * 100
            else:
                profit_pct = ((price_open - price_current) / price_open) * 100
        else:
            profit_pct = 0

        # TP mecanique (EUR)
        ok, msg = self.check_tp_mechanical(profit_eur)
        if ok:
            return True, msg

        # TP force (%)
        ok, msg = self.check_forced_tp(profit_pct)
        if ok:
            return True, msg

        # Emergency close
        ok, msg = self.check_emergency_close(profit_pct, profit_eur)
        if ok:
            return True, msg

        return False, "Aucune condition de fermeture"

    def halt_trading(self, reason: str):
        """Arrete le trading"""
        self.trading_halted = True
        self.halt_reason = reason
        print(f"[Risk] TRADING HALTE: {reason}")

        self.update_state({
            "trading_halted": True,
            "halt_reason": reason,
            "halt_time": datetime.now().isoformat()
        })

    def resume_trading(self):
        """Reprend le trading"""
        self.trading_halted = False
        self.halt_reason = None
        print("[Risk] Trading repris")

        self.update_state({
            "trading_halted": False,
            "halt_reason": None
        })

    def get_status(self) -> Dict:
        """Retourne le status du risk manager"""
        return {
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "initial_balance": self.initial_balance,
            "daily_start_balance": self.daily_start_balance,
            "max_drawdown_pct": self.config.get("max_drawdown_pct"),
            "max_daily_loss_pct": self.config.get("max_daily_loss_pct"),
            "max_positions": self.config.get("max_positions_total"),
            "emergency_close_pct": self.config.get("emergency_close_pct"),
            "tp_eur": self.spread_config.get("tp_eur"),
            "sl_eur": self.spread_config.get("sl_eur"),
            "max_spread": self.spread_config.get("max_spread_points")
        }


# Singleton
_risk_instance = None

def get_risk_manager() -> RiskManager:
    """Retourne l'instance RiskManager singleton"""
    global _risk_instance
    if _risk_instance is None:
        _risk_instance = RiskManager()
    return _risk_instance


def reload_risk_config():
    """Recharge la config et reset le singleton"""
    global _risk_instance
    _risk_instance = None
    return get_risk_manager()
