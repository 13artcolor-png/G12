# -*- coding: utf-8 -*-
"""
G12 - Boucle de trading principale
Orchestre les 3 agents pour ouvrir des positions
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Optional
import sys
sys.path.append('..')

from config import INTERVALS, AGENTS_CONFIG, SPREAD_CONFIG
from core.mt5_connector import get_mt5
from data.aggregator import get_aggregator
import traceback
from agents.fibo1 import get_fibo1_agent
from agents.fibo2 import get_fibo2_agent
from agents.fibo3 import get_fibo3_agent
from risk.risk_manager import get_risk_manager
from utils.logger import get_logger
from strategist import get_strategist
from session_logger import get_session_logger
from utils.telegram_service import get_telegram


class TradingLoop:
    """Boucle principale de trading G12"""

    def __init__(self):
        self.mt5 = get_mt5("fibo1")  # Compte de reference
        self.aggregator = get_aggregator()
        self.risk = get_risk_manager()
        self.logger = get_logger()
        self.strategist = get_strategist()
        self.session_logger = get_session_logger() # Added
        self.telegram = get_telegram() # Added

        # Agents
        self.agents = {
            "fibo1": get_fibo1_agent(),
            "fibo2": get_fibo2_agent(),
            "fibo3": get_fibo3_agent()
        }

        self.running = False
        self.interval = INTERVALS.get("trading_loop", 10)
        self.last_context = None
        self._tpsl_config_cache = None
        self._tpsl_config_time = 0
        self._iteration_count = 0

    def _load_tpsl_config(self) -> Dict:
        """Charge la config TP/SL depuis spread_runtime_config.json"""
        import json
        from pathlib import Path

        # Cache de 10 secondes pour eviter de lire le fichier a chaque trade
        current_time = time.time()
        if self._tpsl_config_cache and (current_time - self._tpsl_config_time) < 10:
            return self._tpsl_config_cache

        try:
            config_file = Path(__file__).parent.parent / "database" / "spread_runtime_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self._tpsl_config_cache = json.load(f)
                    self._tpsl_config_time = current_time
                    return self._tpsl_config_cache
        except Exception as e:
            print(f"[TradingLoop] Erreur chargement config TP/SL: {e}")

        # Config par defaut
        return SPREAD_CONFIG

    def start(self):
        """Demarre la boucle de trading"""
        self.running = True
        print("[TradingLoop] Demarrage...")

        # Connexion MT5
        if not self.mt5.connect():
            print("[TradingLoop] Impossible de se connecter a MT5")
            return

        # Initialiser la balance
        account = self.mt5.get_account_info()
        if account:
            self.risk.set_initial_balance(account["balance"])
            print(f"[TradingLoop] Balance initiale: {account['balance']} EUR")

        # Boucle principale
        while self.running:
            try:
                self._loop_iteration()
            except Exception as e:
                print(f"[TradingLoop] Erreur: {e}")
                traceback.print_exc()
                self.logger.log_error("trading_loop", str(e))

            time.sleep(self.interval)

    def stop(self):
        """Arrete la boucle (ne bloque pas sur MT5)"""
        self.running = False
        print("[TradingLoop] Arrete")

    def _loop_iteration(self):
        """Une iteration de la boucle"""
        self._iteration_count += 1
        context = None
        
        # Recharger les configs runtime pour voir les changements (ex: Telegram /stop)
        self.risk.reload_config()

        # === AUTO-OPTIMISATION (toutes les 10 iterations) ===
        if self._iteration_count % 10 == 0:
            try:
                result = self.strategist.auto_optimize()
                if result.get('status') == 'optimized' and result.get('executed_count', 0) > 0:
                    print(f"[TradingLoop] Strategist: {result['executed_count']} corrections appliquees")
            except Exception as e:
                print(f"[TradingLoop] Erreur strategist: {e}")

        # Recuperer le contexte complet
        context = self.aggregator.get_full_context()
        self.last_context = context

        # Verifier que le contexte est valide
        if not context:
            print("[TradingLoop] Contexte non disponible")
            return

        if not context.get("price"):
            print("[TradingLoop] Pas de donnees prix")
            return

        # Infos compte (gerer None avec or {})
        account = context.get("account") or {}
        equity = account.get("equity", 0)
        positions = account.get("positions", [])
        spread = (context.get("price") or {}).get("spread_points", 999)

        # Verifier si on peut trader
        can_trade, reason = self.risk.can_open_position(equity, spread, positions, context)

        if not can_trade:
            print(f"[TradingLoop] Trading bloque: {reason}")
            return

        # Mettre a jour les positions des agents
        self._update_agent_positions(positions)

        # Faire decider chaque agent
        for agent_id, agent in self.agents.items():
            if not agent.enabled:
                continue

            # Note: La verification max_positions est faite dans agent.decide_open()

            # Demander une decision
            decision = agent.decide_open(context)

            # Logger la decision
            self.logger.log_decision(
                agent_id=agent_id,
                decision=decision.get("action", "HOLD"),
                reason=decision.get("reason", ""),
                executed=False,
                context_snapshot={
                    "price": context.get("price", {}).get("price"),
                    "fibo1_1m": context.get("price", {}).get("momentum", {}).get("1m"),
                    "funding": context.get("futures", {}).get("funding_rate")
                }
            )

            # Executer si BUY ou SELL
            if decision.get("action") in ["BUY", "SELL"]:
                self._execute_trade(agent, decision, context)

        # === SNAPSHOT PERFORMANCE (pour persistance graphiques) ===
        try:
            session_logger = get_session_logger()
            for agent_id, agent in self.agents.items():
                # Calculer le P&L flottant pour cet agent
                # (On filtre les positions du contexte qui appartiennent a cet agent)
                agent_positions = [
                    pos for pos in positions 
                    if pos.get("_agent_id") == agent_id or f"G12_{agent_id}" in pos.get("comment", "")
                ]
                floating_pnl = sum(pos.get("profit", 0) for pos in agent_positions)
                
                # Snapshot: P&L ferme (session_pnl) + P&L flottant
                session_logger.log_performance_snapshot(
                    agent_id=agent_id,
                    closed_pnl=agent.session_pnl,
                    floating_pnl=floating_pnl
                )
        except Exception as e:
            print(f"[TradingLoop] Erreur snapshot performance: {e}")

    def _update_agent_positions(self, positions: list):
        """Met a jour les positions assignees aux agents (multi-comptes, multi-positions)

        Note: La detection des fermetures (TP/SL) est geree par CloserLoop.
        Cette methode ne fait que synchroniser les positions MT5 -> agent.
        """
        # Creer un dict des positions G12 par ticket
        g11_positions = {}
        for pos in positions:
            comment = pos.get("comment", "")
            if pos.get("magic") == 11 or "G12" in comment:
                g11_positions[pos["ticket"]] = pos

        # Pour chaque agent
        for agent_id, agent in self.agents.items():
            tracked_tickets = {p.get("ticket") for p in agent.open_positions}
            positions_changed = False

            # Garder les positions encore ouvertes
            still_open = []
            for pos in agent.open_positions:
                ticket = pos.get("ticket")
                if ticket in g11_positions:
                    still_open.append(pos)
                # Note: Les positions fermees sont detectees par CloserLoop

            # Ajouter les positions MT5 non trackees qui appartiennent a cet agent
            for ticket, pos in g11_positions.items():
                if ticket not in tracked_tickets:
                    comment = pos.get("comment", "")
                    if f"G12_{agent_id}" in comment:
                        print(f"[TradingLoop] Position {ticket} ajoutee a {agent_id} (sync MT5)")
                        still_open.append({
                            "ticket": ticket,
                            "direction": pos.get("type", "UNKNOWN"),
                            "entry_price": pos.get("price_open", 0),
                            "entry_time": pos.get("time", ""),
                            "sl": pos.get("sl", 0),
                            "tp": pos.get("tp", 0),
                            "volume": pos.get("volume", 0)
                        })
                        positions_changed = True

            # Sauvegarder seulement si changement
            if len(still_open) != len(agent.open_positions) or positions_changed:
                agent.open_positions = still_open
                agent.save_positions()

    def _execute_trade(self, agent, decision: Dict, context: Dict):
        """Execute un trade"""
        direction = decision.get("action")
        price_data = context.get("price", {})
        account = context.get("account", {})

        # Recuperer l'equity SPECIFIQUE du compte de cet agent (pas le total)
        accounts_data = account.get("accounts", {})
        agent_account = accounts_data.get(agent.agent_id, {})
        equity = agent_account.get("equity", 0)

        # Fallback sur equity totale si pas de compte specifique
        if equity == 0:
            equity = account.get("equity", 0)

        position_size_unit = agent.config.get("position_size_pct", 0.01)

        # Scaling: 0.01 lot pour 1000 EUR d'equity (standard BTCUSD)
        volume = round((equity / 1000.0) * position_size_unit, 2)
        volume = max(0.01, volume) # Minimum MT5

        print(f"[TradingLoop] {agent.name} -> Volume calcule: {volume} lots (Equity: {equity:.2f}, Unit: {position_size_unit})")

        # Charger config TP/SL depuis spread_runtime_config.json
        price = price_data.get("price", 0)
        tpsl_config = self._load_tpsl_config()
        tp_pct = tpsl_config.get("tp_pct", 0.5) / 100  # Convertir en decimal
        sl_pct = tpsl_config.get("sl_pct", 1.0) / 100  # Convertir en decimal

        # Calculer SL/TP en pourcentage du prix
        # BUY: TP au-dessus du prix, SL en-dessous
        # SELL: TP en-dessous du prix, SL au-dessus
        if direction == "BUY":
            tp = price * (1 + tp_pct)  # TP = prix + X%
            sl = price * (1 - sl_pct)  # SL = prix - X%
        else:
            tp = price * (1 - tp_pct)  # TP = prix - X%
            sl = price * (1 + sl_pct)  # SL = prix + X%

        print(f"[TradingLoop] TP/SL calcules: {direction} @ {price:.2f} -> TP={tp:.2f} ({tp_pct*100}%), SL={sl:.2f} ({sl_pct*100}%)")

        # Utiliser le compte MT5 specifique de l'agent
        agent_mt5 = get_mt5(agent.agent_id)
        if not agent_mt5.connect():
            print(f"[TradingLoop] {agent.name} -> Impossible de se connecter au compte MT5")
            self.logger.log_error(
                "mt5_connection",
                f"Echec connexion MT5 pour {agent.agent_id}",
                agent_id=agent.agent_id
            )
            return

        # Executer sur le compte MT5 de l'agent
        comment = f"G12_{agent.agent_id}"
        result = agent_mt5.open_position(
            direction=direction,
            volume=volume,
            sl=sl,
            tp=tp,
            comment=comment
        )

        if result:
            print(f"[TradingLoop] {agent.name} -> {direction} @ {result['price']}")

            # Ajouter la position a la liste de l'agent (thread-safe)
            agent.add_position({
                "ticket": result["ticket"],
                "direction": direction,
                "entry_price": result["price"],
                "entry_time": datetime.now().isoformat(),
                "sl": sl,
                "tp": tp,
                "volume": volume
            })
            agent.save_positions()  # Persister les positions

            # Logger comme execute
            self.logger.log_decision(
                agent_id=agent.agent_id,
                decision=direction,
                reason=decision.get("reason", ""),
                executed=True,
                context_snapshot={
                    "entry_price": result["price"],
                    "sl": sl,
                    "tp": tp
                }
            )
        else:
            print(f"[TradingLoop] {agent.name} -> Echec execution {direction}")
            self.logger.log_error(
                "trade_execution",
                f"Echec {direction} pour {agent.agent_id}",
                agent_id=agent.agent_id
            )

    def get_status(self) -> Dict:
        """Retourne le status de la boucle"""
        return {
            "running": self.running,
            "interval": self.interval,
            "mt5_connected": self.mt5.connected,
            "last_context_time": self.last_context.get("timestamp") if self.last_context else None,
            "agents": {
                agent_id: agent.get_status()
                for agent_id, agent in self.agents.items()
            }
        }


# Instance globale
_trading_loop = None

def get_trading_loop() -> TradingLoop:
    """Retourne l'instance TradingLoop singleton"""
    global _trading_loop
    if _trading_loop is None:
        _trading_loop = TradingLoop()
    return _trading_loop
