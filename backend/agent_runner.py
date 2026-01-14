# -*- coding: utf-8 -*-
"""
G12 - Agent Runner
Lance un seul agent avec sa propre connexion MT5
Usage: python agent_runner.py <agent_id>
"""

import sys
import time
import threading
from datetime import datetime

# Ajouter le chemin backend
sys.path.insert(0, '.')

from core.mt5_connector import get_mt5
from risk.risk_manager import get_risk_manager
from data.aggregator import get_aggregator
from utils.logger import get_logger
from session_logger import get_session_logger
from config import INTERVALS


def load_agent(agent_id: str):
    """Charge l'agent specifique"""
    if agent_id == "fibo1":
        from agents.fibo1 import Fibo1Agent
        return Fibo1Agent()
    elif agent_id == "fibo2":
        from agents.fibo2 import Fibo2Agent
        return Fibo2Agent()
    elif agent_id == "fibo3":
        from agents.fibo3 import Fibo3Agent
        return Fibo3Agent()
    else:
        raise ValueError(f"Agent inconnu: {agent_id}")


class AgentLoop:
    """Boucle de trading pour un seul agent"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.mt5 = get_mt5(agent_id)
        self.agent = load_agent(agent_id)
        self.aggregator = get_aggregator()
        self.risk = get_risk_manager()
        self.logger = get_logger()
        self.session_logger = get_session_logger()

        self.running = False
        self.trading_interval = INTERVALS.get("trading_loop", 5)
        self.closer_interval = INTERVALS.get("closer_loop", 5)

    def start(self):
        """Demarre la boucle"""
        # Connexion initiale
        if not self.mt5.connect():
            print(f"[{self.agent_id.upper()}] ERREUR: Impossible de se connecter a MT5")
            return False

        account = self.mt5.get_account_info()
        if account:
            print(f"[{self.agent_id.upper()}] Connecte au compte {account.get('login')}")
            print(f"[{self.agent_id.upper()}] Balance: {account.get('balance', 0):.2f} EUR")

        self.running = True

        # Thread trading
        trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        trading_thread.start()

        # Thread closer
        closer_thread = threading.Thread(target=self._closer_loop, daemon=True)
        closer_thread.start()

        print(f"[{self.agent_id.upper()}] Boucles demarrees")
        return True

    def stop(self):
        """Arrete la boucle"""
        self.running = False
        print(f"[{self.agent_id.upper()}] Arrete")

    def _trading_loop(self):
        """Boucle de trading"""
        while self.running:
            try:
                self._trading_iteration()
            except Exception as e:
                print(f"[{self.agent_id.upper()}] Erreur trading: {e}")
                self.logger.log_error("trading_loop", str(e), agent_id=self.agent_id)
            time.sleep(self.trading_interval)

    def _closer_loop(self):
        """Boucle de fermeture"""
        while self.running:
            try:
                self._closer_iteration()
            except Exception as e:
                print(f"[{self.agent_id.upper()}] Erreur closer: {e}")
                self.logger.log_error("closer_loop", str(e), agent_id=self.agent_id)
            time.sleep(self.closer_interval)

    def _trading_iteration(self):
        """Une iteration de trading"""
        if not self.agent.enabled:
            return

        # Verifier si l'agent a deja une position
        if self.agent.open_position:
            return

        # Verifier cooldown
        if self.agent.last_trade_time:
            cooldown = getattr(self.agent, 'cooldown_seconds', 60)
            elapsed = (datetime.now() - self.agent.last_trade_time).total_seconds()
            if elapsed < cooldown:
                return

        # Recuperer contexte (utilise le compte momentum pour le prix)
        context = self.aggregator.get_full_context()
        if not context or not context.get("price"):
            return

        # Verifier spread
        spread_config = self.risk.spread_config
        if spread_config.get("spread_check_enabled", True):
            current_spread = context.get("price", {}).get("spread_points", 0)
            max_spread = spread_config.get("max_spread_points", 2000)
            if current_spread > max_spread:
                return  # Spread trop eleve

        # Verifier risque global
        can_trade, reason = self.risk.can_open_position(
            self.agent_id,
            context.get("account", {}).get("equity", 0)
        )
        if not can_trade:
            return

        # Demander decision a l'agent
        decision = self.agent.decide_open(context)

        if decision.get("action") in ["BUY", "SELL"]:
            self._execute_trade(decision, context)

    def _execute_trade(self, decision: dict, context: dict):
        """Execute un trade"""
        direction = decision["action"]
        price = context.get("price", {}).get("price", 0)

        # Volume
        account = self.mt5.get_account_info()
        if not account:
            return
        balance = account.get("balance", 0)
        position_size_pct = getattr(self.agent, 'position_size_pct', 0.01)
        volume = round(balance * position_size_pct / 100000, 2)
        volume = max(0.01, min(volume, 1.0))

        # TP/SL
        spread_config = self.risk.spread_config
        tp_pct = spread_config.get("tp_pct", 0.5) / 100
        sl_pct = spread_config.get("sl_pct", 0.3) / 100

        if direction == "BUY":
            tp = price * (1 + tp_pct)
            sl = price * (1 - sl_pct)
        else:
            tp = price * (1 - tp_pct)
            sl = price * (1 + sl_pct)

        # Ouvrir position
        comment = f"G12_{self.agent_id}"
        result = self.mt5.open_position(
            direction=direction,
            volume=volume,
            sl=sl,
            tp=tp,
            comment=comment
        )

        if result:
            print(f"[{self.agent_id.upper()}] {direction} @ {result['price']:.2f}")
            self.agent.open_position = result
            self.agent.last_trade_time = datetime.now()

            # Logger
            self.logger.log_decision(
                self.agent_id,
                decision.get("action"),
                decision.get("reason", ""),
                decision.get("confidence", 0)
            )
            self.session_logger.log_trade({
                'agent': self.agent_id,
                'direction': direction,
                'entry_price': result['price'],
                'volume': volume,
                'ticket': result.get('ticket')
            })

    def _closer_iteration(self):
        """Une iteration de fermeture"""
        # Recuperer positions sur CE compte uniquement
        positions = self.mt5.get_positions()
        if not positions:
            return

        context = self.aggregator.get_full_context()

        for position in positions:
            # Verifier si c'est une position G12 de cet agent
            comment = position.get("comment", "")
            if self.agent_id not in comment.lower():
                continue

            ticket = position["ticket"]
            profit = position.get("profit", 0)

            # Verifier fermeture mecanique (risk manager)
            should_close, reason = self.risk.should_close_position(
                position,
                context.get("account", {}).get("equity", 0)
            )

            if should_close:
                self._close_position(position, reason)
                continue

            # Fermeture IA (si activee et conditions remplies)
            if getattr(self.agent, 'ia_close_enabled', False):
                open_time = position.get("time")
                if open_time:
                    age_seconds = (datetime.now() - datetime.fromtimestamp(open_time)).total_seconds()
                    min_hold = getattr(self.agent, 'min_hold_seconds', 300)
                    min_loss = getattr(self.agent, 'min_loss_eur_for_ia', 1.0)

                    if age_seconds > min_hold and profit < -min_loss:
                        decision = self.agent.decide_close(context, position)
                        if decision.get("action") == "CLOSE":
                            self._close_position(position, f"IA: {decision.get('reason', '')}")

    def _close_position(self, position: dict, reason: str):
        """Ferme une position"""
        ticket = position["ticket"]
        result = self.mt5.close_position(ticket)

        if result:
            profit = result.get("profit", 0)
            print(f"[{self.agent_id.upper()}] Ferme #{ticket}: {profit:+.2f} EUR ({reason})")

            self.agent.open_position = None

            self.session_logger.log_trade({
                'agent': self.agent_id,
                'ticket': ticket,
                'profit': profit,
                'close_reason': reason
            })


def main():
    if len(sys.argv) < 2:
        print("Usage: python agent_runner.py <agent_id>")
        print("  agent_id: momentum, fibo, ou liquidation")
        sys.exit(1)

    agent_id = sys.argv[1].lower()

    if agent_id not in ["fibo1", "fibo2", "fibo3"]:
        print(f"Agent inconnu: {agent_id}")
        print("Agents valides: momentum, fibo, liquidation")
        sys.exit(1)

    print("=" * 60)
    print(f"  G12 - Agent {agent_id.upper()}")
    print("=" * 60)

    loop = AgentLoop(agent_id)

    if not loop.start():
        sys.exit(1)

    print("\nAppuyez sur Ctrl+C pour arreter.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArret en cours...")
        loop.stop()


if __name__ == "__main__":
    main()
