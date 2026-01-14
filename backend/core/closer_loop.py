# -*- coding: utf-8 -*-
"""
G12 - Boucle de fermeture
Gere la fermeture intelligente des positions
Detecte aussi les positions fermees par MT5 (TP/SL)
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
import sys
sys.path.append('..')

from config import INTERVALS, RISK_CONFIG
from core.mt5_connector import get_mt5
from data.aggregator import get_aggregator
from core.trading_loop import get_trading_loop
from risk.risk_manager import get_risk_manager
from utils.logger import get_logger
from session_logger import get_session_logger


class CloserLoop:
    """Boucle de fermeture G12"""

    def __init__(self):
        self.mt5 = get_mt5("fibo1")  # Fallback sur momentum
        self.aggregator = get_aggregator()
        self.risk = get_risk_manager()
        self.logger = get_logger()
        self.session_logger = get_session_logger()

        self.running = False
        self.interval = INTERVALS.get("closer_loop", 5)
        self.last_sync_time = None  # Pour le sync MT5 periodique
        self.sync_interval = 10  # Sync MT5 toutes les 10 secondes pour temps reel

        # Tracking des positions fantomes (ticket -> timestamp premiere detection)
        self.phantom_positions: Dict[int, datetime] = {}
        self.phantom_max_age_minutes = 30  # Nettoyer apres 30 minutes sans trouver le deal

    def start(self):
        """Demarre la boucle de fermeture"""
        self.running = True
        print("[CloserLoop] Demarrage...")

        # SYNC INITIAL: Recuperer tous les trades MT5 depuis le debut de la session
        # Cela garantit que les stats correspondent aux VRAIS trades MT5
        print("[CloserLoop] Synchronisation initiale avec MT5...")
        try:
            sync_result = self.session_logger.sync_with_mt5_history()
            if sync_result.get('success'):
                print(f"[CloserLoop] Sync initial OK: {sync_result.get('total_trades', 0)} trades, PnL={sync_result.get('total_synced_pnl', 0):+.2f} EUR")
            else:
                print(f"[CloserLoop] Sync initial: {sync_result.get('message', 'pas de session active')}")
        except Exception as e:
            print(f"[CloserLoop] Erreur sync initial: {e}")

        self.last_sync_time = time.time()

        while self.running:
            try:
                self._loop_iteration()

                # CONTROLE PERIODIQUE: Sync MT5 toutes les 10 secondes
                # Verifie que les stats G12 correspondent aux vrais trades MT5
                now = time.time()
                if self.last_sync_time and (now - self.last_sync_time) >= self.sync_interval:
                    print("[CloserLoop] Controle periodique MT5 (10s)...")
                    try:
                        sync_result = self.session_logger.sync_with_mt5_history()
                        if sync_result.get('success'):
                            new_trades = sync_result.get('new_trades', 0)
                            if new_trades > 0:
                                print(f"[CloserLoop] {new_trades} nouveaux trades synchronises depuis MT5")
                            else:
                                print("[CloserLoop] Stats G12 = Stats MT5 (OK)")
                    except Exception as e:
                        print(f"[CloserLoop] Erreur sync periodique: {e}")
                    self.last_sync_time = now

            except Exception as e:
                print(f"[CloserLoop] Erreur: {e}")
                self.logger.log_error("closer_loop", str(e))

            time.sleep(self.interval)

    def stop(self):
        """Arrete la boucle"""
        self.running = False
        print("[CloserLoop] Arrete")

    def _loop_iteration(self):
        """Une iteration de la boucle"""
        # Recuperer la trading loop pour acceder aux agents
        trading_loop = get_trading_loop()

        # OPTIMISATION: Ne verifier que les comptes des agents qui ont des positions en memoire
        # Cela evite les reconnexions inutiles
        all_positions = []
        agents_with_positions = []

        # D'abord, identifier les agents qui ont des positions en memoire (thread-safe)
        for agent_id, agent in trading_loop.agents.items():
            if agent.get_positions_count() > 0:
                agents_with_positions.append(agent_id)

        # Si aucun agent n'a de position, ne pas faire de reconnexion
        if not agents_with_positions:
            return

        # Dictionnaire pour tracker les tickets MT5 par agent
        mt5_tickets_by_agent: Dict[str, Set[int]] = {}

        # Ne verifier QUE les comptes des agents avec positions
        for agent_id in agents_with_positions:
            try:
                agent_mt5 = get_mt5(agent_id)
                if agent_mt5.connect():
                    positions = agent_mt5.get_positions()
                    mt5_tickets_by_agent[agent_id] = set()
                    for pos in positions:
                        pos["_agent_id"] = agent_id
                        all_positions.append(pos)
                        mt5_tickets_by_agent[agent_id].add(pos["ticket"])
            except Exception as e:
                print(f"[CloserLoop] Erreur recuperation positions {agent_id}: {e}")

        # DETECTION DES POSITIONS FERMEES PAR TP/SL
        # Comparer les positions en memoire avec celles de MT5
        self._detect_closed_positions(trading_loop, mt5_tickets_by_agent)

        if not all_positions:
            return

        # Recuperer le contexte
        context = self.aggregator.get_full_context() or {}

        for position in all_positions:
            # Verifier si c'est une position G12
            if position.get("magic") != 11 and "G12" not in position.get("comment", ""):
                continue

            ticket = position["ticket"]
            profit = position.get("profit", 0)

            # 1. Verifier fermeture mecanique (risk manager)
            account = context.get("account") or {}
            should_close, reason = self.risk.should_close_position(position, account.get("equity", 0))

            if should_close:
                # Trouver l'agent via le marqueur ou le comment
                agent = None
                if "_agent_id" in position:
                    agent = trading_loop.agents.get(position["_agent_id"])
                if agent is None:
                    agent = self._find_position_owner(position, trading_loop)
                self._close_position(position, reason, trading_loop, agent)
                continue

            # 2. IMPORTANT: Ne PAS consulter l'IA pour fermer les positions!
            # Le spread BTCUSD fait que toute position est "en perte" immediatement.
            # Laisser SL/TP sur MT5 gerer la fermeture.
            #
            # L'IA ne sera consultee QUE si:
            # - La position a plus de 5 MINUTES
            # - ET la perte depasse 1 EUR (pour couvrir le spread)
            # - ET la config autorise la fermeture IA

            if profit < 0:
                agent = self._find_position_owner(position, trading_loop)

                if agent:
                    # Verifier si la fermeture IA est autorisee pour cet agent
                    ia_close_enabled = agent.config.get("ia_close_enabled", False)  # DESACTIVE par defaut!

                    if not ia_close_enabled:
                        continue  # Pas de fermeture IA, laisser SL/TP

                    # PROTECTION 1: Minimum 5 MINUTES (pas 30 secondes!)
                    position_time = position.get("time", 0)
                    if position_time:
                        from datetime import datetime, timezone
                        try:
                            age_seconds = (datetime.now(timezone.utc) - datetime.fromtimestamp(position_time, timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OSError):
                            age_seconds = 0

                        min_hold_time = agent.config.get("min_hold_seconds", 300)  # 5 MINUTES par defaut

                        if age_seconds < min_hold_time:
                            continue  # Position trop jeune

                    # PROTECTION 2: Perte minimum en EUR (pas en %)
                    # Le spread coute environ 0.20-0.50 EUR, donc seuil a 1 EUR
                    min_loss_eur = agent.config.get("min_loss_eur_for_ia", 1.0)

                    if abs(profit) < min_loss_eur:
                        continue  # Perte trop faible (probablement juste le spread)

                    # Si on arrive ici, consulter l'IA
                    decision = agent.decide_close(context, position)

                    if decision.get("action") == "CLOSE":
                        self._close_position(position, f"IA_CLOSE: {decision.get('reason', '')}", trading_loop, agent)

    def _find_position_owner(self, position: Dict, trading_loop) -> Optional[object]:
        """Trouve l'agent proprietaire d'une position"""
        comment = position.get("comment", "")
        ticket = position["ticket"]

        for agent_id, agent in trading_loop.agents.items():
            if agent_id in comment.lower():
                return agent

            # Chercher dans la liste des positions ouvertes
            for open_pos in agent.open_positions:
                if open_pos.get("ticket") == ticket:
                    return agent

        return None

    def _detect_closed_positions(self, trading_loop, mt5_tickets_by_agent: Dict[str, Set[int]]):
        """Detecte les positions fermees par MT5 (TP/SL) et les enregistre"""
        for agent_id, agent in trading_loop.agents.items():
            # Utiliser methode thread-safe
            positions_copy = agent.get_positions_copy()
            if not positions_copy:
                continue

            # Tickets actuellement sur MT5 pour cet agent
            mt5_tickets = mt5_tickets_by_agent.get(agent_id, set())
            mem_tickets = [p.get("ticket") for p in positions_copy]

            # Debug: montrer la comparaison
            if mem_tickets:
                print(f"[CloserLoop] {agent_id}: memoire={mem_tickets} vs MT5={mt5_tickets}")

            # Trouver les positions qui ont disparu de MT5
            positions_to_remove = []
            for mem_pos in positions_copy:
                mem_ticket = mem_pos.get("ticket")
                if mem_ticket and mem_ticket not in mt5_tickets:
                    # Position fermee par MT5 (TP/SL)
                    print(f"[CloserLoop] Position {mem_ticket} disparue de MT5 - recherche deal...")
                    positions_to_remove.append(mem_pos)

            if not positions_to_remove:
                continue

            # Recuperer l'historique des deals pour obtenir les infos de cloture
            agent_mt5 = get_mt5(agent_id)
            if not agent_mt5.connect():
                continue

            # Recuperer les deals de la derniere heure (plus large fenetre)
            from_date = datetime.now() - timedelta(hours=1)
            deals = agent_mt5.get_history_deals(from_date=from_date)

            # Creer des index des deals par position_id ET par ticket
            deals_by_position = {}
            deals_by_ticket = {}
            for deal in deals:
                pos_id = deal.get("position_id")
                ticket = deal.get("ticket")
                if pos_id:
                    deals_by_position[pos_id] = deal
                if ticket:
                    deals_by_ticket[ticket] = deal

            # Tracker les positions vraiment fermees (avec deal trouve)
            actually_closed_tickets = set()

            for mem_pos in positions_to_remove:
                mem_ticket = mem_pos.get("ticket")
                position_id = mem_pos.get("position_id", mem_ticket)

                # Chercher le deal de cloture (par position_id OU par ticket)
                deal = deals_by_position.get(position_id) or deals_by_position.get(mem_ticket) or deals_by_ticket.get(mem_ticket)

                if deal:
                    # Deal trouve - on peut logger avec le vrai profit
                    profit = deal.get("profit", 0)
                    close_reason = "TP" if profit > 0 else "SL"
                    exit_price = deal.get("price", 0)
                    direction = deal.get("type", mem_pos.get("direction", "BUY"))
                    entry_price = mem_pos.get("entry_price", 0)
                    volume = deal.get("volume", mem_pos.get("volume", 0.01))

                    print(f"[CloserLoop] Deal trouve pour position {mem_ticket}: profit={profit:.2f} EUR")
                    print(f"[CloserLoop] Position {mem_ticket} fermee par {close_reason}: {profit:+.2f} EUR (agent: {agent_id})")

                    # Enregistrer le trade dans les stats de l'agent
                    agent.record_trade(profit, profit > 0)

                    # Logger le trade dans l'historique global
                    self.logger.log_trade(
                        agent_id=agent_id,
                        direction=direction,
                        volume=volume,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit=profit,
                        close_reason=close_reason,
                        ticket=mem_ticket
                    )

                    # Logger le trade dans la session active
                    self.session_logger.log_trade({
                        'agent': agent_id,
                        'direction': direction,
                        'volume': volume,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'close_reason': close_reason,
                        'ticket': mem_ticket
                    })

                    actually_closed_tickets.add(mem_ticket)

                    # Retirer du tracking fantome si present
                    if mem_ticket in self.phantom_positions:
                        del self.phantom_positions[mem_ticket]
                else:
                    # Deal NON trouve - tracker comme position fantome
                    should_cleanup = self._handle_phantom_position(mem_ticket, mem_pos, agent_id)
                    if should_cleanup:
                        actually_closed_tickets.add(mem_ticket)

            # Retirer SEULEMENT les positions dont on a trouve le deal (thread-safe)
            for closed_ticket in actually_closed_tickets:
                agent.remove_position_by_ticket(closed_ticket)
            agent.save_positions()  # Persister les positions

    def _handle_phantom_position(self, ticket: int, position: Dict, agent_id: str) -> bool:
        """
        Gere une position fantome (presente en memoire mais absente de MT5).

        Args:
            ticket: Numero du ticket de la position
            position: Donnees de la position en memoire
            agent_id: ID de l'agent proprietaire

        Returns:
            True si la position doit etre nettoyee (supprimee de la memoire)
            False si on doit continuer a chercher le deal
        """
        now = datetime.now()

        # Premiere detection de cette position fantome?
        if ticket not in self.phantom_positions:
            self.phantom_positions[ticket] = now
            print(f"[CloserLoop] Position fantome detectee: {ticket} (agent: {agent_id}) - debut du tracking")
            print(f"[CloserLoop] Reessai pendant {self.phantom_max_age_minutes} min avant nettoyage automatique")
            return False

        # Calculer l'age de la detection
        first_detection = self.phantom_positions[ticket]
        age_minutes = (now - first_detection).total_seconds() / 60

        if age_minutes >= self.phantom_max_age_minutes:
            # Position fantome depuis trop longtemps - nettoyer
            entry_price = position.get("entry_price", position.get("price_open", 0))
            volume = position.get("volume", 0)
            direction = position.get("direction", position.get("type", "?"))

            print(f"[CloserLoop] NETTOYAGE AUTOMATIQUE: Position fantome {ticket} (agent: {agent_id})")
            print(f"[CloserLoop] Details: {direction} {volume} lots @ {entry_price}")
            print(f"[CloserLoop] Position non trouvee dans MT5 depuis {age_minutes:.1f} minutes")
            print(f"[CloserLoop] Deal de fermeture introuvable - position supprimee de la memoire G12")

            # Logger l'evenement
            self.logger.log_error(
                "phantom_cleanup",
                f"Position {ticket} ({direction} {volume}@{entry_price}) nettoyee apres {age_minutes:.1f}min sans deal (agent: {agent_id})"
            )

            # Retirer du tracking
            del self.phantom_positions[ticket]

            return True  # Signaler de supprimer de la memoire
        else:
            # Continuer a chercher
            remaining = self.phantom_max_age_minutes - age_minutes
            print(f"[CloserLoop] Position fantome {ticket}: deal non trouve, reessai ({remaining:.1f} min restantes)")
            return False

    def _close_position(self, position: Dict, reason: str, trading_loop, agent=None):
        """Ferme une position et log le resultat"""
        ticket = position["ticket"]
        profit = position.get("profit", 0)
        direction = position.get("type", "BUY")
        entry_price = position.get("price_open", 0)

        # Trouver l'agent proprietaire si non fourni
        if agent is None:
            agent = self._find_position_owner(position, trading_loop)

        # Utiliser le compte MT5 specifique de l'agent
        if agent:
            agent_mt5 = get_mt5(agent.agent_id)
            if not agent_mt5.connect():
                print(f"[CloserLoop] Impossible de se connecter au compte MT5 de {agent.agent_id}")
                return
        else:
            agent_mt5 = self.mt5  # Fallback sur le compte par defaut

        # Fermer sur MT5
        result = agent_mt5.close_position(ticket)

        if result:
            exit_price = result.get("close_price", 0)

            print(f"[CloserLoop] Position {ticket} fermee: {profit:+.2f} EUR ({reason})")

            # Trouver l'agent et mettre a jour
            agent = self._find_position_owner(position, trading_loop)

            if agent:
                agent.record_trade(profit, profit > 0)
                # Retirer cette position de la liste (thread-safe)
                agent.remove_position_by_ticket(ticket)
                agent.save_positions()  # Persister les positions

                # Logger le trade dans l'historique global
                self.logger.log_trade(
                    agent_id=agent.agent_id,
                    direction=direction,
                    volume=position.get("volume", 0),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    profit=profit,
                    close_reason=reason,
                    ticket=ticket
                )

                # Logger le trade dans la session active
                self.session_logger.log_trade({
                    'agent': agent.agent_id,
                    'direction': direction,
                    'volume': position.get("volume", 0),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'close_reason': reason,
                    'ticket': ticket
                })
        else:
            print(f"[CloserLoop] Echec fermeture position {ticket}")
            self.logger.log_error("close_position", f"Echec fermeture {ticket}")

    def close_all(self) -> Dict:
        """Ferme toutes les positions G12 sur tous les comptes agents"""
        results = []
        trading_loop = get_trading_loop()

        # Parcourir tous les comptes agents
        for agent_id in ["fibo1", "fibo2", "fibo3"]:
            try:
                agent_mt5 = get_mt5(agent_id)
                if agent_mt5.connect():
                    positions = agent_mt5.get_positions()
                    agent = trading_loop.agents.get(agent_id)

                    for position in positions:
                        if position.get("magic") == 11 or "G12" in position.get("comment", ""):
                            self._close_position(position, "MANUAL_CLOSE_ALL", trading_loop, agent)
                            results.append({
                                "ticket": position["ticket"],
                                "profit": position.get("profit", 0),
                                "agent": agent_id
                            })
            except Exception as e:
                print(f"[CloserLoop] Erreur fermeture positions {agent_id}: {e}")

        return {
            "closed": len(results),
            "positions": results,
            "total_profit": sum(r["profit"] for r in results)
        }

    def get_status(self) -> Dict:
        """Retourne le status de la boucle"""
        return {
            "running": self.running,
            "interval": self.interval
        }


# Instance globale
_closer_loop = None

def get_closer_loop() -> CloserLoop:
    """Retourne l'instance CloserLoop singleton"""
    global _closer_loop
    if _closer_loop is None:
        _closer_loop = CloserLoop()
    return _closer_loop
