# -*- coding: utf-8 -*-
"""
G12 - Logger
Enregistre les trades et decisions
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from config import DATABASE_DIR, LOG_CONFIG


class G12Logger:
    """Logger pour G12"""

    def __init__(self):
        self.trades_file = DATABASE_DIR / "trades.json"
        self.decisions_file = DATABASE_DIR / "decisions.json"
        self.logs_dir = DATABASE_DIR / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.max_trades = LOG_CONFIG.get("max_trades_history", 1000)
        self.max_decisions = LOG_CONFIG.get("max_decisions_history", 5000)

    def _load_json(self, filepath: Path) -> Dict:
        """Charge un fichier JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            # Log l'erreur seulement si ce n'est pas un fichier manquant (normal au demarrage)
            if not isinstance(e, FileNotFoundError):
                print(f"[Logger] Erreur chargement {filepath}: {e}")
            return {}

    def _save_json(self, filepath: Path, data: Dict):
        """Sauvegarde un fichier JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Logger] Erreur sauvegarde {filepath}: {e}")

    def log_trade(self, agent_id: str, direction: str, volume: float,
                  entry_price: float, exit_price: float, profit: float,
                  close_reason: str, entry_context: Dict = None,
                  exit_context: Dict = None, ticket: int = None) -> str:
        """Enregistre un trade termine"""
        trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ticket or 0}"

        trade = {
            "id": trade_id,
            "ticket": ticket,
            "agent_id": agent_id,
            "symbol": "BTCUSD",
            "direction": direction,
            "volume": volume,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_eur": round(profit, 2),
            "close_reason": close_reason,
            "entry_context": entry_context,
            "exit_context": exit_context,
            "timestamp": datetime.now().isoformat(),
            "won": profit > 0
        }

        # Charger et ajouter
        data = self._load_json(self.trades_file)
        trades = data.get("trades", [])
        trades.insert(0, trade)

        # Limiter la taille
        if len(trades) > self.max_trades:
            trades = trades[:self.max_trades]

        data["trades"] = trades
        self._save_json(self.trades_file, data)

        print(f"[Logger] Trade enregistre: {agent_id} {direction} {profit:+.2f} EUR ({close_reason})")
        return trade_id

    def log_decision(self, agent_id: str, decision: str, reason: str,
                     executed: bool = False, context_snapshot: Dict = None) -> str:
        """Enregistre une decision d'agent"""
        decision_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{agent_id}"

        decision_record = {
            "id": decision_id,
            "agent_id": agent_id,
            "symbol": "BTCUSD",
            "decision": decision,
            "reason": reason[:500],  # Limiter la taille
            "executed": executed,
            "context_snapshot": context_snapshot,
            "timestamp": datetime.now().isoformat()
        }

        # Charger et ajouter
        data = self._load_json(self.decisions_file)
        decisions = data.get("decisions", [])
        decisions.insert(0, decision_record)

        # Limiter la taille
        if len(decisions) > self.max_decisions:
            decisions = decisions[:self.max_decisions]

        data["decisions"] = decisions
        self._save_json(self.decisions_file, data)

        return decision_id

    def log_error(self, error_type: str, message: str, agent_id: str = None,
                  context: Dict = None):
        """Enregistre une erreur"""
        error_file = self.logs_dir / "errors.json"

        error = {
            "type": error_type,
            "message": message,
            "agent_id": agent_id,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

        # Charger et ajouter
        data = self._load_json(error_file)
        if not data:
            data = {"errors": []}

        errors = data.get("errors", [])
        errors.insert(0, error)

        # Garder les 500 dernieres erreurs
        if len(errors) > 500:
            errors = errors[:500]

        data["errors"] = errors
        self._save_json(error_file, data)

        print(f"[Logger] ERREUR {error_type}: {message}")

    def get_recent_trades(self, agent_id: str = None, limit: int = 50) -> List[Dict]:
        """Recupere les trades recents"""
        data = self._load_json(self.trades_file)
        trades = data.get("trades", [])

        if agent_id:
            trades = [t for t in trades if t.get("agent_id") == agent_id]

        return trades[:limit]

    def get_recent_decisions(self, agent_id: str = None, limit: int = 100) -> List[Dict]:
        """Recupere les decisions recentes"""
        data = self._load_json(self.decisions_file)
        decisions = data.get("decisions", [])

        if agent_id:
            decisions = [d for d in decisions if d.get("agent_id") == agent_id]

        return decisions[:limit]

    def get_agent_stats(self, agent_id: str, days: int = 7) -> Dict:
        """Calcule les stats d'un agent - utilise stats_*.json (source de verite MT5)"""
        # PRIORITE: Utiliser les stats synchronisees avec MT5
        stats_file = DATABASE_DIR / f"stats_{agent_id}.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                    session_pnl = stats_data.get("session_pnl", 0)
                    session_trades = stats_data.get("session_trades", 0)
                    session_wins = stats_data.get("session_wins", 0)
                    return {
                        "agent_id": agent_id,
                        "total_trades": session_trades,
                        "winning_trades": session_wins,
                        "losing_trades": session_trades - session_wins,
                        "win_rate": round(session_wins / session_trades * 100, 1) if session_trades > 0 else 0,
                        "total_profit": round(session_pnl, 2),
                        "avg_profit": round(session_pnl / session_trades, 2) if session_trades > 0 else 0,
                        "best_trade": 0,  # Non disponible dans stats_*.json
                        "worst_trade": 0
                    }
            except Exception as e:
                print(f"[Logger] Erreur lecture stats_{agent_id}.json: {e}")

        # Fallback: calculer depuis les trades logs
        cutoff = datetime.now() - timedelta(days=days)

        # Charger les trades du fichier principal
        data = self._load_json(self.trades_file)
        trades = data.get("trades", [])
        
        # Charger egalement les trades des fichiers de sessions
        sessions_dir = DATABASE_DIR / "sessions"
        if sessions_dir.exists():
            for session_file in sessions_dir.iterdir():
                # Lire les fichiers session_*.json (pas les dossiers archive_*)
                if session_file.is_file() and session_file.name.startswith("session_") and session_file.suffix == ".json":
                    try:
                        session_data = self._load_json(session_file)
                        session_trades = session_data.get("trades", [])
                        # Normaliser le format des trades de session
                        for t in session_trades:
                            # Le format session utilise 'agent' au lieu de 'agent_id'
                            if "agent" in t and "agent_id" not in t:
                                # Normaliser: "G12_fibo1" -> "fibo1"
                                agent_name = t["agent"]
                                if agent_name.startswith("G12_"):
                                    agent_name = agent_name[4:]
                                t["agent_id"] = agent_name
                            # Le format session utilise 'profit' au lieu de 'profit_eur'
                            if "profit" in t and "profit_eur" not in t:
                                t["profit_eur"] = t["profit"]
                            # Determiner si c'est un trade gagnant
                            if "won" not in t:
                                t["won"] = t.get("profit_eur", t.get("profit", 0)) > 0
                        trades.extend(session_trades)
                    except Exception as e:
                        print(f"[Logger] Erreur lecture session {session_file}: {e}")

        # Charger AUSSI la session active (session.json)
        active_session_file = DATABASE_DIR / "session.json"
        if active_session_file.exists():
            try:
                session_data = self._load_json(active_session_file)
                session_trades = session_data.get("trades", [])
                for t in session_trades:
                    # Normalisation (meme logique que ci-dessus)
                    if "agent" in t and "agent_id" not in t:
                        agent_name = t["agent"]
                        if agent_name.startswith("G12_"):
                            agent_name = agent_name[4:]
                        t["agent_id"] = agent_name
                    if "profit" in t and "profit_eur" not in t:
                        t["profit_eur"] = t["profit"]
                    if "won" not in t:
                        t["won"] = t.get("profit_eur", t.get("profit", 0)) > 0
                trades.extend(session_trades)
            except Exception as e:
                print(f"[Logger] Erreur lecture active session: {e}")

        # Filtrer par agent et date
        agent_trades = []
        for t in trades:
            if t.get("agent_id") != agent_id:
                continue
            try:
                trade_time = datetime.fromisoformat(t.get("timestamp", ""))
                if trade_time >= cutoff:
                    agent_trades.append(t)
            except (ValueError, TypeError):
                continue

        if not agent_trades:
            return {
                "agent_id": agent_id,
                "total_trades": 0,
                "win_rate": 0,
                "total_profit": 0,
                "avg_profit": 0,
                "best_trade": 0,
                "worst_trade": 0
            }

        wins = sum(1 for t in agent_trades if t.get("won", False))
        profits = [t.get("profit_eur", 0) for t in agent_trades]

        return {
            "agent_id": agent_id,
            "total_trades": len(agent_trades),
            "winning_trades": wins,
            "losing_trades": len(agent_trades) - wins,
            "win_rate": round(wins / len(agent_trades) * 100, 1) if agent_trades else 0,
            "total_profit": round(sum(profits), 2),
            "avg_profit": round(sum(profits) / len(profits), 2) if profits else 0,
            "best_trade": round(max(profits), 2) if profits else 0,
            "worst_trade": round(min(profits), 2) if profits else 0
        }

    def get_global_stats(self, days: int = 7) -> Dict:
        """Calcule les stats globales"""
        fibo1_stats = self.get_agent_stats("fibo1", days)
        fibo_stats = self.get_agent_stats("fibo2", days)
        fibo3_stats = self.get_agent_stats("fibo3", days)

        total_trades = (fibo1_stats["total_trades"] +
                       fibo_stats["total_trades"] +
                       fibo3_stats["total_trades"])

        total_wins = (fibo1_stats.get("winning_trades", 0) +
                     fibo_stats.get("winning_trades", 0) +
                     fibo3_stats.get("winning_trades", 0))

        total_profit = (fibo1_stats["total_profit"] +
                       fibo_stats["total_profit"] +
                       fibo3_stats["total_profit"])

        return {
            "period_days": days,
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_trades - total_wins,
            "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "total_profit": round(total_profit, 2),
            "agents": {
                "fibo1": fibo1_stats,
                "fibo2": fibo_stats,
                "fibo3": fibo3_stats
            }
        }


# Singleton
_logger_instance = None

def get_logger() -> G12Logger:
    """Retourne l'instance Logger singleton"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = G12Logger()
    return _logger_instance
