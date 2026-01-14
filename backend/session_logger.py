# -*- coding: utf-8 -*-
"""
G12 - Session Logger
Gere les sessions de trading et les logs
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import uuid

# Chemins
DATABASE_DIR = Path(__file__).parent / "database"
SESSIONS_DIR = DATABASE_DIR / "sessions"
SESSION_FILE = DATABASE_DIR / "session.json"

# Creer les dossiers
SESSIONS_DIR.mkdir(exist_ok=True)


class SessionLogger:
    """Gestionnaire de sessions de trading"""

    def __init__(self):
        self.session_id = None
        self.start_time = None
        self.trades = []
        self.decisions = []
        self.performance_history = {}  # agent_id -> list of snapshots
        self.balance_start = 0
        self.balance_end = 0
        self._load_session()

    def _load_session(self):
        """Charge la session active depuis session.json"""
        try:
            if SESSION_FILE.exists():
                with open(SESSION_FILE, 'r') as f:
                    data = json.load(f)
                    self.session_id = data.get('id')
                    self.start_time = data.get('start_time')
                    self.trades = data.get('trades', [])
                    self.decisions = data.get('decisions', [])
                    self.performance_history = data.get('performance_history', {})
                    self.balance_start = data.get('balance_start', 0)
                    # Verifier que la session est valide
                    if self.session_id:
                        print(f"[SessionLogger] Session chargee: {self.session_id} ({len(self.trades)} trades)")
                        print(f"[SessionLogger] Performance history: {list(self.performance_history.keys())}")
                    else:
                        print("[SessionLogger] session.json invalide, aucune session active")
                        # Ne pas demarrer automatiquement - laisser l'utilisateur choisir
            else:
                # Pas de session.json - chercher la session la plus recente dans sessions/
                print("[SessionLogger] session.json non trouve, recherche dans sessions/...")
                self._load_latest_session()
        except Exception as e:
            print(f"[SessionLogger] Erreur chargement session: {e}")
            # Ne pas demarrer automatiquement en cas d'erreur

    def _load_latest_session(self):
        """Charge la session la plus recente depuis le dossier sessions/"""
        try:
            session_files = sorted(SESSIONS_DIR.glob("session_*.json"), reverse=True)
            if session_files:
                # Charger la plus recente
                with open(session_files[0], 'r') as f:
                    data = json.load(f)
                    # Verifier si la session n'est pas terminee (pas de end_time)
                    if not data.get('end_time'):
                        self.session_id = data.get('session_id') or data.get('id')
                        self.start_time = data.get('start_time')
                        self.trades = data.get('trades', [])
                        self.decisions = data.get('decisions', [])
                        self.performance_history = data.get('performance_history', {})
                        self.balance_start = data.get('balance_start', 0)
                        self._save_session()  # Re-sauvegarder dans session.json
                        print(f"[SessionLogger] Session restauree: {self.session_id}")
                    else:
                        # Derniere session terminee - ne pas demarrer automatiquement
                        print(f"[SessionLogger] Derniere session terminee, aucune session active")
                        print("[SessionLogger] Cliquez sur 'Nouvelle' pour demarrer une session")
            else:
                # Aucune session trouvee - ne pas demarrer automatiquement
                print("[SessionLogger] Aucune session trouvee")
                print("[SessionLogger] Cliquez sur 'Nouvelle' pour demarrer une session")
        except Exception as e:
            print(f"[SessionLogger] Erreur chargement derniere session: {e}")

    def _save_session(self):
        """Sauvegarde la session active"""
        try:
            data = {
                'id': self.session_id,
                'start_time': self.start_time,
                'trades': self.trades,
                'decisions': self.decisions[-100:],  # Garder 100 dernieres decisions
                'performance_history': self.performance_history,
                'balance_start': self.balance_start,
                'updated_at': datetime.now().isoformat()
            }
            with open(SESSION_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[SessionLogger] Erreur sauvegarde session: {e}")

    def _start_new_session(self):
        """Demarre une nouvelle session"""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now().isoformat()
        self.trades = []
        self.performance_history = {
            "fibo1": [],
            "fibo2": [],
            "fibo3": [],
            "master": []
        }
        self._save_session()
        print(f"[SessionLogger] Nouvelle session: {self.session_id}")

    def start_session(self, balance: float = 0, reset_stats: bool = True) -> Dict:
        """Demarre une nouvelle session (appel API)"""
        # Archiver l'ancienne session si elle existe
        if self.session_id and len(self.trades) > 0:
            self.end_session()

        self._start_new_session()
        self.balance_start = balance
        self._save_session()

        # RESET COMPLET: Nettoyer les donnees historiques pour les stats
        if reset_stats:
            self._reset_statistics()

        return {
            'success': True,
            'session_id': self.session_id,
            'start_time': self.start_time,
            'stats_reset': reset_stats
        }

    def _reset_statistics(self):
        """Reinitialise toutes les statistiques (trades.json et fichiers session)"""
        try:
            # 1. Vider trades.json
            trades_file = DATABASE_DIR / "trades.json"
            with open(trades_file, 'w') as f:
                json.dump({"trades": []}, f, indent=2)
            print("[SessionLogger] trades.json reinitialise")

            # 2. Supprimer les fichiers session_*.json dans sessions/ (pas dans archived/)
            for session_file in SESSIONS_DIR.glob("session_*.json"):
                try:
                    session_file.unlink()
                    print(f"[SessionLogger] Supprime: {session_file.name}")
                except Exception as e:
                    print(f"[SessionLogger] Erreur suppression {session_file}: {e}")

            print("[SessionLogger] Statistiques reinitalisees")
        except Exception as e:
            print(f"[SessionLogger] Erreur reset statistiques: {e}")

    def end_session(self, balance: float = 0) -> Dict:
        """Termine la session et genere le rapport ULTRA COMPLET"""
        if not self.session_id:
            return {'success': False, 'message': 'Aucune session active'}

        # 1. Fermer toutes les positions ouvertes
        closed_positions = self._close_all_positions()

        self.balance_end = balance

        # 2. Collecter TOUTES les donnees
        stats = self._calculate_stats()
        
        # Recuperer les logs du Strategist (description detaillee)
        strategist_actions = self._get_detailed_strategist_actions()
        
        # Analyser les horaires (Best/Worst hours)
        hourly_perf = self._analyze_hourly_performance()
        
        # Force le chargement de TOUTES les decisions de la session
        session_decisions = self._get_global_decisions()

        # 3. Generer le resume "Ultra Complet"
        summary = self._generate_ultra_complete_summary(stats, strategist_actions, hourly_perf)

        # 4. Rapport complet (JSON)
        report = {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'end_time': datetime.now().isoformat(),
            'duration_minutes': self._get_duration_minutes(),
            'balance_start': self.balance_start,
            'balance_end': self.balance_end,
            'pnl': round(self.balance_end - self.balance_start, 2),
            'summary': summary,
            'stats': stats,
            'trades': self.trades,
            'decisions': session_decisions,
            'strategist_actions': strategist_actions,
            'hourly_performance': hourly_perf,
            'closed_positions_at_end': closed_positions
        }

        # 5. Reset COMPLET du site (Archive les anciens fichiers de sessions)
        self._archive_past_sessions()

        # Sauvegarder le nouveau rapport
        log_file = SESSIONS_DIR / f"session_{self.session_id}.json"
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SessionLogger] Erreur sauvegarde JSON: {e}")

        # Sauvegarder le resume texte
        txt_file = SESSIONS_DIR / f"session_{self.session_id}_resume.txt"
        try:
            self._save_ultra_text_summary(txt_file, report)
        except Exception as e:
            print(f"[SessionLogger] Erreur sauvegarde texte: {e}")

        # 6. Reset des logs et stats temps reel
        self._reset_agents_stats()
        self._archive_and_clear_current_logs()

        # Reset state
        old_id = self.session_id
        self.session_id = None
        self.start_time = None
        self.trades = []
        self.decisions = []
        self.performance_history = {}
        
        # Nettoyage fichier actif
        if SESSION_FILE.exists():
            try:
                SESSION_FILE.unlink()
            except Exception:
                pass

        return {
            'success': True,
            'session_id': old_id,
            'log_file': log_file.name,
            'txt_file': txt_file.name,
            'summary': summary
        }

    def _get_detailed_strategist_actions(self) -> List[Dict]:
        """Recupere les actions reelles executees par le Strategist"""
        try:
            from strategist import get_strategist
            strat = get_strategist()
            all_actions = strat.get_executed_actions(limit=100)
            if self.start_time:
                return [a for a in all_actions if a.get('timestamp', '') >= self.start_time]
            return all_actions
        except Exception:
            return []

    def _analyze_hourly_performance(self) -> Dict:
        """Analyse le profit par heure de la journee"""
        hourly = {} # hour -> profit
        for t in self.trades:
            try:
                dt = datetime.fromisoformat(t.get('timestamp', ''))
                h = dt.hour
                hourly[h] = hourly.get(h, 0) + t.get('profit', 0)
            except (ValueError, TypeError):
                continue
        return {str(k): round(v, 2) for k, v in sorted(hourly.items())}

    def _generate_ultra_complete_summary(self, stats: Dict, actions: List, hourly: Dict) -> Dict:
        """Genere l'analyse fine de la session"""
        
        # Best/Worst Agent
        best_agent = None
        worst_agent = None
        if stats.get('by_agent'):
            agents = stats['by_agent']
            best_agent = max(agents.keys(), key=lambda x: agents[x].get('profit', -999999))
            worst_agent = min(agents.keys(), key=lambda x: agents[x].get('profit', 999999))

        # Best Hour
        best_hour = None
        if hourly:
            best_hour = max(hourly.keys(), key=lambda x: hourly[x])

        # Points forts / faibles
        strong_points = []
        weak_points = []
        
        if stats.get('win_rate', 0) > 55: strong_points.append("Excellent Win Rate global")
        elif stats.get('win_rate', 0) < 40: weak_points.append("Win Rate faible, nécessite l'ajustement des filtres")
        
        for agent, data in stats.get('by_agent', {}).items():
            if data.get('profit', 0) > 10: strong_points.append(f"Agent {agent} très rentable")
            if data.get('wins', 0) == 0 and data.get('trades', 0) > 2: weak_points.append(f"Agent {agent} en échec total")

        return {
            'best_agent': best_agent,
            'worst_agent': worst_agent,
            'best_hour': best_hour,
            'strong_points': strong_points,
            'weak_points': weak_points,
            'strategist_changes_count': len(actions),
            'global_win_rate': stats.get('win_rate', 0),
            'total_profit': stats.get('total_profit', 0)
        }

    def _archive_past_sessions(self):
        """Deplace les anciens fichiers de sessions dans un sous-dossier pour reset les stats du site"""
        archived_dir = SESSIONS_DIR / "archived"
        archived_dir.mkdir(exist_ok=True)
        
        for f in SESSIONS_DIR.iterdir():
            if f.is_file() and f.name.startswith("session_") and f.suffix in [".json", ".txt"]:
                try:
                    import shutil
                    shutil.move(str(f), str(archived_dir / f.name))
                except Exception:
                    pass

    def _save_ultra_text_summary(self, filepath: Path, report: Dict):
        """Sauvegarde le rapport ultra complet en format texte"""
        s = report['summary']
        stats = report['stats']
        
        lines = [
            "="*60,
            "RAPPORT DE SESSION ULTRA COMPLET - G12",
            "="*60,
            f"ID: {report['session_id']}",
            f"Durée: {report['duration_minutes']} min",
            f"P&L Final: {report['pnl']:+.2f} EUR",
            f"Win Rate: {s['global_win_rate']}% ({stats['total_trades']} trades)",
            "",
            "--- ANALYSE DE PERFORMANCE ---",
            f"Meilleur Agent: {s['best_agent']}",
            f"Pire Agent: {s['worst_agent']}",
            f"Meilleur Horaire: {s['best_hour']}h",
            "",
            "--- POINTS FORTS ---"
        ]
        
        if s['strong_points']:
            lines.extend([f"+ {p}" for p in s['strong_points']])
        else:
            lines.append("Aucun point fort marqué")
            
        lines.append("")
        lines.append("--- POINTS FAIBLES ---")
        if s['weak_points']:
            lines.extend([f"- {p}" for p in s['weak_points']])
        else:
            lines.append("Aucune faiblesse majeure détectée")
            
        lines.append("")
        lines.append("--- MODIFICATIONS STRATÉGISTE ---")
        lines.append(f"Total d'actions exécutées: {s['strategist_changes_count']}")
        
        for a in report['strategist_actions']:
            lines.append(f"  [{a['timestamp'][11:16]}] {a['type']}: {a['reason']}")

        lines.extend([
            "",
            "--- DETAILS PAR AGENT ---"
        ])
        for agent, data in stats.get('by_agent', {}).items():
            lines.append(f"  {agent.upper()}: {data['profit']:+.2f} EUR | {data['wins']}/{data['trades']} wins")

        lines.extend(["", "="*60, "FIN DU RAPPORT", "="*60])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _archive_and_clear_current_logs(self):
        """Archive trades.json, decisions.json et strategist_logs.json"""
        import shutil
        archive_dir = SESSIONS_DIR / f"archive_{self.session_id}"
        archive_dir.mkdir(exist_ok=True)
        
        mapping = {
            "trades.json": {"trades": []},
            "decisions.json": {"decisions": []},
            "strategist_logs.json": {"logs": []},
            "strategist_actions_history.json": {"updated_at": None, "actions": {}}
        }
        
        for fname, empty in mapping.items():
            fpath = DATABASE_DIR / fname
            if fpath.exists():
                try:
                    shutil.copy2(fpath, archive_dir / fname)
                    with open(fpath, 'w') as f:
                        json.dump(empty, f, indent=2)
                except Exception:
                    pass

    def _get_strategist_logs(self) -> List[Dict]:
        """Recupere tous les logs du Strategist pour cette session"""
        try:
            logs_file = DATABASE_DIR / "strategist_logs.json"
            if logs_file.exists():
                with open(logs_file, 'r') as f:
                    data = json.load(f)
                    logs = data.get('logs', [])
                    # Filtrer les logs de cette session (apres start_time)
                    if self.start_time:
                        return [l for l in logs if l.get('timestamp', '') >= self.start_time]
                    return logs
        except Exception as e:
            print(f"[SessionLogger] Erreur lecture logs Strategist: {e}")
        return []

    def _get_global_decisions(self) -> List[Dict]:
        """Recupere les decisions globales depuis decisions.json"""
        try:
            decisions_file = DATABASE_DIR / "decisions.json"
            if decisions_file.exists():
                with open(decisions_file, 'r') as f:
                    data = json.load(f)
                    decisions = data.get('decisions', [])
                    # Filtrer les decisions de cette session
                    if self.start_time:
                        return [d for d in decisions if d.get('timestamp', '') >= self.start_time]
                    return decisions
        except Exception as e:
            print(f"[SessionLogger] Erreur lecture decisions globales: {e}")
        return []

    def _generate_detailed_summary(self, stats: Dict, strategist_logs: List) -> Dict:
        """Genere un resume detaille de la session"""
        # Compter les decisions par type
        decisions_by_type = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        decisions_by_agent = {}
        for d in self.decisions:
            decision = d.get('decision', 'HOLD')
            agent = d.get('agent', 'unknown')
            decisions_by_type[decision] = decisions_by_type.get(decision, 0) + 1
            if agent not in decisions_by_agent:
                decisions_by_agent[agent] = {'total': 0, 'BUY': 0, 'SELL': 0, 'HOLD': 0}
            decisions_by_agent[agent]['total'] += 1
            decisions_by_agent[agent][decision] = decisions_by_agent[agent].get(decision, 0) + 1

        # Compter les actions du Strategist
        strategist_actions = len([l for l in strategist_logs if l.get('type') == 'ACTION_EXECUTED'])

        return {
            'duration_hours': round(self._get_duration_minutes() / 60, 2),
            'total_trades': stats.get('total_trades', 0),
            'win_rate': stats.get('win_rate', 0),
            'total_pnl': stats.get('total_profit', 0),
            'best_trade': stats.get('max_profit', 0),
            'worst_trade': stats.get('max_loss', 0),
            'total_decisions': len(self.decisions),
            'decisions_by_type': decisions_by_type,
            'decisions_by_agent': decisions_by_agent,
            'strategist_actions': strategist_actions,
            'agents_performance': stats.get('by_agent', {})
        }

    def _save_text_summary(self, filepath: Path, report: Dict, summary: Dict):
        """Sauvegarde un resume lisible en texte"""
        lines = [
            "=" * 60,
            f"SESSION G12 - RESUME COMPLET",
            "=" * 60,
            f"",
            f"Session ID: {report['session_id']}",
            f"Debut: {report['start_time']}",
            f"Fin: {report['end_time']}",
            f"Duree: {summary['duration_hours']} heures",
            f"",
            "-" * 40,
            "RESULTATS FINANCIERS",
            "-" * 40,
            f"Capital debut: {report['balance_start']:.2f} EUR",
            f"Capital fin: {report['balance_end']:.2f} EUR",
            f"P&L Session: {report['pnl']:+.2f} EUR",
            f"",
            "-" * 40,
            "STATISTIQUES TRADES",
            "-" * 40,
            f"Total trades: {summary['total_trades']}",
            f"Win Rate: {summary['win_rate']}%",
            f"Meilleur trade: {summary['best_trade']:+.2f} EUR",
            f"Pire trade: {summary['worst_trade']:+.2f} EUR",
            f"",
            "-" * 40,
            "DECISIONS IA",
            "-" * 40,
            f"Total decisions: {summary['total_decisions']}",
            f"  - BUY: {summary['decisions_by_type'].get('BUY', 0)}",
            f"  - SELL: {summary['decisions_by_type'].get('SELL', 0)}",
            f"  - HOLD: {summary['decisions_by_type'].get('HOLD', 0)}",
            f"",
            "-" * 40,
            "PERFORMANCE PAR AGENT",
            "-" * 40,
        ]

        for agent, data in summary.get('agents_performance', {}).items():
            wr = round(data.get('wins', 0) / data.get('trades', 1) * 100, 1) if data.get('trades', 0) > 0 else 0
            lines.append(f"{agent.upper()}: {data.get('trades', 0)} trades, P&L: {data.get('profit', 0):+.2f} EUR, WR: {wr}%")

        lines.extend([
            f"",
            "-" * 40,
            "STRATEGIST",
            "-" * 40,
            f"Actions executees: {summary['strategist_actions']}",
            f"",
            "-" * 40,
            "LISTE DES TRADES",
            "-" * 40,
        ])

        for i, trade in enumerate(report.get('trades', []), 1):
            lines.append(f"{i}. {trade.get('agent', '?')} | {trade.get('direction', '?')} | {trade.get('profit', 0):+.2f} EUR | {trade.get('close_reason', '?')}")

        lines.extend([
            f"",
            "=" * 60,
            "FIN DU RESUME",
            "=" * 60,
        ])

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _close_all_positions(self) -> List[Dict]:
        """Ferme toutes les positions ouvertes sur TOUS les comptes agents"""
        closed = []
        try:
            from core.mt5_connector import get_mt5

            # Parcourir tous les comptes agents
            for agent_id in ["fibo1", "fibo2", "fibo3"]:
                try:
                    mt5 = get_mt5(agent_id)
                    if not mt5.connect():
                        print(f"[SessionLogger] Impossible de connecter {agent_id}")
                        continue

                    positions = mt5.get_positions()
                    if positions:
                        print(f"[SessionLogger] Fermeture de {len(positions)} position(s) sur {agent_id}...")
                        for pos in positions:
                            # Verifier si c'est une position G12
                            if pos.get("magic") != 11 and "G12" not in pos.get("comment", ""):
                                continue

                            ticket = pos.get('ticket')
                            result = mt5.close_position(ticket)
                            if result:
                                profit = result.get('profit', 0)
                                closed.append({
                                    'ticket': ticket,
                                    'symbol': pos.get('symbol'),
                                    'direction': pos.get('type'),
                                    'profit': profit,
                                    'agent': agent_id,
                                    'closed_reason': 'session_end'
                                })
                                # Logger le trade ferme
                                self.log_trade({
                                    'ticket': ticket,
                                    'agent': agent_id,
                                    'direction': pos.get('type'),
                                    'profit': profit,
                                    'close_reason': 'Fin de session'
                                })
                                print(f"[SessionLogger] Position {ticket} fermee ({agent_id}): {profit:+.2f} EUR")
                            else:
                                print(f"[SessionLogger] ERREUR fermeture position {ticket}")
                except Exception as e:
                    print(f"[SessionLogger] Erreur fermeture {agent_id}: {e}")

            if not closed:
                print("[SessionLogger] Aucune position G12 ouverte a fermer")
        except Exception as e:
            print(f"[SessionLogger] Erreur fermeture positions: {e}")

        return closed

    def _reset_agents_stats(self):
        """Reset les stats de session de tous les agents"""
        try:
            # Importer les agents et reset leurs stats
            from core.trading_loop import get_trading_loop
            trading_loop = get_trading_loop()
            for agent_id, agent in trading_loop.agents.items():
                agent.reset_session_stats()  # Utilise la nouvelle methode qui persiste
                agent.cooldown_until = None
                print(f"[SessionLogger] Stats {agent_id} reset et persiste")
        except Exception as e:
            print(f"[SessionLogger] Erreur reset agents: {e}")

    def _archive_and_clear_logs(self):
        """Archive et vide les fichiers de logs pour une session vierge"""
        import shutil

        # Liste des fichiers a archiver et vider
        files_to_clear = [
            ("trades.json", {"trades": []}),
            ("decisions.json", {"decisions": []}),
            ("strategist_logs.json", {"logs": [], "last_run": None})
        ]

        session_archive_dir = SESSIONS_DIR / f"archive_{self.session_id}"

        try:
            # Creer le dossier d'archive de cette session
            session_archive_dir.mkdir(exist_ok=True)

            for filename, empty_content in files_to_clear:
                filepath = DATABASE_DIR / filename
                if filepath.exists():
                    # Archiver le fichier
                    archive_path = session_archive_dir / filename
                    shutil.copy2(filepath, archive_path)
                    print(f"[SessionLogger] Archive: {filename} -> {archive_path.name}")

                    # Vider le fichier
                    with open(filepath, 'w') as f:
                        json.dump(empty_content, f, indent=2)
                    print(f"[SessionLogger] Reset: {filename}")

            print(f"[SessionLogger] Logs archives dans: {session_archive_dir}")

        except Exception as e:
            print(f"[SessionLogger] Erreur archivage logs: {e}")

    def _calculate_stats(self) -> Dict:
        """Calcule les statistiques de la session"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'by_agent': {}
            }

        profits = [t.get('profit', 0) for t in self.trades]
        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p < 0]

        # Stats par agent
        by_agent = {}
        for trade in self.trades:
            agent = trade.get('agent', 'unknown')
            if agent not in by_agent:
                by_agent[agent] = {'trades': 0, 'profit': 0, 'wins': 0}
            by_agent[agent]['trades'] += 1
            by_agent[agent]['profit'] += trade.get('profit', 0)
            if trade.get('profit', 0) > 0:
                by_agent[agent]['wins'] += 1

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': round(len(winning) / len(self.trades) * 100, 1) if self.trades else 0,
            'total_profit': round(sum(profits), 2),
            'avg_profit': round(sum(profits) / len(self.trades), 2) if self.trades else 0,
            'max_profit': round(max(profits), 2) if profits else 0,
            'max_loss': round(min(profits), 2) if profits else 0,
            'by_agent': by_agent
        }

    def _get_duration_minutes(self) -> int:
        """Calcule la duree de la session en minutes"""
        if not self.start_time:
            return 0
        try:
            start = datetime.fromisoformat(self.start_time)
            duration = datetime.now() - start
            return int(duration.total_seconds() / 60)
        except (ValueError, TypeError):
            return 0

    def log_trade(self, trade: Dict):
        """Enregistre un trade dans la session active"""
        if not self.session_id:
            print(f"[SessionLogger] WARN: Pas de session active, trade non enregistre dans session.json")
            return

        trade['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade)
        self._save_session()
        print(f"[SessionLogger] Trade logged ({len(self.trades)} total): {trade.get('direction')} profit={trade.get('profit')}")

    def sync_with_mt5_history(self) -> Dict:
        """
        Synchronise les trades de la session avec l'historique MT5.
        Recupere tous les trades fermes depuis le debut de la session
        et ajoute ceux qui ne sont pas encore dans trades.json.
        JAMAIS DE DONNEES FICTIVES - uniquement les vrais trades MT5.
        """
        if not self.session_id or not self.start_time:
            return {'success': False, 'message': 'Aucune session active'}

        # Import ici pour eviter import circulaire
        from core.mt5_connector import get_mt5

        # Convertir start_time en datetime
        try:
            session_start = datetime.fromisoformat(self.start_time)
        except (ValueError, TypeError) as e:
            return {'success': False, 'message': f'Date de session invalide: {e}'}

        print(f"[SessionLogger] Sync MT5 depuis {session_start.isoformat()}")

        # Tickets deja enregistres
        existing_tickets = set()
        for trade in self.trades:
            ticket = trade.get('ticket')
            if ticket:
                existing_tickets.add(int(ticket))

        print(f"[SessionLogger] {len(existing_tickets)} trades deja enregistres")

        # Stats par agent (pour mise a jour des stats_*.json)
        stats_by_agent = {
            'fibo1': {'pnl': 0, 'trades': 0, 'wins': 0},
            'fibo2': {'pnl': 0, 'trades': 0, 'wins': 0},
            'fibo3': {'pnl': 0, 'trades': 0, 'wins': 0}
        }

        new_trades_count = 0
        total_synced_pnl = 0

        # Pour chaque agent, recuperer les deals MT5
        for agent_id in ['fibo1', 'fibo2', 'fibo3']:
            try:
                mt5 = get_mt5(agent_id)
                if not mt5.connect():
                    print(f"[SessionLogger] Impossible de connecter {agent_id}")
                    continue

                # Recuperer les deals depuis le debut de la session
                deals = mt5.get_history_deals(from_date=session_start)
                print(f"[SessionLogger] {agent_id}: {len(deals)} deals trouves dans MT5")

                for deal in deals:
                    ticket = deal.get('ticket')
                    if not ticket:
                        continue

                    # Verifier si ce trade est deja enregistre
                    if int(ticket) in existing_tickets:
                        # Deja enregistre, mais compter pour les stats
                        profit = deal.get('profit', 0)
                        stats_by_agent[agent_id]['pnl'] += profit
                        stats_by_agent[agent_id]['trades'] += 1
                        if profit > 0:
                            stats_by_agent[agent_id]['wins'] += 1
                        continue

                    # Nouveau trade - l'ajouter
                    profit = deal.get('profit', 0)
                    trade_data = {
                        'agent': agent_id,
                        'ticket': ticket,
                        'direction': deal.get('type', 'UNKNOWN'),
                        'volume': deal.get('volume', 0),
                        'entry_price': 0,  # Non disponible dans deal
                        'exit_price': deal.get('price', 0),
                        'profit': profit,
                        'close_reason': 'MT5_SYNC',
                        'timestamp': datetime.fromtimestamp(deal.get('time', 0)).isoformat() if deal.get('time') else datetime.now().isoformat()
                    }

                    self.trades.append(trade_data)
                    existing_tickets.add(int(ticket))
                    new_trades_count += 1
                    total_synced_pnl += profit

                    # Mettre a jour les stats
                    stats_by_agent[agent_id]['pnl'] += profit
                    stats_by_agent[agent_id]['trades'] += 1
                    if profit > 0:
                        stats_by_agent[agent_id]['wins'] += 1

                    print(f"[SessionLogger] SYNC: {agent_id} ticket #{ticket} profit={profit:+.2f} EUR")

            except Exception as e:
                print(f"[SessionLogger] Erreur sync {agent_id}: {e}")

        # Sauvegarder les trades mis a jour
        if new_trades_count > 0:
            self._save_session()

        # Mettre a jour les fichiers stats_*.json avec les VRAIS chiffres MT5
        for agent_id, stats in stats_by_agent.items():
            stats_file = DATABASE_DIR / f"stats_{agent_id}.json"
            try:
                stats_data = {
                    'agent_id': agent_id,
                    'updated_at': datetime.now().isoformat(),
                    'session_pnl': round(stats['pnl'], 2),
                    'session_trades': stats['trades'],
                    'session_wins': stats['wins']
                }
                with open(stats_file, 'w') as f:
                    json.dump(stats_data, f, indent=2)
                print(f"[SessionLogger] Stats {agent_id} mis a jour: PnL={stats['pnl']:+.2f} EUR, {stats['trades']} trades, {stats['wins']} wins")
            except Exception as e:
                print(f"[SessionLogger] Erreur sauvegarde stats {agent_id}: {e}")

        # IMPORTANT: Recharger les stats des agents en memoire apres sync
        # Sinon agent.session_pnl reste a 0 et les graphiques sont incorrects
        try:
            from core.trading_loop import get_trading_loop
            trading_loop = get_trading_loop()
            if trading_loop and trading_loop.agents:
                for agent_id, agent in trading_loop.agents.items():
                    agent._load_session_stats()
                print(f"[SessionLogger] Stats agents recharges en memoire")
        except Exception as e:
            print(f"[SessionLogger] Erreur rechargement stats agents: {e}")

        result = {
            'success': True,
            'new_trades': new_trades_count,
            'total_synced_pnl': round(total_synced_pnl, 2),
            'total_trades': len(self.trades),
            'stats_by_agent': stats_by_agent
        }

        print(f"[SessionLogger] Sync termine: {new_trades_count} nouveaux trades, PnL total sync: {total_synced_pnl:+.2f} EUR")
        return result

    def log_performance_snapshot(self, agent_id: str, closed_pnl: float, floating_pnl: float):
        """Enregistre un snapshot de performance pour un agent et met a jour le Master"""
        if not self.session_id:
            return

        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []

        timestamp = datetime.now().isoformat()
        snapshot = {
            'timestamp': timestamp,
            'closed_pnl': round(closed_pnl, 2),
            'floating_pnl': round(floating_pnl, 2)
        }
        
        self.performance_history[agent_id].append(snapshot)
        
        # Limiter a 500 points
        if len(self.performance_history[agent_id]) > 500:
            self.performance_history[agent_id] = self.performance_history[agent_id][-500:]

        # --- MISE A JOUR MASTER ---
        # Somme des derniers snapshots de chaque agent
        total_closed = 0
        total_floating = 0
        
        for aid in ["fibo1", "fibo2", "fibo3"]:
            if aid in self.performance_history and self.performance_history[aid]:
                last = self.performance_history[aid][-1]
                total_closed += last.get('closed_pnl', 0)
                total_floating += last.get('floating_pnl', 0)
        
        if "master" not in self.performance_history:
            self.performance_history["master"] = []
            
        self.performance_history["master"].append({
            'timestamp': timestamp,
            'closed_pnl': round(total_closed, 2),
            'floating_pnl': round(total_floating, 2)
        })
        
        if len(self.performance_history["master"]) > 500:
            self.performance_history["master"] = self.performance_history["master"][-500:]
            
        # Sauvegarder pour persistance (refresh page)
        self._save_session()

        # Sauvegarder periodiquement
        if len(self.performance_history[agent_id]) % 5 == 0:
            self._save_session()

    def log_decision(self, agent: str, decision: str, reason: str, context: Dict = None):
        """Enregistre une decision d'agent"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': agent,
            'decision': decision,
            'reason': reason,
            'context': context or {}
        }
        self.decisions.append(entry)
        # Sauvegarder toutes les 10 decisions
        if len(self.decisions) % 10 == 0:
            self._save_session()

    def get_status(self) -> Dict:
        """Retourne le status de la session"""
        # Si pas de session active, retourner un status vide mais informatif
        if not self.session_id:
            return {
                'id': None,
                'active': False,
                'message': 'Aucune session active. Cliquez sur "Nouvelle" pour demarrer.',
                'start_time': None,
                'duration_minutes': 0,
                'trades_count': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'by_agent': {},
                'balance_start': 0
            }

        # SOURCE UNIQUE DE VERITE: stats_*.json (synchronises avec MT5)
        # Evite la divergence entre self.trades et les stats des agents
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        by_agent = {}

        for agent_id in ['fibo1', 'fibo2', 'fibo3']:
            stats_file = DATABASE_DIR / f"stats_{agent_id}.json"
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        agent_stats = json.load(f)
                    pnl = agent_stats.get('session_pnl', 0)
                    trades = agent_stats.get('session_trades', 0)
                    wins = agent_stats.get('session_wins', 0)
                    total_pnl += pnl
                    total_trades += trades
                    total_wins += wins
                    by_agent[agent_id] = {
                        'trades': trades,
                        'profit': round(pnl, 2),
                        'wins': wins
                    }
                except Exception as e:
                    print(f"[SessionLogger] Erreur lecture stats_{agent_id}.json: {e}")

        win_rate = round((total_wins / total_trades * 100), 1) if total_trades > 0 else 0

        return {
            'id': self.session_id,
            'active': True,
            'start_time': self.start_time,
            'duration_minutes': self._get_duration_minutes(),
            'trades_count': total_trades,
            'total_pnl': round(total_pnl, 2),
            'win_rate': win_rate,
            'by_agent': by_agent,
            'balance_start': self.balance_start
        }

    def export_session(self) -> Dict:
        """Exporte les donnees de la session actuelle"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'exported_at': datetime.now().isoformat(),
            'stats': self._calculate_stats(),
            'trades': self.trades,
            'decisions': self.decisions
        }

    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Retourne l'historique des sessions"""
        sessions = []
        try:
            for f in sorted(SESSIONS_DIR.glob("session_*.json"), reverse=True)[:limit]:
                with open(f, 'r') as file:
                    data = json.load(file)
                    sessions.append({
                        'session_id': data.get('session_id'),
                        'start_time': data.get('start_time'),
                        'end_time': data.get('end_time'),
                        'pnl': data.get('pnl', 0),
                        'trades_count': len(data.get('trades', []))
                    })
        except Exception as e:
            print(f"[SessionLogger] Erreur lecture historique: {e}")
        return sessions


# Singleton
_session_logger = None

def get_session_logger() -> SessionLogger:
    """Retourne l'instance SessionLogger singleton"""
    global _session_logger
    if _session_logger is None:
        _session_logger = SessionLogger()
    return _session_logger
