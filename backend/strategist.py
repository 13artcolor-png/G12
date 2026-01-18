# -*- coding: utf-8 -*-
"""
G12 - Strategist
Analyse les trades et execute des corrections AUTOMATIQUEMENT
Integre la methodologie d'analyse API pour optimisation avancee
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from data.aggregator import get_aggregator
from utils.telegram_service import get_telegram

DATABASE_DIR = Path(__file__).parent / "database"
SESSIONS_DIR = DATABASE_DIR / "sessions"
TRADES_FILE = DATABASE_DIR / "trades.json"
SUGGESTIONS_FILE = DATABASE_DIR / "strategy_suggestions.json"
LOGS_FILE = DATABASE_DIR / "strategist_logs.json"
AGENTS_CONFIG_FILE = DATABASE_DIR / "agents_runtime_config.json"
SPREAD_CONFIG_FILE = DATABASE_DIR / "spread_runtime_config.json"
ACTIONS_HISTORY_FILE = DATABASE_DIR / "strategist_actions_history.json"
LAST_RUN_FILE = DATABASE_DIR / "strategist_last_run.json"
INACTIVITY_STATE_FILE = DATABASE_DIR / "strategist_inactivity_state.json"
METHODOLOGY_FILE = DATABASE_DIR / "API_ANALYSE_METHODOLOGY.json"
SESSION_ANALYSIS_FILE = DATABASE_DIR / "last_session_analysis.json"
STRATEGIST_CONFIG_FILE = DATABASE_DIR / "strategist_runtime_config.json"
ANALYSIS_WEIGHTS_FILE = DATABASE_DIR / "analysis_weights_config.json"

# Intervalles d'optimisation
MIN_TRADES_BETWEEN_OPTIM = 5  # Minimum trades entre 2 optimisations
MIN_SECONDS_BETWEEN_OPTIM = 60  # Minimum secondes entre 2 optimisations
ACTION_COOLDOWN_HOURS = 1  # Cooldown avant de re-executer la meme action
SELF_LEARNING_INTERVAL_HOURS = 6  # Intervalle pour le self-learning
SESSION_ANALYSIS_INTERVAL_HOURS = 1  # Analyse des sessions toutes les heures

# URL API Requesty (pour appels IA)
REQUESTY_URL = "https://router.requesty.ai/v1/chat/completions"

class Strategist:
    """Analyse les performances et execute des corrections AUTOMATIQUEMENT"""

    def __init__(self):
        self.trades = []
        self.suggestions = []
        self._last_optimization_time = 0
        self._last_trades_count = 0
        self._last_inactivity_reduction_time = self._load_inactivity_state()  # Persiste entre redemarrages
        self._executed_action_hashes: Dict[str, float] = {}  # hash -> timestamp
        self._last_self_learning_run = 0  # Intervalle self-learning
        self._last_session_analysis_run = 0  # Intervalle analyse sessions
        self._methodology = None  # Methodologie d'analyse chargee
        self._session_analysis_done_at_startup = False  # Flag pour analyse au demarrage
        self._load_trades()
        self._load_action_history()
        self._load_last_run_info()
        self._load_methodology()

        # Analyse des sessions passees au demarrage
        self._run_startup_session_analysis()

    def _load_action_history(self):
        """Charge l'historique des actions executees pour eviter les repetitions"""
        try:
            if ACTIONS_HISTORY_FILE.exists():
                with open(ACTIONS_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    # Convertir en dict hash -> timestamp
                    self._executed_action_hashes = data.get('actions', {})
                    print(f"[Strategist] {len(self._executed_action_hashes)} actions en memoire")
        except Exception as e:
            print(f"[Strategist] Erreur chargement historique actions: {e}")
            self._executed_action_hashes = {}

    def _save_action_history(self):
        """Sauvegarde l'historique des actions"""
        try:
            # Nettoyer les actions trop anciennes (> 24h)
            cutoff = time.time() - (24 * 3600)
            self._executed_action_hashes = {
                h: t for h, t in self._executed_action_hashes.items()
                if t > cutoff
            }

            with open(ACTIONS_HISTORY_FILE, 'w') as f:
                json.dump({
                    'updated_at': datetime.now().isoformat(),
                    'actions': self._executed_action_hashes
                }, f, indent=2)
        except Exception as e:
            print(f"[Strategist] Erreur sauvegarde historique: {e}")

    def _load_inactivity_state(self) -> float:
        """Charge le timer d'inactivite depuis le fichier (persiste entre redemarrages)"""
        try:
            if INACTIVITY_STATE_FILE.exists():
                with open(INACTIVITY_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    last_time = data.get('last_inactivity_reduction_time', 0)
                    # Aussi restaurer le compteur de trades
                    self._last_trades_count = data.get('last_trades_count', 0)
                    if last_time > 0:
                        print(f"[Strategist] Timer d'inactivite restaure: {int((time.time() - last_time)/60)} min d'inactivite, {self._last_trades_count} trades")
                        return last_time
        except Exception as e:
            print(f"[Strategist] Erreur chargement etat inactivite: {e}")
        # Par defaut, utiliser le temps actuel (nouveau demarrage)
        return time.time()

    def _save_inactivity_state(self):
        """Sauvegarde le timer d'inactivite dans un fichier"""
        try:
            with open(INACTIVITY_STATE_FILE, 'w') as f:
                json.dump({
                    'last_inactivity_reduction_time': self._last_inactivity_reduction_time,
                    'last_trades_count': self._last_trades_count,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"[Strategist] Erreur sauvegarde etat inactivite: {e}")

    def _load_last_run_info(self):
        """Charge les informations de la derniere execution (pour le self-learning)"""
        try:
            if LAST_RUN_FILE.exists():
                with open(LAST_RUN_FILE, 'r') as f:
                    data = json.load(f)
                    self._last_self_learning_run = data.get('last_self_learning_run', 0)
                    self._last_session_analysis_run = data.get('last_session_analysis_run', 0)
        except Exception as e:
            print(f"[Strategist] Erreur chargement last run info: {e}")
            self._last_self_learning_run = 0
            self._last_session_analysis_run = 0

    def _load_methodology(self):
        """Charge la methodologie d'analyse depuis API_ANALYSE_METHODOLOGY.json"""
        try:
            if METHODOLOGY_FILE.exists():
                with open(METHODOLOGY_FILE, 'r', encoding='utf-8') as f:
                    self._methodology = json.load(f)
                print(f"[Strategist] Methodologie d'analyse chargee (v{self._methodology.get('version', '?')})")
            else:
                print(f"[Strategist] WARN: Fichier methodologie non trouve: {METHODOLOGY_FILE}")
                self._methodology = None
        except Exception as e:
            print(f"[Strategist] Erreur chargement methodologie: {e}")
            self._methodology = None

    def _is_session_analysis_enabled(self) -> bool:
        """Verifie si l'utilisation du rapport session est activee dans la config"""
        try:
            if STRATEGIST_CONFIG_FILE.exists():
                with open(STRATEGIST_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    return config.get('use_session_analysis', True)
        except Exception:
            pass
        return True  # Active par defaut

    def _run_startup_session_analysis(self):
        """Execute l'analyse des sessions passees au demarrage du bot"""
        if self._session_analysis_done_at_startup:
            return

        print("[Strategist] Analyse des sessions passees au demarrage...")
        try:
            # Charger les sessions passees
            past_sessions = self._load_past_sessions(limit=5)
            if not past_sessions:
                print("[Strategist] Aucune session passee a analyser")
                self._session_analysis_done_at_startup = True
                return

            # Analyser la derniere session complete
            latest_session = past_sessions[0] if past_sessions else None
            if latest_session:
                print(f"[Strategist] Analyse de la session: {latest_session.get('metadata', {}).get('filename', 'unknown')}")
                analysis_result = self._analyze_session_with_methodology(latest_session)

                if analysis_result:
                    # Toujours sauvegarder l'analyse (rapport genere meme si non utilise)
                    self._save_session_analysis(analysis_result)

                    # Appliquer les recommandations SEULEMENT si active dans la config
                    if self._is_session_analysis_enabled():
                        if analysis_result.get('recommended_actions'):
                            self._apply_session_recommendations(analysis_result)
                    else:
                        print("[Strategist] Utilisation rapport session DESACTIVEE - recommandations non appliquees")

            self._session_analysis_done_at_startup = True
            self._last_session_analysis_run = time.time()
            self._save_last_run_info()
            print("[Strategist] Analyse de demarrage terminee")

        except Exception as e:
            print(f"[Strategist] Erreur analyse demarrage: {e}")
            self._session_analysis_done_at_startup = True

    def _should_run_session_analysis(self) -> bool:
        """Verifie si l'analyse de session doit etre executee (toutes les heures)"""
        current_time = time.time()
        interval_seconds = SESSION_ANALYSIS_INTERVAL_HOURS * 3600
        return (current_time - self._last_session_analysis_run) > interval_seconds

    def _load_past_sessions(self, limit: int = 5) -> List[Dict]:
        """Charge les fichiers de session depuis backend/database/sessions/"""
        sessions = []
        try:
            if not SESSIONS_DIR.exists():
                return sessions

            # Lister les fichiers de session (format: G12_YYYY-MM-DD_*.json)
            session_files = sorted(
                [f for f in SESSIONS_DIR.glob("G12_*.json") if f.is_file()],
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )

            for session_file in session_files[:limit]:
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        sessions.append(session_data)
                except Exception as e:
                    print(f"[Strategist] Erreur lecture session {session_file.name}: {e}")

            print(f"[Strategist] {len(sessions)} sessions passees chargees")
        except Exception as e:
            print(f"[Strategist] Erreur chargement sessions: {e}")

        return sessions

    def _analyze_session_with_methodology(self, session_data: Dict) -> Dict:
        """
        Analyse une session en appliquant la methodologie.
        Calcule les metriques localement et prepare le contexte pour l'API IA.
        """
        if not session_data:
            return {}

        try:
            # Extraire les donnees de la session
            trades = session_data.get('trades', [])
            stats = session_data.get('stats_globales', {})
            stats_par_agent = session_data.get('stats_par_agent', {})
            perf_horaire = session_data.get('performance_horaire', {})
            periode = session_data.get('periode', {})
            resultat = session_data.get('resultat', {})

            if not trades and not stats.get('total_trades', 0):
                print("[Strategist] Session sans trades, skip analyse")
                return {}

            # --- STEP 2: Calcul des metriques fondamentales ---
            profits = [t.get('profit', 0) for t in trades if t.get('profit') is not None]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]

            total_trades = len(profits) or stats.get('total_trades', 0)
            total_wins = len(wins) or stats.get('trades_gagnants', 0)
            total_profit = sum(profits) if profits else resultat.get('pnl_session', 0)

            # Profit Factor
            sum_wins = sum(wins) if wins else 0
            sum_losses = abs(sum(losses)) if losses else 1
            profit_factor = round(sum_wins / sum_losses, 2) if sum_losses > 0 else 0

            # Reward/Risk Ratio
            avg_win = sum_wins / len(wins) if wins else 0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 1
            rr_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

            # Win Rate
            win_rate = round((total_wins / total_trades) * 100, 1) if total_trades > 0 else 0

            # Frequence de trading
            duration_minutes = periode.get('duree_minutes', 60) or 60
            trades_per_minute = round(total_trades / duration_minutes, 3)

            # --- STEP 5: Analyse des raisons de cloture ---
            close_reasons = {}
            for t in trades:
                reason = t.get('close_reason', 'UNKNOWN')
                close_reasons[reason] = close_reasons.get(reason, 0) + 1

            tp_closes_pct = round((close_reasons.get('TP', 0) / total_trades) * 100, 1) if total_trades > 0 else 0
            sl_closes_pct = round((close_reasons.get('SL', 0) / total_trades) * 100, 1) if total_trades > 0 else 0
            sync_closes_pct = round((close_reasons.get('MT5_SYNC', 0) / total_trades) * 100, 1) if total_trades > 0 else 0

            # --- STEP 7: Score de risque ---
            risk_score = min(100, int(
                (1 - min(profit_factor, 2) / 2) * 30 +
                (1 - min(rr_ratio, 2) / 2) * 30 +
                min(trades_per_minute, 2) * 20 +
                sync_closes_pct / 5
            ))

            # --- Identification des problemes critiques ---
            critical_issues = []

            if rr_ratio < 0.8:
                critical_issues.append({
                    'type': 'INVERTED_RR_RATIO',
                    'severity': 'CRITICAL',
                    'description': 'Ratio reward/risk inverse - gains moyens inferieurs aux pertes',
                    'metric_value': rr_ratio,
                    'threshold': 1.0
                })

            if profit_factor < 1.0:
                critical_issues.append({
                    'type': 'LOW_PROFIT_FACTOR',
                    'severity': 'CRITICAL',
                    'description': 'Profit factor < 1.0 = strategie mathematiquement perdante',
                    'metric_value': profit_factor,
                    'threshold': 1.0
                })

            if sync_closes_pct > 50:
                critical_issues.append({
                    'type': 'EXCESSIVE_MT5_SYNC_CLOSES',
                    'severity': 'HIGH',
                    'description': f'{sync_closes_pct}% des trades fermes par sync plutot que TP/SL',
                    'metric_value': sync_closes_pct,
                    'threshold': 20.0
                })

            if trades_per_minute > 1.0:
                critical_issues.append({
                    'type': 'OVERTRADING',
                    'severity': 'HIGH',
                    'description': f'Frequence excessive ({trades_per_minute} trades/min)',
                    'metric_value': trades_per_minute,
                    'threshold': 0.5
                })

            if win_rate < 40:
                critical_issues.append({
                    'type': 'LOW_WIN_RATE',
                    'severity': 'MEDIUM',
                    'description': f'Win rate faible ({win_rate}%)',
                    'metric_value': win_rate,
                    'threshold': 40
                })

            # --- Analyse par agent ---
            agent_analysis = {}
            for agent_id, agent_stats in stats_par_agent.items():
                agent_trades = agent_stats.get('trades', 0)
                agent_profit = agent_stats.get('profit', 0)
                agent_wins = agent_stats.get('wins', 0)
                agent_wr = round((agent_wins / agent_trades) * 100, 1) if agent_trades > 0 else 0
                contribution_pct = round((agent_trades / total_trades) * 100, 1) if total_trades > 0 else 0

                status = 'PROFITABLE' if agent_profit > 0 and agent_wr > 50 else \
                         'MARGINAL' if agent_profit > -10 else \
                         'LOSING' if agent_wr > 45 else 'CRITICAL'

                agent_analysis[agent_id] = {
                    'trades': agent_trades,
                    'profit': agent_profit,
                    'win_rate': agent_wr,
                    'contribution_pct': contribution_pct,
                    'status': status
                }

                # Detection surconcentration
                if contribution_pct > 50 and agent_profit < 0:
                    critical_issues.append({
                        'type': 'AGENT_OVERCONCENTRATION',
                        'severity': 'MEDIUM',
                        'description': f'{agent_id} = {contribution_pct}% des trades mais en perte',
                        'metric_value': contribution_pct,
                        'threshold': 50
                    })

            # --- Analyse horaire ---
            losing_hours = []
            profitable_hours = []
            for hour_str, pnl in perf_horaire.items():
                hour = int(hour_str) if hour_str.isdigit() else 0
                if pnl < -30:
                    losing_hours.append(hour)
                elif pnl > 10:
                    profitable_hours.append(hour)

            # --- Generation des recommandations ---
            recommended_actions = []
            priority = 1

            # Recommandation TP/SL si RR inverse
            if rr_ratio < 0.8:
                current_spread_config = self._load_spread_config()
                current_tp = current_spread_config.get('tp_pct', 0.3)
                current_sl = current_spread_config.get('sl_pct', 0.5)
                recommended_actions.append({
                    'priority': priority,
                    'category': 'TP_SL',
                    'action': 'INVERT_TP_SL_RATIO',
                    'params': {
                        'new_tp_percent': round(max(current_sl, 1.0), 2),
                        'new_sl_percent': round(min(current_tp, 0.5), 2)
                    },
                    'expected_impact': f'Transformer RR de {rr_ratio}:1 vers 2:1 minimum'
                })
                priority += 1

            # Recommandation frequence
            if trades_per_minute > 1.0:
                recommended_actions.append({
                    'priority': priority,
                    'category': 'FREQUENCY',
                    'action': 'REDUCE_TRADING_FREQUENCY',
                    'params': {
                        'increase_cooldown_by': 30,
                        'target_trades_per_minute': 0.3
                    },
                    'expected_impact': 'Reduire overtrading et ameliorer qualite des entrees'
                })
                priority += 1

            # Recommandation agents perdants
            for agent_id, analysis in agent_analysis.items():
                if analysis['status'] == 'CRITICAL':
                    recommended_actions.append({
                        'priority': priority,
                        'category': 'AGENT',
                        'action': 'TIGHTEN_AGENT_PARAMETERS',
                        'params': {
                            'agent': agent_id,
                            'reduce_tolerance_by': 0.5,
                            'increase_momentum_by': 0.02
                        },
                        'expected_impact': f'Reduire les pertes de {agent_id}'
                    })
                    priority += 1

            # Recommandation heures perdantes
            if losing_hours:
                recommended_actions.append({
                    'priority': priority,
                    'category': 'SCHEDULE',
                    'action': 'BLACKLIST_LOSING_HOURS',
                    'params': {
                        'blacklisted_hours': losing_hours,
                        'preferred_hours': profitable_hours
                    },
                    'expected_impact': 'Eviter les heures les plus perdantes'
                })
                priority += 1

            # --- Resultat final ---
            analysis_result = {
                'session_analyzed': session_data.get('metadata', {}).get('filename', 'unknown'),
                'analyzed_at': datetime.now().isoformat(),
                'session_summary': {
                    'pnl': round(total_profit, 2),
                    'profit_factor': profit_factor,
                    'rr_ratio': rr_ratio,
                    'win_rate': win_rate,
                    'risk_score': risk_score,
                    'total_trades': total_trades
                },
                'critical_issues': critical_issues,
                'agent_analysis': agent_analysis,
                'close_reasons': {
                    'TP_pct': tp_closes_pct,
                    'SL_pct': sl_closes_pct,
                    'MT5_SYNC_pct': sync_closes_pct
                },
                'recommended_actions': recommended_actions
            }

            print(f"[Strategist] Analyse session: PF={profit_factor}, RR={rr_ratio}, WR={win_rate}%, Risk={risk_score}")
            print(f"[Strategist] {len(critical_issues)} problemes critiques, {len(recommended_actions)} recommandations")

            return analysis_result

        except Exception as e:
            print(f"[Strategist] Erreur analyse session: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _save_session_analysis(self, analysis: Dict):
        """Sauvegarde l'analyse de session dans un fichier"""
        try:
            with open(SESSION_ANALYSIS_FILE, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"[Strategist] Analyse sauvegardee: {SESSION_ANALYSIS_FILE.name}")
        except Exception as e:
            print(f"[Strategist] Erreur sauvegarde analyse: {e}")

    def _apply_session_recommendations(self, analysis: Dict):
        """
        Applique les recommandations de l'analyse de session.
        Respecte les permissions strategist_permissions des agents.
        """
        if not analysis.get('recommended_actions'):
            return

        print(f"[Strategist] Application de {len(analysis['recommended_actions'])} recommandations...")

        agents_config = self._load_agents_config()
        spread_config = self._load_spread_config()
        applied_count = 0

        for rec in analysis['recommended_actions']:
            category = rec.get('category', '')
            action = rec.get('action', '')
            params = rec.get('params', {})

            try:
                if category == 'TP_SL' and action == 'INVERT_TP_SL_RATIO':
                    # Verifier permission tp_sl sur au moins un agent
                    has_permission = any(
                        self._has_permission(agents_config.get(aid, {}), 'tp_sl')
                        for aid in agents_config if agents_config.get(aid, {}).get('enabled', False)
                    )
                    if has_permission:
                        new_tp = params.get('new_tp_percent', spread_config.get('tp_pct', 0.3))
                        new_sl = params.get('new_sl_percent', spread_config.get('sl_pct', 0.5))

                        # Validation: TP doit etre > SL
                        if new_tp > new_sl:
                            old_tp = spread_config.get('tp_pct')
                            old_sl = spread_config.get('sl_pct')
                            spread_config['tp_pct'] = new_tp
                            spread_config['sl_pct'] = new_sl
                            print(f"[Strategist] TP/SL ajuste: TP {old_tp}% -> {new_tp}%, SL {old_sl}% -> {new_sl}%")
                            self.log_decision("SESSION_RECOMMENDATION", {
                                'action': action,
                                'old_tp': old_tp, 'new_tp': new_tp,
                                'old_sl': old_sl, 'new_sl': new_sl
                            }, "Ajustement TP/SL base sur analyse session")
                            applied_count += 1

                elif category == 'FREQUENCY' and action == 'REDUCE_TRADING_FREQUENCY':
                    # Augmenter le cooldown de tous les agents (avec permission)
                    increase_by = params.get('increase_cooldown_by', 30)
                    for agent_id, config in agents_config.items():
                        if self._has_permission(config, 'cooldown'):
                            old_cd = config.get('cooldown_seconds', 60)
                            config['cooldown_seconds'] = min(300, old_cd + increase_by)
                            print(f"[Strategist] Cooldown {agent_id}: {old_cd}s -> {config['cooldown_seconds']}s")
                    applied_count += 1

                elif category == 'AGENT' and action == 'TIGHTEN_AGENT_PARAMETERS':
                    agent_id = params.get('agent')
                    if agent_id and agent_id in agents_config:
                        config = agents_config[agent_id]

                        # Reduire tolerance (si permission)
                        if self._has_permission(config, 'tolerance'):
                            reduce_by = params.get('reduce_tolerance_by', 0.5)
                            old_tol = config.get('fibo_tolerance_pct', 1.0)
                            config['fibo_tolerance_pct'] = max(0.3, round(old_tol - reduce_by, 2))
                            print(f"[Strategist] Tolerance {agent_id}: {old_tol}% -> {config['fibo_tolerance_pct']}%")

                        # Augmenter momentum min (si permission)
                        if self._has_permission(config, 'momentum'):
                            increase_by = params.get('increase_momentum_by', 0.02)
                            old_mom = config.get('min_momentum_pct', 0.05)
                            config['min_momentum_pct'] = min(0.15, round(old_mom + increase_by, 3))
                            print(f"[Strategist] Momentum {agent_id}: {old_mom}% -> {config['min_momentum_pct']}%")

                        applied_count += 1

            except Exception as e:
                print(f"[Strategist] Erreur application recommandation {action}: {e}")

        # Sauvegarder les configs modifiees
        if applied_count > 0:
            self._save_agents_config(agents_config)
            self._save_spread_config(spread_config)
            print(f"[Strategist] {applied_count} recommandations appliquees")

            # Notification Telegram
            try:
                telegram = get_telegram()
                if telegram:
                    issues_text = "\n".join([f"- {i['type']}: {i['description']}" for i in analysis.get('critical_issues', [])[:3]])
                    telegram.send_message(
                        f"*Strategist - Analyse Session*\n"
                        f"PnL: {analysis['session_summary']['pnl']:.2f} EUR\n"
                        f"Profit Factor: {analysis['session_summary']['profit_factor']}\n"
                        f"Risk Score: {analysis['session_summary']['risk_score']}/100\n\n"
                        f"*Problemes:*\n{issues_text or 'Aucun'}\n\n"
                        f"*{applied_count} ajustements appliques*"
                    )
            except Exception:
                pass

    def run_hourly_session_analysis(self) -> Dict:
        """
        Execute l'analyse de session horaire.
        Appelee par auto_optimize() si l'intervalle est ecoule.
        Le rapport est toujours genere, mais applique seulement si active.
        """
        if not self._should_run_session_analysis():
            return {'status': 'skip', 'reason': 'interval_not_reached'}

        print("[Strategist] Analyse de session horaire...")

        try:
            past_sessions = self._load_past_sessions(limit=3)
            if not past_sessions:
                return {'status': 'skip', 'reason': 'no_sessions'}

            # Analyser la session la plus recente
            latest = past_sessions[0]
            analysis = self._analyze_session_with_methodology(latest)

            recommendations_applied = 0
            if analysis:
                # Toujours sauvegarder l'analyse (rapport toujours genere)
                self._save_session_analysis(analysis)

                # Appliquer les recommandations SEULEMENT si active dans la config
                if self._is_session_analysis_enabled():
                    self._apply_session_recommendations(analysis)
                    recommendations_applied = len(analysis.get('recommended_actions', []))
                else:
                    print("[Strategist] Utilisation rapport session DESACTIVEE - recommandations non appliquees")

            self._last_session_analysis_run = time.time()
            self._save_last_run_info()

            return {
                'status': 'completed',
                'session_analyzed': latest.get('metadata', {}).get('filename'),
                'issues_found': len(analysis.get('critical_issues', [])),
                'recommendations_applied': recommendations_applied,
                'session_analysis_enabled': self._is_session_analysis_enabled()
            }

        except Exception as e:
            print(f"[Strategist] Erreur analyse horaire: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_last_session_analysis(self) -> Dict:
        """Retourne la derniere analyse de session sauvegardee"""
        try:
            if SESSION_ANALYSIS_FILE.exists():
                with open(SESSION_ANALYSIS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[Strategist] Erreur lecture analyse: {e}")
        return {}

    def _save_last_run_info(self):
        """Sauvegarde les informations de la derniere execution (pour le self-learning et analyse sessions)"""
        try:
            with open(LAST_RUN_FILE, 'w') as f:
                json.dump({
                    'last_self_learning_run': self._last_self_learning_run,
                    'last_session_analysis_run': self._last_session_analysis_run,
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"[Strategist] Erreur sauvegarde last run info: {e}")

    def _should_run_self_learning(self) -> bool:
        """Verifie si le self-learning doit etre execute"""
        current_time = time.time()
        if (current_time - self._last_self_learning_run) > (SELF_LEARNING_INTERVAL_HOURS * 3600):
            return True
        return False

    def _mark_self_learning_run(self):
        """Marque le self-learning comme execute"""
        self._last_self_learning_run = time.time()
        self._save_last_run_info()

    def _action_hash(self, action_type: str, target: str, value: str) -> str:
        """Cree un hash unique pour une action"""
        key = f"{action_type}:{target}:{value}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _was_action_executed_recently(self, action_hash: str) -> bool:
        """Verifie si une action a deja ete executee recemment"""
        if action_hash not in self._executed_action_hashes:
            return False

        last_time = self._executed_action_hashes[action_hash]
        cooldown = ACTION_COOLDOWN_HOURS * 3600
        return (time.time() - last_time) < cooldown

    def _mark_action_executed(self, action_hash: str):
        """Marque une action comme executee"""
        self._executed_action_hashes[action_hash] = time.time()
        self._save_action_history()

    def auto_optimize(self) -> Dict:
        """
        METHODE PRINCIPALE - Appelee automatiquement par la trading loop.
        Analyse les performances et execute les corrections necessaires.
        """
        current_time = time.time()
        self._load_trades()

        # Calculate trades since last optimization (if applicable)
        trades_since_last = len(self.trades) - self._last_trades_count if self._last_trades_count > 0 else len(self.trades)

        # Check if optimization is due based on time or new trades
        time_since_last_optim = current_time - self._last_optimization_time
        if time_since_last_optim < MIN_SECONDS_BETWEEN_OPTIM and trades_since_last < MIN_TRADES_BETWEEN_OPTIM:
            # print(f"[Strategist] Skip auto-optimisation: {trades_since_last} trades < {MIN_TRADES_BETWEEN_OPTIM} ou {int(time_since_last_optim)}s < {MIN_SECONDS_BETWEEN_OPTIM}s")
            return {'status': 'skip', 'executed_count': 0, 'actions': []}

        # Executer l'optimisation
        print(f"[Strategist] Auto-optimisation ({len(self.trades)} trades, {trades_since_last} nouveaux)")

        # 1. Corrections basees sur la performance (si assez de trades)
        result = {'status': 'skip', 'executed_count': 0, 'actions': []}
        if len(self.trades) >= 5:
            result = self._execute_auto_corrections()
        else:
            print(f"[Strategist] Skip corrections de performance: trades={len(self.trades)} < 5")

        # 2. Correction d'inactivite (si PAS de trades)
        inactivity_result = self._check_inactivity_and_correct()
        if inactivity_result.get('executed_count', 0) > 0:
            if result['status'] == 'skip':
                result = inactivity_result
                result['status'] = 'optimized'
            else:
                result['executed_count'] += inactivity_result['executed_count']
                result['actions'].extend(inactivity_result['actions'])

        # 3. Verification coherence positions G12 vs MT5
        positions_result = self._check_positions_and_alert()
        if positions_result.get('executed_count', 0) > 0:
            if result['status'] == 'skip':
                result['status'] = 'positions_alert'
            result['executed_count'] += positions_result['executed_count']
            result['actions'].extend(positions_result['actions'])
        # Toujours inclure l'analyse des positions dans le resultat
        result['positions_analysis'] = positions_result.get('analysis', {})

        # 4. Self-Learning (if due)
        if self._should_run_self_learning():
            print("[Strategist] Lancement du Self-Learning...")
            try:
                self.self_learning_optimization()
            except Exception as e:
                print(f"[Strategist] Erreur self-learning: {e}")
            self._mark_self_learning_run()

        # 5. Analyse de session horaire (basee sur methodologie)
        if self._should_run_session_analysis():
            session_result = self.run_hourly_session_analysis()
            if session_result.get('status') == 'completed':
                result['session_analysis'] = session_result
                if result['status'] == 'skip':
                    result['status'] = 'session_analyzed'

        # Mettre a jour les compteurs
        self._last_optimization_time = current_time
        self._last_trades_count = len(self.trades)

        return result

    def _check_inactivity_and_correct(self) -> Dict:
        """
        Analyse intelligente des agents et ajuste les parametres selon la performance.
        - Augmente tolerance SI: agent inactif ET historique rentable avec tolerance plus large
        - Diminue tolerance SI: agent actif mais perdant (entrees trop larges)
        - Ne change rien SI: agent performant OU pas assez de donnees

        IMPORTANT: Analyse chaque agent INDIVIDUELLEMENT pour l'inactivite
        (un agent actif ne doit pas masquer l'inactivite des autres)
        """
        current_time = time.time()

        # Charger les trades de la session actuelle depuis session.json
        session_trades = self._load_session_trades()

        # Mettre a jour le compteur global (pour reference)
        if len(session_trades) > self._last_trades_count:
            self._last_trades_count = len(session_trades)
            self._save_inactivity_state()

        # Verifier temps depuis derniere optimisation GLOBALE
        inactivity_seconds = current_time - self._last_inactivity_reduction_time
        INACTIVITY_THRESHOLD = 900  # 15 minutes

        if inactivity_seconds < INACTIVITY_THRESHOLD:
            return {'executed_count': 0}

        print(f"[Strategist] Analyse des agents ({int(inactivity_seconds/60)} min depuis derniere action)...")

        agents_config = self._load_agents_config()
        executed_actions = []

        for agent_name, config in agents_config.items():
            if not config.get('enabled', False):
                continue

            # Analyser la performance de cet agent INDIVIDUELLEMENT
            agent_trades = [t for t in session_trades if t.get('agent_id') == agent_name or t.get('agent', '').endswith(agent_name)]

            # Calculer l'inactivite INDIVIDUELLE de cet agent
            if agent_trades:
                # Dernier trade de cet agent
                last_trade_time = None
                for t in reversed(agent_trades):
                    ts = t.get('timestamp') or t.get('close_time')
                    if ts:
                        try:
                            last_trade_time = datetime.fromisoformat(ts).timestamp()
                            break
                        except (ValueError, TypeError):
                            pass
                if last_trade_time:
                    agent_inactivity = current_time - last_trade_time
                else:
                    agent_inactivity = inactivity_seconds
            else:
                # Agent n'a JAMAIS trade dans cette session
                agent_inactivity = inactivity_seconds

            # Seuil d'inactivite par agent: 20 minutes
            AGENT_INACTIVITY_THRESHOLD = 1200  # 20 min

            if len(agent_trades) == 0 or agent_inactivity > AGENT_INACTIVITY_THRESHOLD:
                # Agent inactif - analyser pourquoi
                action = self._analyze_inactive_agent(agent_name, config, agent_inactivity)
                if action:
                    executed_actions.append(action)
            elif len(agent_trades) >= 5:
                # Agent actif avec assez de trades - analyser performance
                action = self._analyze_active_agent(agent_name, config, agent_trades)
                if action:
                    executed_actions.append(action)

        if executed_actions:
            self._save_agents_config(agents_config)
            self._last_inactivity_reduction_time = current_time
            self._save_inactivity_state()
            print(f"[Strategist] {len(executed_actions)} ajustements effectues")

        return {
            'executed_count': len(executed_actions),
            'actions': executed_actions
        }

    def _load_session_trades(self) -> list:
        """Charge les trades de la session actuelle"""
        try:
            session_file = DATABASE_DIR / "session.json"
            if session_file.exists():
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    return data.get('trades', [])
        except Exception as e:
            print(f"[Strategist] Erreur chargement session trades: {e}")
        return []

    def _has_permission(self, config: dict, permission: str) -> bool:
        """Verifie si le Strategist a la permission de modifier un parametre"""
        permissions = config.get('strategist_permissions', {})
        # Par defaut, toutes les permissions sont accordees
        return permissions.get(permission, True)

    def _analyze_inactive_agent(self, agent_name: str, config: dict, inactivity_seconds: float) -> dict:
        """
        Analyse un agent inactif et decide si ajuster les parametres.
        Retourne une action si un ajustement est justifie, None sinon.
        Respecte les permissions strategist_permissions de l'agent.
        """
        changes = []
        reasons = []
        current_tolerance = config.get('fibo_tolerance_pct', 1.0)
        current_momentum = config.get('min_momentum_pct', 0.05)

        # Verifier l'historique: est-ce que des tolerances plus larges ont ete rentables?
        # Pour l'instant, on verifie si la tolerance actuelle est restrictive

        # REGLE 1: Si tolerance < 2% et inactif > 30min, augmenter prudemment
        # PERMISSION: tolerance
        if self._has_permission(config, 'tolerance') and current_tolerance < 2.0 and inactivity_seconds > 1800:
            new_tolerance = min(2.0, round(current_tolerance + 0.5, 2))
            if new_tolerance > current_tolerance:
                config['fibo_tolerance_pct'] = new_tolerance
                changes.append(f"fibo_tolerance_pct: {current_tolerance} -> {new_tolerance}")
                reasons.append(f"Tolerance trop restrictive ({current_tolerance}%) avec 30+ min d'inactivite")

        # REGLE 2: Si tolerance deja >= 2% et inactif, le probleme est ailleurs
        # Ne pas augmenter indefiniment - verifier le fibo_level
        elif current_tolerance >= 2.0 and inactivity_seconds > 1800:
            # Peut-etre le niveau Fibo n'est pas atteint par le marche
            # Log mais ne pas changer automatiquement
            reasons.append(f"Tolerance deja a {current_tolerance}%, verifier conditions marche")
            # Pas de changement automatique - laisser l'utilisateur decider

        # REGLE 3: Si min_momentum_pct > 0.02 et inactif > 20min, reduire le seuil
        # Le seuil de momentum est peut-etre trop restrictif
        # PERMISSION: momentum
        if self._has_permission(config, 'momentum') and current_momentum > 0.02 and inactivity_seconds > 1200:
            new_momentum = max(0.02, round(current_momentum - 0.01, 3))
            if new_momentum < current_momentum:
                config['min_momentum_pct'] = new_momentum
                changes.append(f"min_momentum_pct: {current_momentum} -> {new_momentum}")
                reasons.append(f"Momentum min trop restrictif ({current_momentum}%) - reduction pour plus de trades")
                print(f"[Strategist] {agent_name}: REDUIRE min_momentum {current_momentum}% -> {new_momentum}%")

        if not changes:
            return None

        action = {
            'action': 'AJUSTEMENT_INACTIVITE',
            'agent': agent_name,
            'changes': changes,
            'reason': " | ".join(reasons),
            'analysis': {
                'inactivity_minutes': int(inactivity_seconds / 60),
                'previous_tolerance': current_tolerance,
                'decision': 'AUGMENTER_TOLERANCE' if changes else 'AUCUN_CHANGEMENT'
            },
            'timestamp': datetime.now().isoformat()
        }

        self.log_decision("ACTION_EXECUTED", action,
            f"Agent {agent_name}: {action['reason']}")

        return action

    def _analyze_active_agent(self, agent_name: str, config: dict, trades: list) -> dict:
        """
        Analyse un agent actif et decide si ajuster les parametres.
        - Si winrate < 40% : reduire tolerance ET augmenter min_momentum (trop de mauvais trades)
        - Si winrate > 70% : garder ou legere augmentation
        - Si winrate 40-70% : pas de changement
        Respecte les permissions strategist_permissions de l'agent.
        """
        if len(trades) < 5:
            return None

        # Calculer les stats des 10 derniers trades
        recent_trades = trades[-10:]
        wins = sum(1 for t in recent_trades if t.get('profit', t.get('profit_eur', 0)) > 0)
        losses = len(recent_trades) - wins
        winrate = (wins / len(recent_trades)) * 100 if recent_trades else 0
        total_profit = sum(t.get('profit', t.get('profit_eur', 0)) for t in recent_trades)

        changes = []
        reasons = []
        current_tolerance = config.get('fibo_tolerance_pct', 1.0)
        current_momentum = config.get('min_momentum_pct', 0.05)

        # REGLE 1: Winrate < 40% et perdant = tolerance trop large
        # PERMISSION: tolerance
        if winrate < 40 and total_profit < 0 and self._has_permission(config, 'tolerance'):
            new_tolerance = max(0.5, round(current_tolerance - 0.5, 2))
            if new_tolerance < current_tolerance:
                config['fibo_tolerance_pct'] = new_tolerance
                changes.append(f"fibo_tolerance_pct: {current_tolerance} -> {new_tolerance}")
                reasons.append(f"Winrate faible ({winrate:.0f}%) et pertes ({total_profit:.2f} EUR) - resserrer les entrees")
                print(f"[Strategist] {agent_name}: REDUIRE tolerance {current_tolerance}% -> {new_tolerance}% (WR={winrate:.0f}%)")

        # REGLE 1b: Augmenter min_momentum si winrate < 30% (trades sur mouvements trop faibles)
        # PERMISSION: momentum
        if winrate < 30 and current_momentum < 0.1 and self._has_permission(config, 'momentum'):
            new_momentum = min(0.1, round(current_momentum + 0.01, 3))
            if new_momentum > current_momentum:
                config['min_momentum_pct'] = new_momentum
                changes.append(f"min_momentum_pct: {current_momentum} -> {new_momentum}")
                reasons.append(f"Trades sur mouvements trop faibles - augmenter seuil momentum")
                print(f"[Strategist] {agent_name}: AUGMENTER min_momentum {current_momentum}% -> {new_momentum}% (WR={winrate:.0f}%)")

        # REGLE 2: Winrate > 70% et profitable = config optimale, ne pas toucher
        elif winrate > 70 and total_profit > 0:
            pass  # Config optimale, ne pas toucher

        # REGLE 3: Winrate 40-70% = zone neutre, pas de changement
        else:
            pass  # Zone neutre

        if not changes:
            return None

        action = {
            'action': 'AJUSTEMENT_PERFORMANCE',
            'agent': agent_name,
            'changes': changes,
            'reason': " | ".join(reasons),
            'analysis': {
                'recent_trades': len(recent_trades),
                'wins': wins,
                'losses': losses,
                'winrate': round(winrate, 1),
                'total_profit': round(total_profit, 2),
                'decision': 'REDUIRE_TOLERANCE' if changes else 'AUCUN_CHANGEMENT'
            },
            'timestamp': datetime.now().isoformat()
        }

        self.log_decision("ACTION_EXECUTED", action,
            f"Agent {agent_name}: {action['reason']}")

        return action

    def _execute_auto_corrections(self) -> Dict:
        """Execute les corrections automatiques basees sur l'analyse"""
        executed_actions = []
        skipped_actions = []

        # Analyser les stats
        global_stats = self._analyze_global()
        by_agent = self._analyze_by_agent()

        # Charger configs
        agents_config = self._load_agents_config()
        spread_config = self._load_spread_config()

        # === CORRECTION 1: Agent avec win rate < 30% -> AUGMENTER SEUILS ===
        # NOTE: Le Strategist n'a PAS le droit de desactiver les agents!
        # Il ne peut QUE ajuster les parametres de strategie.
        # RESPECTE les permissions strategist_permissions de chaque agent.
        for agent_name, stats in by_agent.items():
            if agent_name not in agents_config:
                continue

            agent_config = agents_config[agent_name]

            if stats.get('win_rate', 0) < 30 and stats.get('total_trades', 0) >= 5:
                action_hash = self._action_hash('RAISE_THRESHOLDS', agent_name, str(stats['total_trades']))

                if self._was_action_executed_recently(action_hash):
                    skipped_actions.append(f"RAISE_{agent_name} (cooldown)")
                    continue

                changes = []

                # Augmenter min_momentum_pct (PERMISSION: momentum)
                if 'min_momentum_pct' in agent_config and self._has_permission(agent_config, 'momentum'):
                    old_val = agent_config['min_momentum_pct']
                    new_val = max(0.05, old_val + 0.02) if old_val >= 0 else 0.05
                    if new_val != old_val:
                        agent_config['min_momentum_pct'] = round(new_val, 3)
                        changes.append(f"min_momentum_pct: {old_val} -> {new_val}")

                # Reduire fibo_tolerance_pct (plus strict) (PERMISSION: tolerance)
                if 'fibo_tolerance_pct' in agent_config and self._has_permission(agent_config, 'tolerance'):
                    old_val = agent_config['fibo_tolerance_pct']
                    new_val = max(0.1, old_val - 0.1)  # Min 0.1%
                    if new_val != old_val:
                        agent_config['fibo_tolerance_pct'] = round(new_val, 2)
                        changes.append(f"fibo_tolerance_pct: {old_val} -> {new_val}")

                # Changer vers un niveau Fibo plus fiable (pas de permission specifique)
                if 'fibo_level' in agent_config:
                    fibo_priority = ["0.618", "0.5", "0.382", "0.786", "0.236"]
                    current = agent_config['fibo_level']
                    if current in fibo_priority:
                        idx = fibo_priority.index(current)
                        if idx < len(fibo_priority) - 1:
                            new_level = fibo_priority[idx + 1]
                            agent_config['fibo_level'] = new_level
                            changes.append(f"fibo_level: {current} -> {new_level}")

                # Augmenter cooldown (PERMISSION: cooldown)
                if 'cooldown_seconds' in agent_config and self._has_permission(agent_config, 'cooldown'):
                    old_val = agent_config['cooldown_seconds']
                    new_val = min(300, old_val + 30)
                    if new_val != old_val:
                        agent_config['cooldown_seconds'] = new_val
                        changes.append(f"cooldown: {old_val}s -> {new_val}s")

                if changes:
                    action = {
                        'action': 'AUGMENTER_SEUILS',
                        'agent': agent_name,
                        'changes': changes,
                        'reason': f"Win rate {stats['win_rate']}% < 30%",
                        'timestamp': datetime.now().isoformat()
                    }
                    executed_actions.append(action)
                    self._mark_action_executed(action_hash)

                    self.log_decision("ACTION_EXECUTED", action,
                        f"Agent {agent_name}: {', '.join(changes)}")

        # === CORRECTION 3: Profit factor < 1.0 -> AJUSTER TP/SL ===
        # Un profit_factor < 1.0 signifie que les pertes > gains (meme avec bon win rate)
        # PERMISSION: tp_sl (verifie sur au moins un agent actif)
        has_tpsl_permission = any(
            self._has_permission(agents_config.get(agent_name, {}), 'tp_sl')
            for agent_name in agents_config
            if agents_config.get(agent_name, {}).get('enabled', False)
        )

        if has_tpsl_permission and global_stats.get('profit_factor', 0) < 1.0 and global_stats.get('total_trades', 0) >= 10:
            action_hash = self._action_hash('ADJUST_TPSL', 'global', str(global_stats.get('total_trades', 0) // 10))

            if not self._was_action_executed_recently(action_hash):
                old_tp = spread_config.get('tp_pct', 0.5)
                old_sl = spread_config.get('sl_pct', 1.0)

                new_tp = round(min(2.0, old_tp + 0.2), 2)
                new_sl = round(max(0.3, old_sl - 0.1), 2)

                if new_tp != old_tp or new_sl != old_sl:
                    spread_config['tp_pct'] = new_tp
                    spread_config['sl_pct'] = new_sl

                    action = {
                        'action': 'AJUSTER_TPSL',
                        'old_tp': old_tp,
                        'new_tp': new_tp,
                        'old_sl': old_sl,
                        'new_sl': new_sl,
                        'reason': f"Profit factor {global_stats['profit_factor']} < 1.0 (pertes > gains)",
                        'timestamp': datetime.now().isoformat()
                    }
                    executed_actions.append(action)
                    self._mark_action_executed(action_hash)

                    self.log_decision("ACTION_EXECUTED", action,
                        f"TP/SL: TP {old_tp}% -> {new_tp}%, SL {old_sl}% -> {new_sl}%")

        # === CONSENSUS MULTI-AI ===
        consensus = self.calculate_multi_ai_consensus()
        if consensus.get('bias') != 'neutral' and consensus.get('confidence', 0) >= 60:
            # Si consensus fort, on peut ajuster l'agressivite globale
            # (Par exemple, reduire les seuils pour suivre le consensus)
            executed_actions.append({
                'action': 'CONSENSUS_BIAS_ADOPTED',
                'bias': consensus['bias'],
                'confidence': consensus['confidence'],
                'reason': f"Consensus fort {consensus['bias']} ({consensus['confidence']}%)",
                'timestamp': datetime.now().isoformat()
            })
            # Ici on pourrait trigger une modification des seuils si necessaire
            # mais on va d'abord logger l'adoption du consensus.

        # === AJUSTEMENT DES POIDS D'ANALYSE ===
        # Si le win rate est bas, augmenter le poids du price momentum (plus fiable que les indicateurs externes)
        if global_stats.get('win_rate', 0) < 40 and global_stats.get('total_trades', 0) >= 15:
            action_hash = self._action_hash('ADJUST_ANALYSIS_WEIGHTS', 'momentum', str(global_stats.get('total_trades', 0) // 15))

            if not self._was_action_executed_recently(action_hash):
                try:
                    if ANALYSIS_WEIGHTS_FILE.exists():
                        with open(ANALYSIS_WEIGHTS_FILE, 'r') as f:
                            weights_config = json.load(f)

                        weights = weights_config.get('weights', {})
                        old_weights = weights.copy()

                        # Augmenter price_momentum (jusqu'a max 30%)
                        # Reduire macro et whales (sources externes moins fiables)
                        if weights.get('price_momentum', 20) < 30:
                            weights['price_momentum'] = min(30, weights.get('price_momentum', 20) + 5)
                            weights['macro'] = max(15, weights.get('macro', 25) - 3)
                            weights['whales'] = max(15, weights.get('whales', 20) - 2)

                            # Ajouter a l'historique
                            history_entry = {
                                'timestamp': datetime.now().isoformat(),
                                'reason': f"Win rate {global_stats['win_rate']}% < 40%",
                                'old_weights': old_weights,
                                'new_weights': weights.copy()
                            }
                            if 'history' not in weights_config:
                                weights_config['history'] = []
                            weights_config['history'].append(history_entry)

                            # Garder 20 dernieres entrees
                            weights_config['history'] = weights_config['history'][-20:]
                            weights_config['updated_at'] = datetime.now().isoformat()
                            weights_config['weights'] = weights

                            # Sauvegarder
                            with open(ANALYSIS_WEIGHTS_FILE, 'w') as f:
                                json.dump(weights_config, f, indent=2)

                            # Recharger dans l'aggregator
                            from data.aggregator import get_aggregator
                            aggregator = get_aggregator()
                            aggregator.reload_analysis_weights()

                            action = {
                                'action': 'AJUSTER_POIDS_ANALYSE',
                                'old_weights': old_weights,
                                'new_weights': weights,
                                'reason': f"Win rate {global_stats['win_rate']}% faible - Priorite au price momentum",
                                'timestamp': datetime.now().isoformat()
                            }
                            executed_actions.append(action)
                            self._mark_action_executed(action_hash)

                            self.log_decision("ACTION_EXECUTED", action,
                                f"Poids analyse ajustes: momentum {old_weights.get('price_momentum')}% -> {weights['price_momentum']}%")
                except Exception as e:
                    print(f"[Strategist] Erreur ajustement poids analyse: {e}")

        # Sauvegarder les configs modifiees
        if executed_actions:
            self._save_agents_config(agents_config)
            self._save_spread_config(spread_config)
            print(f"[Strategist] {len(executed_actions)} corrections executees")

        return {
            'status': 'optimized',
            'executed_count': len(executed_actions),
            'skipped_count': len(skipped_actions),
            'actions': executed_actions,
            'skipped': skipped_actions,
            'global_stats': global_stats
        }

    def get_actions_context_for_ia(self) -> str:
        """
        Retourne un contexte textuel des actions deja executees.
        A inclure dans les prompts IA pour eviter les suggestions repetitives.
        """
        recent_actions = self.get_executed_actions(limit=20)

        if not recent_actions:
            return "Aucune correction recente."

        lines = ["CORRECTIONS DEJA APPLIQUEES:"]
        for action in recent_actions[:10]:
            timestamp = action.get('timestamp', '')[:16]
            details = action.get('details', {})
            action_type = details.get('action', 'UNKNOWN')
            agent = details.get('agent', '')
            reason = action.get('reason', '')[:60]

            if agent:
                lines.append(f"- [{timestamp}] {action_type} sur {agent}: {reason}")
            else:
                lines.append(f"- [{timestamp}] {action_type}: {reason}")

        return "\n".join(lines)

    def _load_trades(self):
        """Charge l'historique des trades (session actuelle + archives)"""
        try:
            trades = []

            # 1. Charger la SESSION ACTUELLE (prioritaire)
            current_session_file = DATABASE_DIR / "session.json"
            if current_session_file.exists():
                try:
                    with open(current_session_file, 'r') as f:
                        session_data = json.load(f)
                        session_trades = session_data.get("trades", [])
                        # Normaliser le format
                        for t in session_trades:
                            if "agent" in t and "agent_id" not in t:
                                agent_name = t["agent"]
                                if agent_name.startswith("G12_"):
                                    agent_name = agent_name[4:]
                                t["agent_id"] = agent_name
                            if "profit" in t and "profit_eur" not in t:
                                t["profit_eur"] = t["profit"]
                        trades.extend(session_trades)
                except Exception as e:
                    print(f"[Strategist] Erreur chargement session actuelle: {e}")

            # 2. Charger du fichier principal trades.json (souvent vide)
            if TRADES_FILE.exists():
                with open(TRADES_FILE, 'r') as f:
                    data = json.load(f)
                    old_trades = data.get("trades", []) if isinstance(data, dict) else data
                    trades.extend(old_trades)

            # 3. Charger des sessions archivees
            sessions_dir = DATABASE_DIR / "sessions"
            if sessions_dir.exists():
                for session_file in sessions_dir.iterdir():
                    if session_file.is_file() and session_file.name.startswith("session_") and session_file.suffix == ".json":
                        try:
                            with open(session_file, 'r') as f:
                                session_data = json.load(f)
                                session_trades = session_data.get("trades", [])
                                for t in session_trades:
                                    if "agent" in t and "agent_id" not in t:
                                        agent_name = t["agent"]
                                        if agent_name.startswith("G12_"):
                                            agent_name = agent_name[4:]
                                        t["agent_id"] = agent_name
                                    if "profit" in t and "profit_eur" not in t:
                                        t["profit_eur"] = t["profit"]
                                trades.extend(session_trades)
                        except Exception:
                            continue

            self.trades = trades
            print(f"[Strategist] {len(self.trades)} trades charges")
        except Exception as e:
            print(f"[Strategist] Erreur chargement trades: {e}")
            self.trades = []

    def analyze(self) -> Dict:
        """Analyse complete des performances (sans execution)"""
        self._load_trades()

        if len(self.trades) < 5:
            return {
                'status': 'insufficient_data',
                'message': f'Besoin de plus de trades ({len(self.trades)}/5 minimum)',
                'suggestions': []
            }

        analysis = {
            'global': self._analyze_global(),
            'by_agent': self._analyze_by_agent(),
            'by_session': self._analyze_by_session(),
            'patterns': self._find_patterns(),
            'consensus': self.calculate_multi_ai_consensus(),
            'suggestions': self._generate_suggestions(),
            'recent_actions': self.get_executed_actions(limit=10)
        }

        return analysis

    def _analyze_global(self) -> Dict:
        """Statistiques globales"""
        if not self.trades:
            return {}

        profits = [t.get('profit', t.get('profit_eur', 0)) for t in self.trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        return {
            'total_trades': len(self.trades),
            'total_profit': round(sum(profits), 2),
            'avg_profit': round(sum(profits) / len(profits), 2) if profits else 0,
            'win_rate': round(len(wins) / len(profits) * 100, 1) if profits else 0,
            'avg_win': round(sum(wins) / len(wins), 2) if wins else 0,
            'avg_loss': round(sum(losses) / len(losses), 2) if losses else 0,
            'profit_factor': round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 0,
            'consecutive_losses': self._max_consecutive(profits, False)
        }

    def _max_consecutive(self, profits: List[float], wins: bool) -> int:
        """Calcule les trades consecutifs gagnants/perdants"""
        max_count = 0
        current = 0
        for p in profits:
            if (wins and p > 0) or (not wins and p < 0):
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return max_count

    def _analyze_by_agent(self) -> Dict:
        """Analyse par agent - utilise stats_*.json (source de verite MT5)

        IMPORTANT: Lit depuis stats_*.json pour coherence avec les autres affichages.
        Les fichiers stats_*.json sont synchronises avec l'historique MT5 reel.
        """
        results = {}

        for agent_id in ['fibo1', 'fibo2', 'fibo3']:
            stats_file = DATABASE_DIR / f"stats_{agent_id}.json"
            try:
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)

                    session_trades = stats.get('session_trades', 0)
                    session_wins = stats.get('session_wins', 0)
                    session_pnl = stats.get('session_pnl', 0)
                    session_losses = session_trades - session_wins

                    # Calculer profit factor approximatif
                    # PF = gains / pertes = (wins * avg_win) / (losses * avg_loss)
                    # Approximation: si winrate et pnl connus
                    if session_trades > 0 and session_wins > 0 and session_losses > 0:
                        avg_win = session_pnl / session_wins if session_pnl > 0 else 0
                        avg_loss = abs(session_pnl) / session_losses if session_pnl < 0 else 1
                        profit_factor = round(avg_win * session_wins / (avg_loss * session_losses), 2) if avg_loss > 0 else 0
                    else:
                        profit_factor = 0

                    results[agent_id] = {
                        'total_trades': session_trades,
                        'total_profit': round(session_pnl, 2),
                        'win_rate': round(session_wins / session_trades * 100, 1) if session_trades > 0 else 0,
                        'profit_factor': profit_factor
                    }
                else:
                    results[agent_id] = {
                        'total_trades': 0,
                        'total_profit': 0,
                        'win_rate': 0,
                        'profit_factor': 0
                    }
            except Exception as e:
                print(f"[Strategist] Erreur lecture stats {agent_id}: {e}")
                results[agent_id] = {
                    'total_trades': 0,
                    'total_profit': 0,
                    'win_rate': 0,
                    'profit_factor': 0
                }

        return results

    def _analyze_open_positions(self) -> Dict:
        """
        Analyse les positions ouvertes et detecte les incoherences G12 vs MT5.
        Retourne un dict avec:
        - g12_positions: positions trackees par G12 (positions_fibo*.json)
        - mt5_positions: positions reelles sur MT5 (via aggregator)
        - inconsistencies: liste des problemes detectes
        - alerts: alertes critiques necessitant attention
        """
        result = {
            'g12_positions': {},
            'mt5_positions': [],
            'inconsistencies': [],
            'alerts': [],
            'total_floating_pnl': 0,
            'positions_count': {'g12': 0, 'mt5': 0}
        }

        # 1. Charger les positions G12 depuis positions_fibo*.json
        g12_all_positions = {}
        for agent_id in ['fibo1', 'fibo2', 'fibo3']:
            try:
                pos_file = DATABASE_DIR / f"positions_{agent_id}.json"
                if pos_file.exists():
                    with open(pos_file, 'r') as f:
                        data = json.load(f)
                        positions = data.get('positions', [])
                        g12_all_positions[agent_id] = positions
                        result['positions_count']['g12'] += len(positions)
                else:
                    g12_all_positions[agent_id] = []
            except Exception as e:
                print(f"[Strategist] Erreur lecture positions G12 {agent_id}: {e}")
                g12_all_positions[agent_id] = []

        result['g12_positions'] = g12_all_positions

        # 2. Charger les positions MT5 reelles via aggregator
        try:
            aggregator = get_aggregator()
            account_data = aggregator.get_account_data()
            if account_data and account_data.get('positions'):
                mt5_positions = account_data['positions']
                result['mt5_positions'] = mt5_positions
                result['positions_count']['mt5'] = len(mt5_positions)
                result['total_floating_pnl'] = round(account_data.get('floating_pnl', 0), 2)
        except Exception as e:
            print(f"[Strategist] Erreur recuperation positions MT5: {e}")
            result['alerts'].append({
                'level': 'critical',
                'message': f"Impossible de recuperer les positions MT5: {e}",
                'timestamp': datetime.now().isoformat()
            })

        # 3. Comparer G12 vs MT5 pour detecter les incoherences
        g12_tickets = set()
        for agent_id, positions in g12_all_positions.items():
            for pos in positions:
                ticket = pos.get('ticket')
                if ticket:
                    g12_tickets.add(ticket)

        mt5_tickets = set()
        mt5_by_ticket = {}
        for pos in result['mt5_positions']:
            ticket = pos.get('ticket')
            if ticket:
                mt5_tickets.add(ticket)
                mt5_by_ticket[ticket] = pos

        # 3a. Positions dans G12 mais pas dans MT5 (fermees sans sync?)
        orphan_g12 = g12_tickets - mt5_tickets
        if orphan_g12:
            for ticket in orphan_g12:
                # Trouver l'agent
                for agent_id, positions in g12_all_positions.items():
                    for pos in positions:
                        if pos.get('ticket') == ticket:
                            result['inconsistencies'].append({
                                'type': 'G12_ORPHAN',
                                'ticket': ticket,
                                'agent': agent_id,
                                'message': f"Position #{ticket} dans G12 ({agent_id}) mais fermee sur MT5",
                                'severity': 'warning'
                            })
                            break

        # 3b. Positions dans MT5 mais pas dans G12 (non trackees)
        orphan_mt5 = mt5_tickets - g12_tickets
        if orphan_mt5:
            for ticket in orphan_mt5:
                mt5_pos = mt5_by_ticket.get(ticket, {})
                agent = mt5_pos.get('_agent_id', 'unknown')
                result['inconsistencies'].append({
                    'type': 'MT5_UNTRACKED',
                    'ticket': ticket,
                    'agent': agent,
                    'message': f"Position MT5 #{ticket} non trackee par G12",
                    'severity': 'critical',
                    'position': {
                        'direction': 'BUY' if mt5_pos.get('type', 0) == 0 else 'SELL',
                        'profit': mt5_pos.get('profit', 0),
                        'volume': mt5_pos.get('volume', 0)
                    }
                })
                result['alerts'].append({
                    'level': 'critical',
                    'message': f"Position MT5 #{ticket} non trackee par G12!",
                    'ticket': ticket,
                    'timestamp': datetime.now().isoformat()
                })

        # 3c. Verifier le nombre total de positions
        if result['positions_count']['g12'] != result['positions_count']['mt5']:
            result['inconsistencies'].append({
                'type': 'COUNT_MISMATCH',
                'message': f"Nombre positions different: G12={result['positions_count']['g12']} vs MT5={result['positions_count']['mt5']}",
                'severity': 'warning'
            })

        # 4. Analyser le P&L flottant global
        if result['total_floating_pnl'] < -50:  # Seuil d'alerte configurable
            result['alerts'].append({
                'level': 'warning',
                'message': f"P&L flottant eleve: {result['total_floating_pnl']} EUR",
                'timestamp': datetime.now().isoformat()
            })

        print(f"[Strategist] Analyse positions: G12={result['positions_count']['g12']}, MT5={result['positions_count']['mt5']}, Incoherences={len(result['inconsistencies'])}")

        return result

    def _check_positions_and_alert(self) -> Dict:
        """
        Verifie la coherence des positions et genere des alertes si necessaire.
        Appelee par auto_optimize().
        """
        analysis = self._analyze_open_positions()

        executed_actions = []

        # Generer des actions pour les incoherences critiques
        for inconsistency in analysis.get('inconsistencies', []):
            if inconsistency.get('severity') == 'critical':
                action = {
                    'action': 'POSITION_INCONSISTENCY',
                    'type': inconsistency['type'],
                    'ticket': inconsistency.get('ticket'),
                    'agent': inconsistency.get('agent', 'unknown'),
                    'message': inconsistency['message'],
                    'timestamp': datetime.now().isoformat()
                }
                executed_actions.append(action)

                # Log l'alerte
                self.log_decision("POSITION_ALERT", action,
                    f"Incoherence detectee: {inconsistency['message']}")

                # Envoyer notification Telegram si disponible
                try:
                    telegram = get_telegram()
                    if telegram:
                        telegram.send_alert(f"⚠️ ALERTE POSITIONS\n{inconsistency['message']}")
                except Exception:
                    pass

        return {
            'executed_count': len(executed_actions),
            'actions': executed_actions,
            'analysis': analysis
        }

    def _analyze_by_session(self) -> Dict:
        """Analyse par session de marche"""
        sessions = {
            'asia': {'start': 0, 'end': 8, 'profits': []},
            'europe': {'start': 8, 'end': 14, 'profits': []},
            'usa': {'start': 14, 'end': 22, 'profits': []},
            'night': {'start': 22, 'end': 24, 'profits': []}
        }

        for trade in self.trades:
            try:
                time_str = trade.get('close_time') or trade.get('open_time') or trade.get('timestamp')
                if time_str:
                    hour = datetime.fromisoformat(time_str).hour
                    for name, session in sessions.items():
                        if session['start'] <= hour < session['end']:
                            session['profits'].append(trade.get('profit', trade.get('profit_eur', 0)))
                            break
            except (ValueError, TypeError):
                pass

        results = {}
        for name, session in sessions.items():
            profits = session['profits']
            if profits:
                wins = [p for p in profits if p > 0]
                results[name] = {
                    'trades': len(profits),
                    'total_profit': round(sum(profits), 2),
                    'win_rate': round(len(wins) / len(profits) * 100, 1)
                }
            else:
                results[name] = {'trades': 0, 'total_profit': 0, 'win_rate': 0}

        return results

    def calculate_multi_ai_consensus(self) -> Dict:
        """Calcule un consensus pondere base sur toutes les sources de donnees IA"""
        aggregator = get_aggregator()
        context = aggregator.last_context or aggregator.get_full_context()
        
        scores = {"bullish": 0, "bearish": 0}
        details = {}

        # 1. MACRO ENGINE (Poids: 30%)
        macro = context.get('macro', {})
        macro_bias = macro.get('bias', 'neutral')
        if macro_bias == 'bullish': 
            scores['bullish'] += 30
            details['macro'] = "Bullish (DXY/SPX)"
        elif macro_bias == 'bearish': 
            scores['bearish'] += 30
            details['macro'] = "Bearish (DXY/SPX)"

        # 2. WHALE TRACKER (Poids: 25%)
        whales = context.get('whales', {}) # get_whale_bias returns {'bias': ..., 'reason': ...}
        whale_bias = whales.get('bias', 'neutral')
        if whale_bias == 'bullish':
            scores['bullish'] += 25
            details['whales'] = "Institutional Inflow"
        elif whale_bias == 'bearish':
            scores['bearish'] += 25
            details['whales'] = "Institutional Outflow"

        # 3. SENTIMENT (Fear & Greed, News) (Poids: 15%)
        sentiment = context.get('sentiment', {})
        global_bias = sentiment.get('global_bias', 'neutral')
        if global_bias == 'bullish':
            scores['bullish'] += 15
            details['sentiment'] = "Market Sentiment Bullish"
        elif global_bias == 'bearish':
            scores['bearish'] += 15
            details['sentiment'] = "Market Sentiment Bearish"

        # 4. FUTURES (Funding, LS Ratio) (Poids: 20%)
        # Contrarian logic for extremes, inclusive for normal
        futures = context.get('futures', {})
        funding = futures.get('funding_rate', 0)
        ls_ratio = futures.get('long_short_ratio', 1.0)
        
        if funding < 0 and ls_ratio < 0.9: # Squeeze potential
            scores['bullish'] += 20
            details['futures'] = "Short Squeeze Potential"
        elif funding > 0.05 and ls_ratio > 1.5: # Long Squeeze potential
            scores['bearish'] += 20
            details['futures'] = "Long Squeeze Potential"

        # 5. BTC DOMINANCE (Poids: 10%) - New source
        # High dominance + Bullish BTC = Strongest signal
        # (Note: btc_dominance should be in sentiment or direct in context)
        btc_dom = context.get('sentiment', {}).get('btc_dominance', 50)
        if btc_dom > 55 and macro_bias == 'bullish':
            scores['bullish'] += 10
            details['btc_dom'] = "BTC Safe Haven / Strong Dominance"

        # Calcul final
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

        return {
            "bias": bias,
            "confidence": min(100, confidence),
            "scores": scores,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

    def _find_patterns(self) -> List[Dict]:
        """Detecte des patterns dans les trades"""
        patterns = []
        by_agent = self._analyze_by_agent()

        for agent, data in by_agent.items():
            if data['total_trades'] >= 5 and data['win_rate'] < 35:
                patterns.append({
                    'type': 'underperforming_agent',
                    'agent': agent,
                    'win_rate': data['win_rate'],
                    'trades': data['total_trades']
                })

        return patterns

    def _generate_suggestions(self) -> List[Dict]:
        """Genere des suggestions (informatives, pas executees ici)"""
        suggestions = []
        global_stats = self._analyze_global()

        # Ne pas re-suggerer ce qui a deja ete fait
        recent_actions = self.get_executed_actions(limit=10)
        recent_action_types = set()
        for action in recent_actions:
            details = action.get('details', {})
            recent_action_types.add(f"{details.get('action', '')}_{details.get('agent', '')}")

        if global_stats.get('win_rate', 0) < 45:
            suggestions.append({
                'priority': 'high',
                'category': 'win_rate',
                'suggestion': 'Les seuils seront automatiquement augmentes',
                'reason': f'Win rate: {global_stats["win_rate"]}%'
            })

        if global_stats.get('profit_factor', 0) < 1:
            suggestions.append({
                'priority': 'critical',
                'category': 'profit_factor',
                'suggestion': 'TP/SL seront automatiquement ajustes',
                'reason': f'Profit factor: {global_stats["profit_factor"]}'
            })

        return suggestions

    def log_decision(self, decision_type: str, details: Dict, reason: str):
        """Enregistre une decision/action du Strategist"""
        try:
            logs = self._load_logs()
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': decision_type,
                'details': details,
                'reason': reason
            }
            logs.insert(0, log_entry)
            logs = logs[:200]
            with open(LOGS_FILE, 'w') as f:
                json.dump({'logs': logs}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Strategist] Erreur sauvegarde log: {e}")

    def _load_logs(self) -> List[Dict]:
        """Charge les logs existants"""
        try:
            if LOGS_FILE.exists():
                with open(LOGS_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('logs', [])
        except Exception:
            pass
        return []

    def get_logs(self, limit: int = 50) -> List[Dict]:
        """Retourne les derniers logs"""
        return self._load_logs()[:limit]

    def get_executed_actions(self, limit: int = 50) -> List[Dict]:
        """Retourne les actions executees (filtrees)

        Args:
            limit: Nombre max d'actions a retourner. Si None, retourne toutes les actions.
        """
        logs = self._load_logs()
        actions = [log for log in logs if log.get('type') == 'ACTION_EXECUTED']

        if limit is None:
            return actions
        return actions[:limit]

    def get_quick_insights(self) -> Dict:
        """Retourne des insights rapides pour le dashboard"""
        try:
            global_stats = self._analyze_global()
            by_agent = self._analyze_by_agent()

            # Calculer tendance
            trend = "NEUTRE"
            if global_stats.get('win_rate', 0) >= 50:
                trend = "HAUSSIER"
            elif global_stats.get('win_rate', 0) < 30:
                trend = "BAISSIER"

            # Meilleur et pire agent
            best_agent = None
            worst_agent = None
            best_wr = 0
            worst_wr = 100

            for agent, stats in by_agent.items():
                wr = stats.get('win_rate', 0)
                if stats.get('total_trades', 0) >= 3:
                    if wr > best_wr:
                        best_wr = wr
                        best_agent = agent
                    if wr < worst_wr:
                        worst_wr = wr
                        worst_agent = agent

            return {
                'status': 'ok',
                'recent_win_rate': global_stats.get('win_rate', 0),
                'win_rate': global_stats.get('win_rate', 0),
                'trend': 'up' if trend == "HAUSSIER" else 'down',
                'total_trades': global_stats.get('total_trades', 0),
                'total_profit': global_stats.get('total_profit', 0),
                'best_agent': best_agent,
                'worst_agent': worst_agent,
                'last_actions': len(self.get_executed_actions(10))
            }
        except Exception as e:
            print(f"[Strategist] Erreur get_quick_insights: {e}")
            return {
                'status': 'error',
                'recent_win_rate': 0,
                'win_rate': 0,
                'trend': 'down',
                'total_trades': 0,
                'total_profit': 0,
                'best_agent': None,
                'worst_agent': None,
                'last_actions': 0
            }

    def _load_agents_config(self) -> Dict:
        """Charge la config des agents"""
        try:
            if AGENTS_CONFIG_FILE.exists():
                with open(AGENTS_CONFIG_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[Strategist] Erreur chargement agents config: {e}")
        return {}

    def _save_agents_config(self, config: Dict):
        """Sauvegarde la config des agents"""
        try:
            with open(AGENTS_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"[Strategist] Erreur sauvegarde agents config: {e}")

    def _load_spread_config(self) -> Dict:
        """Charge la config spread/TP/SL"""
        try:
            if SPREAD_CONFIG_FILE.exists():
                with open(SPREAD_CONFIG_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[Strategist] Erreur chargement spread config: {e}")
        return {}

    def _save_spread_config(self, config: Dict):
        """Sauvegarde la config spread/TP/SL"""
        try:
            with open(SPREAD_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"[Strategist] Erreur sauvegarde spread config: {e}")

    def self_learning_optimization(self):
        """
        SELF-LEARNING: Analyse profonde des patterns pour generer des regles.
        Appelee moins frequemment que auto_optimize.
        """
        LEARNED_RULES_FILE = DATABASE_DIR / "learned_rules.json"
        
        # 1. Analyser les meilleures heures
        hourly_stats = self._analyze_by_session()
        best_session = None
        best_wr = 0
        
        for name, stats in hourly_stats.items():
            if stats.get('win_rate', 0) > best_wr and stats.get('trades', 0) >= 5:
                best_wr = stats['win_rate']
                best_session = name
                
        # 2. Generer regles
        rules = []
        if best_session:
            rules.append({
                "type": "best_session",
                "condition": f"session == '{best_session}'",
                "action": "increase_risk",
                "parameter": "position_size_pct", 
                "value": "0.02",
                "reason": f"Session {best_session} a un WR de {best_wr}%"
            })
            
        # 3. Sauvegarder
        try:
            with open(LEARNED_RULES_FILE, 'w') as f:
                json.dump({"rules": rules, "last_update": datetime.now().isoformat()}, f, indent=2)
            
            print(f"[Strategist] Self-learning termine. {len(rules)} regles d'optimisation generees.")
            
            # Notifier via Telegram si possible
            if rules:
                try:
                    get_telegram().send_message(f"🧠 *Strategist Self-Learning*\nNouvelles regles apprises: {len(rules)}")
                except Exception:
                    pass
                
        except Exception as e:
            print(f"[Strategist] Erreur sauvegarde learned rules: {e}")

    # Garder pour compatibilite API
    def execute_suggestions(self, auto_mode: bool = False) -> Dict:
        """Alias pour auto_optimize (compatibilite)"""
        return self.auto_optimize()


# Singleton
_strategist = None

def get_strategist() -> Strategist:
    """Retourne l'instance Strategist singleton"""
    global _strategist
    if _strategist is None:
        _strategist = Strategist()
    return _strategist
