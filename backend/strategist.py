# -*- coding: utf-8 -*-
"""
G12 - Strategist
Analyse les trades et execute des corrections AUTOMATIQUEMENT
"""

import json
import time
import hashlib
from datetime import datetime, timedelta
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
from data.aggregator import get_aggregator
from session_logger import get_session_logger
from utils.telegram_service import get_telegram

DATABASE_DIR = Path(__file__).parent / "database"
TRADES_FILE = DATABASE_DIR / "trades.json"
SUGGESTIONS_FILE = DATABASE_DIR / "strategy_suggestions.json"
LOGS_FILE = DATABASE_DIR / "strategist_logs.json"
AGENTS_CONFIG_FILE = DATABASE_DIR / "agents_runtime_config.json"
SPREAD_CONFIG_FILE = DATABASE_DIR / "spread_runtime_config.json"
ACTIONS_HISTORY_FILE = DATABASE_DIR / "strategist_actions_history.json"
LAST_RUN_FILE = DATABASE_DIR / "strategist_last_run.json" # Added for self-learning
INACTIVITY_STATE_FILE = DATABASE_DIR / "strategist_inactivity_state.json"  # Persiste le timer d'inactivite

# Intervalles d'optimisation
MIN_TRADES_BETWEEN_OPTIM = 5  # Minimum trades entre 2 optimisations
MIN_SECONDS_BETWEEN_OPTIM = 60  # Minimum secondes entre 2 optimisations
ACTION_COOLDOWN_HOURS = 1  # Cooldown avant de re-executer la meme action
SELF_LEARNING_INTERVAL_HOURS = 6 # Intervalle pour le self-learning

class Strategist:
    """Analyse les performances et execute des corrections AUTOMATIQUEMENT"""

    def __init__(self):
        self.trades = []
        self.suggestions = []
        self._last_optimization_time = 0
        self._last_trades_count = 0
        self._last_inactivity_reduction_time = self._load_inactivity_state()  # Persiste entre redemarrages
        self._executed_action_hashes: Dict[str, float] = {}  # hash -> timestamp
        self._last_self_learning_run = 0 # Added for self-learning
        self._load_trades()
        self._load_action_history()
        self._load_last_run_info() # Added for self-learning

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
        except Exception as e:
            print(f"[Strategist] Erreur chargement last run info: {e}")
            self._last_self_learning_run = 0

    def _save_last_run_info(self):
        """Sauvegarde les informations de la derniere execution (pour le self-learning)"""
        try:
            with open(LAST_RUN_FILE, 'w') as f:
                json.dump({
                    'last_self_learning_run': self._last_self_learning_run,
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

        # 3. Self-Learning (if due)
        if self._should_run_self_learning():
            print("[Strategist] Lancement du Self-Learning...")
            try:
                self.self_learning_optimization()
            except Exception as e:
                print(f"[Strategist] Erreur self-learning: {e}")
            self._mark_self_learning_run()


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
        """
        current_time = time.time()

        # Charger les trades de la session actuelle depuis session.json
        session_trades = self._load_session_trades()

        # Si nouveaux trades, reset timer
        if len(session_trades) > self._last_trades_count:
            self._last_inactivity_reduction_time = current_time
            self._save_inactivity_state()
            self._last_trades_count = len(session_trades)
            return {'executed_count': 0}

        # Verifier temps depuis derniere optimisation
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

            # Analyser la performance de cet agent
            agent_trades = [t for t in session_trades if t.get('agent_id') == agent_name or t.get('agent', '').endswith(agent_name)]

            if len(agent_trades) == 0:
                # Agent inactif - analyser pourquoi
                action = self._analyze_inactive_agent(agent_name, config, inactivity_seconds)
                if action:
                    executed_actions.append(action)
            else:
                # Agent actif - analyser performance
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

    def _analyze_inactive_agent(self, agent_name: str, config: dict, inactivity_seconds: float) -> dict:
        """
        Analyse un agent inactif et decide si ajuster les parametres.
        Retourne une action si un ajustement est justifie, None sinon.
        """
        changes = []
        reasons = []
        current_tolerance = config.get('fibo_tolerance_pct', 1.0)

        # Verifier l'historique: est-ce que des tolerances plus larges ont ete rentables?
        # Pour l'instant, on verifie si la tolerance actuelle est restrictive

        # REGLE 1: Si tolerance < 2% et inactif > 30min, augmenter prudemment
        if current_tolerance < 2.0 and inactivity_seconds > 1800:
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
        - Si winrate < 40% : reduire tolerance (entrees trop larges)
        - Si winrate > 70% : garder ou legere augmentation
        - Si winrate 40-70% : pas de changement
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

        # REGLE 1: Winrate < 40% et perdant = tolerance trop large
        if winrate < 40 and total_profit < 0:
            new_tolerance = max(0.5, round(current_tolerance - 0.5, 2))
            if new_tolerance < current_tolerance:
                config['fibo_tolerance_pct'] = new_tolerance
                changes.append(f"fibo_tolerance_pct: {current_tolerance} -> {new_tolerance}")
                reasons.append(f"Winrate faible ({winrate:.0f}%) et pertes ({total_profit:.2f} EUR) - resserrer les entrees")
                print(f"[Strategist] {agent_name}: REDUIRE tolerance {current_tolerance}% -> {new_tolerance}% (WR={winrate:.0f}%)")

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
        for agent_name, stats in by_agent.items():
            if agent_name not in agents_config:
                continue

            if stats.get('win_rate', 0) < 30 and stats.get('total_trades', 0) >= 5:
                action_hash = self._action_hash('RAISE_THRESHOLDS', agent_name, str(stats['total_trades']))

                if self._was_action_executed_recently(action_hash):
                    skipped_actions.append(f"RAISE_{agent_name} (cooldown)")
                    continue

                changes = []

                # Augmenter min_fibo1_pct
                if 'min_fibo1_pct' in agents_config[agent_name]:
                    old_val = agents_config[agent_name]['min_fibo1_pct']
                    new_val = max(0.05, old_val + 0.02) if old_val >= 0 else 0.05
                    if new_val != old_val:
                        agents_config[agent_name]['min_fibo1_pct'] = round(new_val, 3)
                        changes.append(f"min_fibo1_pct: {old_val} -> {new_val}")

                # Reduire fibo_tolerance_pct (plus strict)
                if 'fibo_tolerance_pct' in agents_config[agent_name]:
                    old_val = agents_config[agent_name]['fibo_tolerance_pct']
                    new_val = max(0.1, old_val - 0.1)  # Min 0.1%
                    if new_val != old_val:
                        agents_config[agent_name]['fibo_tolerance_pct'] = round(new_val, 2)
                        changes.append(f"fibo_tolerance_pct: {old_val} -> {new_val}")

                # Changer vers un niveau Fibo plus fiable
                if 'fibo_level' in agents_config[agent_name]:
                    fibo_priority = ["0.618", "0.5", "0.382", "0.786", "0.236"]
                    current = agents_config[agent_name]['fibo_level']
                    if current in fibo_priority:
                        idx = fibo_priority.index(current)
                        if idx < len(fibo_priority) - 1:
                            new_level = fibo_priority[idx + 1]
                            agents_config[agent_name]['fibo_level'] = new_level
                            changes.append(f"fibo_level: {current} -> {new_level}")

                # Augmenter cooldown
                if 'cooldown_seconds' in agents_config[agent_name]:
                    old_val = agents_config[agent_name]['cooldown_seconds']
                    new_val = min(300, old_val + 30)
                    if new_val != old_val:
                        agents_config[agent_name]['cooldown_seconds'] = new_val
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

        # === CORRECTION 3: Profit factor < 0.5 -> AJUSTER TP/SL ===
        if global_stats.get('profit_factor', 0) < 0.5 and global_stats.get('total_trades', 0) >= 10:
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
                        'reason': f"Profit factor {global_stats['profit_factor']} < 0.5",
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
        """Analyse par agent"""
        by_agent = defaultdict(lambda: {'profits': []})

        for trade in self.trades:
            agent = trade.get('agent', trade.get('agent_id', 'unknown'))
            by_agent[agent]['profits'].append(trade.get('profit', trade.get('profit_eur', 0)))

        results = {}
        for agent, data in by_agent.items():
            profits = data['profits']
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]

            results[agent] = {
                'total_trades': len(profits),
                'total_profit': round(sum(profits), 2),
                'win_rate': round(len(wins) / len(profits) * 100, 1) if profits else 0,
                'profit_factor': round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 0
            }

        return results

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
        """Retourne les actions executees (filtrees)"""
        logs = self._load_logs()
        return [log for log in logs if log.get('type') == 'ACTION_EXECUTED'][:limit]

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
