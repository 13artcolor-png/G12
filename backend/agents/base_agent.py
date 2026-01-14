# -*- coding: utf-8 -*-
"""
G12 - Agent de base
Classe abstraite pour les 3 agents de trading
"""

import requests
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Tuple
import json
import sys
sys.path.append('..')

from config import API_KEYS, REQUESTY_URL, AGENTS_CONFIG, RISK_CONFIG, DATABASE_DIR


def load_risk_runtime_config() -> dict:
    """Charge la config risk depuis le fichier runtime (prioritaire sur config.py)"""
    try:
        config_file = DATABASE_DIR / "risk_runtime_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[BaseAgent] Erreur chargement risk_runtime_config: {e}")
    return RISK_CONFIG


def load_spread_runtime_config() -> dict:
    """Charge la config spread depuis le fichier runtime (prioritaire sur config.py)"""
    from config import SPREAD_CONFIG
    try:
        config_file = DATABASE_DIR / "spread_runtime_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[BaseAgent] Erreur chargement spread_runtime_config: {e}")
    return SPREAD_CONFIG


class BaseAgent(ABC):
    """Classe de base pour tous les agents G12"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._last_api_load = 0
        self._last_config_load = 0

        # Charger la config IA et Agent
        self._load_api_config()
        self.config = self._load_runtime_config(agent_id)

        self.name = self.config.get("name", agent_id.upper())
        self.enabled = self.config.get("enabled", True)
        self.color = self.config.get("color", "#666666")
        print(f"[{self.name}] Initialise avec enabled={self.enabled}")

        # Etat
        self.last_decision = None
        self.last_decision_time = None
        self.cooldown_until = None
        self.open_positions = []  # Liste des positions ouvertes (multi-positions)
        self._positions_lock = threading.Lock()  # Verrou pour acces thread-safe
        self.session_pnl = 0
        self.session_trades = 0
        self.session_wins = 0

        # Charger les positions persistees (survit au redemarrage)
        self._load_positions()

        # Charger les stats de session persistees (survit au redemarrage)
        self._load_session_stats()

    @abstractmethod
    def get_opener_prompt(self, context: Dict) -> str:
        """Retourne le prompt d'ouverture specifique a l'agent"""
        pass

    @abstractmethod
    def get_closer_prompt(self, context: Dict, position: Dict) -> str:
        """Retourne le prompt de fermeture specifique a l'agent"""
        pass

    @abstractmethod
    def should_consider_trade(self, context: Dict) -> Tuple[bool, str]:
        """Verifie si les conditions specifiques a l'agent sont reunies"""
        pass

    def is_in_cooldown(self) -> bool:
        """Verifie si l'agent est en cooldown"""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until

    def set_cooldown(self, seconds: int):
        """Met l'agent en cooldown"""
        from datetime import timedelta
        self.cooldown_until = datetime.now() + timedelta(seconds=seconds)

    def _load_runtime_config(self, agent_id: str) -> Dict:
        """Charge la config depuis agents_runtime_config.json (prioritaire sur config.py)"""
        try:
            config_file = DATABASE_DIR / "agents_runtime_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    all_configs = json.load(f)
                    if agent_id in all_configs:
                        return all_configs[agent_id]
        except Exception as e:
            print(f"[BaseAgent] Erreur chargement runtime config: {e}")

        # Fallback sur config.py
        return AGENTS_CONFIG.get(agent_id, {})

    def reload_config(self):
        """Recharge la config depuis agents_runtime_config.json"""
        import time
        current_time = time.time()

        # Recharger max toutes les 10 secondes
        if current_time - self._last_config_load < 10:
            return

        try:
            config_file = DATABASE_DIR / "agents_runtime_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    all_configs = json.load(f)
                    if self.agent_id in all_configs:
                        self.config = all_configs[self.agent_id]
                        self.enabled = self.config.get("enabled", True)
                        self._last_config_load = current_time
        except Exception as e:
            print(f"[{self.name}] Erreur rechargement config: {e}")

    def _load_api_config(self):
        """Charge dynamiquement la cle API associee a l'agent depuis les fichiers budget/selections"""
        import time
        current_time = time.time()
        
        # Recharger toutes les 30 secondes au maximum (pour eviter trop d'I/O)
        if current_time - self._last_api_load < 30 and hasattr(self, 'api_config'):
            return

        try:
            keys_file = DATABASE_DIR / "api_keys.json"
            selections_file = DATABASE_DIR / "api_selections.json"
            
            if not keys_file.exists() or not selections_file.exists():
                # Fallback sur config.py si les fichiers runtime n'existent pas
                from config import API_KEYS as STATIC_KEYS
                self.api_config = STATIC_KEYS.get(self.agent_id, {})
                return

            with open(keys_file, 'r', encoding='utf-8') as f:
                keys_data = json.load(f)
            with open(selections_file, 'r', encoding='utf-8') as f:
                selections_data = json.load(f)

            key_id = selections_data.get("selections", {}).get(self.agent_id)
            if not key_id:
                self.api_config = {}
                return

            # Trouver la cle correspondante
            selected_key = next((k for k in keys_data.get("keys", []) if k["id"] == key_id), None)
            if selected_key:
                self.api_config = selected_key
                self._last_api_load = current_time
            else:
                self.api_config = {}

        except Exception as e:
            print(f"[{self.agent_id}] Erreur chargement API config: {e}")
            # Fallback en cas d'erreur lecture
            if not hasattr(self, 'api_config'):
                self.api_config = {}

    def call_ai(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Appelle l'IA (via Groq Direct ou Requesty)"""
        # Recharger la config si necessaire
        self._load_api_config()
        
        try:
            if not self.api_config.get('key'):
                print(f"[{self.name}] ERREUR: Pas de cle API configuree!")
                return None

            provider = self.api_config.get("provider", "").lower()
            key = self.api_config['key']
            model = self.api_config.get("model", "anthropic/claude-sonnet-4-20250514")

            # --- ROUTAGE DIRECT GROQ ---
            if provider == "groq":
                url = "https://api.groq.com/openai/v1/chat/completions"
                # Si la cle commence par rqsty-, c'est qu'on l'utilise via Requesty
                # Mais si l'utilisateur veut du direct, il doit fournir la cle groq reelle (gsk_...)
                # On check si la cle semble etre une cle Groq reelle
                if not key.startswith("rqsty-"):
                    headers = {
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    }
                else:
                    # Si c'est une cle Requesty mais provider Groq, on utilise Requesty
                    url = REQUESTY_URL
                    headers = {
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    }
            else:
                url = REQUESTY_URL
                headers = {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                }

            # Injection des regles apprises (PHASE 2 - Self-Learning)
            learned_context = ""
            try:
                rules_file = DATABASE_DIR / "learned_rules.json"
                if rules_file.exists():
                    with open(rules_file, 'r') as f:
                        rules_data = json.load(f)
                        rules = rules_data.get('learned_rules', [])
                        if rules:
                            learned_context = "\n\nRÈGLES APPRISES (Applique-les strictement) :\n" + "\n".join([f"- {r}" for r in rules])
            except Exception as e:
                print(f"[{self.name}] Erreur chargement regles apprises: {e}")

            if system_prompt:
                system_prompt = f"{system_prompt}{learned_context}"
            else:
                system_prompt = learned_context # If no system_prompt, learned_context becomes the system_prompt

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.3
            }

            print(f"[{self.name}] Appel API ({provider}): model={model}, url={url}")

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                print(f"[{self.name}] Erreur API {response.status_code}: {response.text[:500]}")
                return None

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"[{self.name}] Reponse API: {content[:100]}...")
            return content

        except requests.exceptions.Timeout:
            print(f"[{self.name}] Timeout API (30s)")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"[{self.name}] Erreur connexion API: {e}")
            return None
        except Exception as e:
            print(f"[{self.name}] Exception API: {type(e).__name__}: {e}")
            return None

    def _check_auto_stop_thresholds(self) -> Tuple[bool, str]:
        """Verifie les seuils d'auto-desactivation (DD max, WR min)"""
        # Recuperer les seuils de la config
        max_dd_pct = self.config.get("max_drawdown_pct", 10.0)  # En pourcentage (ex: 10 = 10%)
        min_wr = self.config.get("min_winrate_pct", 20.0)
        min_trades = self.config.get("min_trades_for_eval", 10)

        # Pas assez de trades pour evaluer
        if self.session_trades < min_trades:
            return True, "OK"

        # Calculer le win rate
        win_rate = (self.session_wins / self.session_trades * 100) if self.session_trades > 0 else 0

        # Verifier win rate minimum
        if win_rate < min_wr:
            self.enabled = False
            self._save_enabled_state(False)
            return False, f"Auto-stop: WR {win_rate:.1f}% < {min_wr}% (sur {self.session_trades} trades)"

        # Verifier drawdown (PnL negatif) en POURCENTAGE du capital
        if self.session_pnl < 0:
            # Recuperer l'equity de l'agent pour calculer le vrai % de drawdown
            try:
                from core.mt5_connector import get_mt5
                mt5 = get_mt5(self.agent_id)
                if mt5.connect():
                    account = mt5.get_account_info()
                    if account:
                        equity = account.get("equity", 0)
                        if equity > 0:
                            # Calculer le drawdown en % du capital
                            dd_pct = (abs(self.session_pnl) / equity) * 100
                            if dd_pct > max_dd_pct:
                                self.enabled = False
                                self._save_enabled_state(False)
                                return False, f"Auto-stop: DD {dd_pct:.1f}% > {max_dd_pct}% max (perte {abs(self.session_pnl):.2f} EUR sur {equity:.2f} EUR)"
            except Exception as e:
                print(f"[{self.name}] Erreur calcul DD: {e}")

        return True, "OK"

    def _save_enabled_state(self, enabled: bool):
        """Sauvegarde l'etat enabled dans la config"""
        try:
            config_file = DATABASE_DIR / "agents_runtime_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    all_configs = json.load(f)
                if self.agent_id in all_configs:
                    all_configs[self.agent_id]['enabled'] = enabled
                    with open(config_file, 'w') as f:
                        json.dump(all_configs, f, indent=2)
                    print(f"[{self.name}] Etat enabled sauvegarde: {enabled}")
        except Exception as e:
            print(f"[{self.name}] Erreur sauvegarde etat: {e}")

    def _get_positions_file(self):
        """Retourne le chemin du fichier de positions pour cet agent"""
        return DATABASE_DIR / f"positions_{self.agent_id}.json"

    def _get_stats_file(self):
        """Retourne le chemin du fichier de stats de session pour cet agent"""
        return DATABASE_DIR / f"stats_{self.agent_id}.json"

    def _load_session_stats(self):
        """Charge les stats de session depuis le fichier (persistance)"""
        try:
            stats_file = self._get_stats_file()
            if stats_file.exists():
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.session_pnl = data.get('session_pnl', 0)
                    self.session_trades = data.get('session_trades', 0)
                    self.session_wins = data.get('session_wins', 0)
                    if self.session_pnl != 0 or self.session_trades > 0:
                        print(f"[{self.name}] Stats chargees: PnL={self.session_pnl:.2f}, Trades={self.session_trades}, Wins={self.session_wins}")
        except Exception as e:
            print(f"[{self.name}] Erreur chargement stats: {e}")

    def _save_session_stats(self):
        """Sauvegarde les stats de session dans un fichier (persistance)"""
        try:
            stats_file = self._get_stats_file()
            data = {
                'agent_id': self.agent_id,
                'updated_at': datetime.now().isoformat(),
                'session_pnl': round(self.session_pnl, 2),
                'session_trades': self.session_trades,
                'session_wins': self.session_wins
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[{self.name}] Erreur sauvegarde stats: {e}")

    def reset_session_stats(self):
        """Remet a zero les stats de session (nouvelle session)"""
        self.session_pnl = 0
        self.session_trades = 0
        self.session_wins = 0
        self._save_session_stats()
        print(f"[{self.name}] Stats de session remises a zero")

    def get_positions_count(self) -> int:
        """Retourne le nombre de positions ouvertes (thread-safe)"""
        with self._positions_lock:
            return len(self.open_positions)

    def get_positions_copy(self) -> list:
        """Retourne une copie des positions ouvertes (thread-safe)"""
        with self._positions_lock:
            return self.open_positions.copy()

    def add_position(self, position: Dict):
        """Ajoute une position (thread-safe)"""
        with self._positions_lock:
            self.open_positions.append(position)

    def remove_position_by_ticket(self, ticket: int):
        """Retire une position par son ticket (thread-safe)"""
        with self._positions_lock:
            self.open_positions = [p for p in self.open_positions if p.get("ticket") != ticket]

    def set_positions(self, positions: list):
        """Remplace toutes les positions (thread-safe)"""
        with self._positions_lock:
            self.open_positions = positions

    def _load_positions(self):
        """Charge les positions ouvertes depuis le fichier (persistance)"""
        try:
            positions_file = self._get_positions_file()
            if positions_file.exists():
                with open(positions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    with self._positions_lock:
                        self.open_positions = data.get('positions', [])
                    if self.open_positions:
                        print(f"[{self.name}] {len(self.open_positions)} position(s) chargee(s) depuis {positions_file.name}")
        except Exception as e:
            print(f"[{self.name}] Erreur chargement positions: {e}")
            with self._positions_lock:
                self.open_positions = []

    def save_positions(self):
        """Sauvegarde les positions ouvertes dans un fichier (persistance) - thread-safe"""
        try:
            positions_file = self._get_positions_file()
            with self._positions_lock:
                data = {
                    'agent_id': self.agent_id,
                    'updated_at': datetime.now().isoformat(),
                    'positions': self.open_positions.copy()  # Copie pour eviter modif pendant ecriture
                }
            with open(positions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[{self.name}] Erreur sauvegarde positions: {e}")

    def decide_open(self, context: Dict) -> Dict:
        """Decide s'il faut ouvrir une position"""
        # Recharger la config si modifiee
        self.reload_config()

        # Verifier les seuils d'auto-stop
        can_trade, stop_reason = self._check_auto_stop_thresholds()
        if not can_trade:
            decision = {
                "action": "HOLD",
                "reason": stop_reason,
                "agent": self.agent_id
            }
            self._record_decision(decision)
            return decision

        # Verifier si agent desactive
        if not self.enabled:
            decision = {
                "action": "HOLD",
                "reason": "Agent desactive",
                "agent": self.agent_id
            }
            self._record_decision(decision)
            return decision

        # Verifier cooldown
        if self.is_in_cooldown():
            decision = {
                "action": "HOLD",
                "reason": f"En cooldown jusqu'a {self.cooldown_until.strftime('%H:%M:%S')}",
                "agent": self.agent_id
            }
            self._record_decision(decision)
            return decision

        # === CONSENSUS MULTI-AI GUARD ===
        analysis = context.get("analysis", {})
        consensus_bias = analysis.get("bias", "neutral")
        confidence = analysis.get("confidence", 0)

        # Si le consensus est fort (>= 65%), on impose l'alignement
        if confidence >= 65 and consensus_bias != "neutral":
            # On laisse l'IA decider d'abord, mais on va filtrer sa reponse plus tard
            # ou on peut injecter une contrainte forte dans le prompt.
            pass

        # Verifier conditions specifiques a l'agent
        should_trade, reason = self.should_consider_trade(context)
        if not should_trade:
            decision = {
                "action": "HOLD",
                "reason": reason,
                "agent": self.agent_id
            }
            self._record_decision(decision)
            return decision

        # Construire le prompt et appeler l'IA
        system_prompt = self._get_system_prompt(context)
        prompt = self.get_opener_prompt(context)

        response = self.call_ai(prompt, system_prompt)

        if response is None:
            decision = {
                "action": "HOLD",
                "reason": "Erreur API IA",
                "agent": self.agent_id
            }
            self._record_decision(decision)
            return decision

        # Parser la reponse
        decision = self._parse_decision(response)
        decision["agent"] = self.agent_id
        decision["raw_response"] = response

        # === FILTRE FINAL DE CONSENSUS ===
        if confidence >= 65 and consensus_bias != "neutral" and decision["action"] != "HOLD":
            if decision["action"] != consensus_bias.upper():
                old_action = decision["action"]
                decision["action"] = "HOLD"
                decision["reason"] = f"FILTRE CONSENSUS: Agent voulait {old_action} mais le consensus est {consensus_bias} ({confidence}%)"
                print(f"[{self.name}] {decision['reason']}")

        self._record_decision(decision)
        return decision

    def _record_decision(self, decision: Dict):
        """Enregistre la decision pour le dashboard"""
        self.last_decision = decision
        self.last_decision_time = datetime.now()

    def decide_close(self, context: Dict, position: Dict) -> Dict:
        """Decide s'il faut fermer une position"""
        profit = position.get("profit", 0)

        # Protection "Winner never becomes loser"
        risk_config = load_risk_runtime_config()
        if risk_config.get("winner_never_loser", True) and profit >= 0:
            return {
                "action": "KEEP",
                "reason": "Protection: Winner never becomes loser (profit >= 0)",
                "agent": self.agent_id
            }

        # Construire le prompt et appeler l'IA
        system_prompt = self._get_closer_system_prompt()
        prompt = self.get_closer_prompt(context, position)

        response = self.call_ai(prompt, system_prompt)

        if response is None:
            return {
                "action": "KEEP",
                "reason": "Erreur API IA",
                "agent": self.agent_id
            }

        # Parser la reponse
        decision = self._parse_close_decision(response)
        decision["agent"] = self.agent_id
        decision["raw_response"] = response

        return decision

    def _get_strategist_context(self) -> str:
        """Recupere le contexte des actions strategist pour informer l'IA"""
        try:
            from strategist import get_strategist
            strategist = get_strategist()
            return strategist.get_actions_context_for_ia()
        except Exception as e:
            return ""


    def _get_system_prompt(self, context=None) -> str:
        """Prompt systeme pour les decisions d'ouverture"""
        strategist_context = self._get_strategist_context()

        base_prompt = f"""Tu es {self.name}, un agent de trading specialise pour BTCUSD.
Tu fais partie d'un systeme Multi-AI Consensus. Ton role est d'analyser le contexte technique tout en respectant le biais global.

BIAIS GLOBAL ACTUEL (CONSENSUS): {self.last_analysis_bias(context)}

REGLES IMPORTANTES:
1. Reponds UNIQUEMENT par: BUY, SELL, ou HOLD suivi d'une raison courte
2. Sois precis et concis
3. Ne trade que si tu as une conviction forte (>70% confiance)
4. Format de reponse: ACTION: [BUY/SELL/HOLD] | RAISON: [explication courte]

Tu es un agent {self.config.get('description', '')}."""

        if strategist_context:
            base_prompt += f"\n\n{strategist_context}"

        return base_prompt

    def _get_closer_system_prompt(self) -> str:
        """Prompt systeme pour les decisions de fermeture"""
        return f"""Tu es {self.name}, tu geres une position ouverte sur BTCUSD.
Tu dois decider: KEEP (garder) ou CLOSE (fermer).

REGLES IMPORTANTES:
1. Reponds UNIQUEMENT par: KEEP ou CLOSE suivi d'une raison courte
2. Format: ACTION: [KEEP/CLOSE] | RAISON: [explication courte]
3. IMPORTANT: Tu ne peux fermer que si la position est en PERTE
4. Si la position est en profit, reponds toujours KEEP"""

    def _parse_decision(self, response: str) -> Dict:
        """Parse la reponse de l'IA pour extraire la decision"""
        response_upper = response.upper()

        if "BUY" in response_upper:
            action = "BUY"
        elif "SELL" in response_upper:
            action = "SELL"
        else:
            action = "HOLD"

        # Extraire la raison
        reason = response
        if "|" in response:
            parts = response.split("|")
            if len(parts) >= 2:
                reason = parts[1].replace("RAISON:", "").strip()

        return {
            "action": action,
            "reason": reason[:200]  # Limiter la longueur
        }

    def _parse_close_decision(self, response: str) -> Dict:
        """Parse la reponse de l'IA pour la fermeture"""
        response_upper = response.upper()

        if "CLOSE" in response_upper:
            action = "CLOSE"
        else:
            action = "KEEP"

        # Extraire la raison
        reason = response
        if "|" in response:
            parts = response.split("|")
            if len(parts) >= 2:
                reason = parts[1].replace("RAISON:", "").strip()

        return {
            "action": action,
            "reason": reason[:200]
        }

    def last_analysis_bias(self, context: Dict) -> str:
        """Retourne une description textuelle du consensus pour le prompt"""
        analysis = context.get("analysis", {})
        bias = analysis.get("bias", "NEUTRAL").upper()
        conf = analysis.get("confidence", 0)
        reasons = analysis.get("reasons", [])
        
        if bias == "NEUTRAL":
            return "NEUTRE (Pas de consensus fort)"
        
        return f"{bias} (Confiance: {conf}%) - Raisons: {', '.join(reasons[:3])}"

    def record_trade(self, profit: float, won: bool):
        """Enregistre le resultat d'un trade"""
        self.session_pnl += profit
        self.session_trades += 1
        if won:
            self.session_wins += 1

        # Persister les stats (survit au redemarrage)
        self._save_session_stats()

        # Appliquer cooldown si perte
        if not won:
            self.set_cooldown(self.config.get("cooldown_seconds", 60))

    def get_status(self) -> Dict:
        """Retourne le status de l'agent"""
        max_positions = self.config.get("max_positions", 1)
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "enabled": self.enabled,
            "color": self.color,
            "in_cooldown": self.is_in_cooldown(),
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "has_position": len(self.open_positions) > 0,
            "open_positions": self.open_positions,
            "position_count": len(self.open_positions),
            "max_positions": max_positions,
            "session_pnl": round(self.session_pnl, 2),
            "session_trades": self.session_trades,
            "session_wins": self.session_wins,
            "win_rate": round((self.session_wins / self.session_trades * 100), 1) if self.session_trades > 0 else 0,
            "last_decision": self.last_decision,
            "last_decision_time": self.last_decision_time.isoformat() if self.last_decision_time else None
        }
