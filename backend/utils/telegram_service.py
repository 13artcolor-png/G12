# -*- coding: utf-8 -*-
"""
G12 - Telegram Service
Gère l'envoi de notifications et l'interaction via Telegram
"""

import requests
import json
import time
from typing import Optional, Dict
from config import TELEGRAM_CONFIG, DATABASE_DIR

import logging
from pathlib import Path

# Configuration du logging fichier pour Telegram
logger = logging.getLogger("Telegram")
logger.setLevel(logging.INFO)

# Creer le repertoire logs s'il n'existe pas
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Ajouter un FileHandler specifique
file_handler = logging.FileHandler(LOG_DIR / "telegram.log", encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Ajouter aussi un StreamHandler pour voir dans la console (optionnel mais utile pour debug)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: [Telegram] %(message)s'))
logger.addHandler(console_handler)

class TelegramService:
    """Service de communication avec Telegram"""

    def __init__(self):
        self.enabled = TELEGRAM_CONFIG.get("enabled", False)
        self.token = TELEGRAM_CONFIG.get("token", "")
        self.chat_id = TELEGRAM_CONFIG.get("chat_id", "")
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Envoie un message simple"""
        if not self.enabled or not self.token or not self.chat_id:
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"[Telegram] Erreur envoi message: {e}")
            return False

    def notify_trade(self, trade_data: Dict):
        """Notifie l'ouverture ou la fermeture d'un trade"""
        if not TELEGRAM_CONFIG.get("notify_trades"):
            return

        agent = trade_data.get('agent', 'Unknown').upper()
        direction = trade_data.get('direction', 'Unknown')
        profit = trade_data.get('profit')
        
        if profit is not None:
            # Trade fermer
            icon = "✅" if profit > 0 else "❌"
            msg = (
                f"{icon} *TRADE FERMÉ - {agent}*\n"
                f"━━━━━━━━━━━━━━━\n"
                f"🔹 *Direction:* {direction}\n"
                f"💰 *P&L:* `{profit:+.2f} EUR`\n"
                f"📝 *Raison:* {trade_data.get('close_reason', 'N/A')}"
            )
        else:
            # Trade ouvert
            msg = (
                f"🚀 *NOUVEAU TRADE - {agent}*\n"
                f"━━━━━━━━━━━━━━━\n"
                f"🔹 *Direction:* {direction}\n"
                f"📊 *Prix d'entrée:* `{trade_data.get('price', 'N/A')}`\n"
                f"⚖️ *Volume:* `{trade_data.get('volume', 'N/A')}`"
            )

        self.send_message(msg)

    def notify_strategist_action(self, action_type: str, reason: str):
        """Notifie un changement de stratégie par le Strategist"""
        if not TELEGRAM_CONFIG.get("notify_strategist"):
            return

        msg = (
            f"🧠 *STRATEGIST ACTION*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🛠 *Type:* `{action_type}`\n"
            f"💡 *Raison:* {reason}"
        )
        self.send_message(msg)

    def send_status_report(self, status_data: Dict):
        """Envoie un rapport de statut complet"""
        pnl = status_data.get('total_pnl', 0)
        icon = "📈" if pnl > 0 else "📉"
        
        msg = (
            f"{icon} *G12 - RAPPORT D'ÉTAT*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"⏱ *Durée:* `{status_data.get('duration_minutes', 0)} min`\n"
            f"💰 *P&L Total:* `{pnl:+.2f} EUR`\n"
            f"🎯 *Win Rate:* `{status_data.get('win_rate', 0)}%`\n"
            f"📦 *Trades:* `{status_data.get('trades_count', 0)}`"
        )
        self.send_message(msg)


    def poll_commands(self):
        """Lit les nouveaux messages et execute les commandes"""
        if not self.enabled or not self.token:
            return

        try:
            url = f"{self.base_url}/getUpdates"
            params = {"offset": self.last_update_id + 1, "timeout": 5}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for update in data.get("result", []):
                    self.last_update_id = update["update_id"]
                    if "message" in update:
                        msg = update["message"]
                        text = msg.get("text", "")
                        sender_id = str(msg["chat"]["id"])
                        
                        logger.info(f"Check sender: {sender_id} vs config: {self.chat_id}")
                        
                        # Verifier que c'est bien l'utilisateur autorise
                        if self.chat_id and sender_id != str(self.chat_id):
                            logger.warning(f"Message ignore de {sender_id} (Attendu: {self.chat_id})")
                            continue
                            
                        if text.startswith("/"):
                            logger.info(f"Action sur commande: {text}")
                            self.handle_command(text)
                        else:
                            logger.info(f"Message texte ignoré: {text}")

            elif response.status_code != 200:
                logger.error(f"Erreur API: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Erreur polling: {e}")

    def handle_command(self, command: str):
        """Traite une commande reçue"""
        cmd = command.split()[0].lower()
        print(f"[Telegram] Commande reçue : {cmd}")

        if cmd == "/start":
            self.send_message("👋 *Bienvenue sur le Command Center G12 !*\n\nCommandes disponibles :\n/status - État du bot\n/pnl - Profit actuel\n/stop - Arrêt d'urgence")
        
        elif cmd == "/status":
            self.send_status_report_auto()
            
        elif cmd == "/pnl":
            self.send_pnl_report()
            
        elif cmd == "/stop":
            # Appel a une fonction globale d'arret (a implementer dans trading_loop ou via flag)
            self.send_message("🛑 *COMMANDE D'ARRÊT REÇUE*\nLe trading va être stoppé au prochain cycle.")
            # Pour l'instant, on peut emettre un signal ou modifier un fichier de config runtime
            self._trigger_emergency_stop()

    def send_status_report_auto(self):
        """Genere et envoie un rapport de statut en interrogeant le backend"""
        try:
            # On recupere les donnees via le SessionLogger
            from session_logger import get_session_logger
            logger = get_session_logger()
            stats = logger._calculate_stats()
            
            pnl = stats.get('total_profit', 0)
            icon = "📈" if pnl >= 0 else "📉"
            
            msg = (
                f"{icon} *G12 - STATUT ACTUEL*\n"
                f"━━━━━━━━━━━━━━━\n"
                f"💰 *P&L Session:* `{pnl:+.2f} EUR`\n"
                f"🎯 *Win Rate:* `{stats.get('win_rate', 0)}%`\n"
                f"📦 *Trades:* `{len(logger.trades)}`"
            )
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Erreur rapport : {e}")

    def send_pnl_report(self):
        """Envoie le detail du P&L par agent"""
        try:
            from session_logger import get_session_logger
            logger = get_session_logger()
            
            msg = "*📊 P&L PAR AGENT*\n━━━━━━━━━━━━━━━\n"
            
            # Note: on pourrait enrichir cela avec l'equity reelle
            for agent in ["fibo1", "fibo2", "fibo3"]:
                pnl = 0
                for t in logger.trades:
                    if t.get('agent') == agent:
                        pnl += t.get('profit', 0)
                icon = "🔹"
                msg += f"{icon} *{agent.upper()}:* `{pnl:+.2f} EUR`\n"
                
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Erreur P&L : {e}")

    def _trigger_emergency_stop(self):
        """Désactive le trading dans la config de risque"""
        try:
            config_path = DATABASE_DIR / "risk_runtime_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                config["trading_enabled"] = False
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.send_message("✅ *CONFIGURATION MISE À JOUR*\n`trading_enabled` défini sur `False`.")
                print("[Telegram] Arrêt d'urgence déclenché via config.")
        except Exception as e:
            print(f"[Telegram] Erreur arrêt d'urgence: {e}")
            self.send_message(f"❌ Erreur lors de l'arrêt : {e}")

    def _load_data_sources(self):
        """Charge les sources de données nécessaires (aggregator, etc.)"""
        pass

# Singleton
_telegram_instance = None

def get_telegram() -> TelegramService:
    """Retourne l'instance Telegram singleton"""
    global _telegram_instance
    if _telegram_instance is None:
        _telegram_instance = TelegramService()
    return _telegram_instance
