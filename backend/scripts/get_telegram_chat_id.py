# -*- coding: utf-8 -*-
"""
G12 - Utilitaire de détection du Chat ID Telegram
Lancez ce script, envoyez un message à votre bot sur Telegram, et le script affichera votre Chat ID.
"""

import requests
import time
import sys
import os

# Ajouter le chemin du backend pour importer la config
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from config import TELEGRAM_CONFIG

def get_chat_id():
    token = TELEGRAM_CONFIG.get("token")
    if not token or token.startswith("VOTRE_"):
        print("❌ Token Telegram non configuré dans config.py !")
        return

    print("--- UTILITAIRE DE DÉTECTION CHAT ID ---")
    print(f"Bot Token: {token[:10]}...{token[-5:]}")
    print("\nÉTAPE 1 : Ouvrez Telegram et cherchez votre bot.")
    print("ÉTAPE 2 : Envoyez-lui n'importe quel message (ex: 'Coucou').")
    print("ÉTAPE 3 : Attendez que ce script détecte votre ID...\n")

    base_url = f"https://api.telegram.org/bot{token}/getUpdates"
    last_update_id = 0

    try:
        while True:
            params = {"offset": last_update_id + 1, "timeout": 30}
            response = requests.get(base_url, params=params, timeout=35)
            
            if response.status_code == 200:
                data = response.json()
                for update in data.get("result", []):
                    last_update_id = update["update_id"]
                    if "message" in update:
                        chat_id = update["message"]["chat"]["id"]
                        user_name = update["message"]["from"].get("first_name", "Inconnu")
                        print(f"✅ ID DÉTECTÉ !")
                        print(f"👤 Utilisateur : {user_name}")
                        print(f"🆔 Chat ID : {chat_id}")
                        print("\n👉 Copiez cet ID dans config.py sous 'chat_id' :")
                        print(f"   'chat_id': '{chat_id}'")
                        input("\nAppuyez sur Entrée pour fermer...")
                        return
            else:
                print(f"❌ Erreur API : {response.status_code}")
                break
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nArrêt du script.")
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    get_chat_id()
