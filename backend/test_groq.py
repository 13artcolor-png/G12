# -*- coding: utf-8 -*-
"""
Test direct de l'API Groq
"""
import requests
import json
import time

def test_groq_direct(api_key, model="llama-3.3-70b-versatile"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Tu es un assistant de trading."},
            {"role": "user", "content": "Analyse rapide de BTCUSD: BULLISH ou BEARISH ? reponds en 1 mot."}
        ],
        "max_tokens": 10,
        "temperature": 0.2
    }

    print(f"Test Groq Direct avec {model}...")
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"SUCCESS! Reponse: {content}")
            print(f"Latence: {duration:.2f}s")
            return True
        else:
            print(f"FAILED. Code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

if __name__ == "__main__":
    # Tester avec une cle d'un agent si elle est configuree
    import sys
    import os
    from pathlib import Path

    # Ajouter le chemin parent pour importer DATABASE_DIR
    sys.path.append(str(Path(__file__).parent.parent))
    
    try:
        from backend.config import DATABASE_DIR
        keys_file = DATABASE_DIR / "api_keys.json"
        if keys_file.exists():
            with open(keys_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Chercher une cle Groq
                groq_keys = [k for k in data.get("keys", []) if k.get("provider") == "groq"]
                if groq_keys:
                    key = groq_keys[0]["key"]
                    if not key.startswith("rqsty-"):
                        test_groq_direct(key, groq_keys[0].get("model", "llama-3.3-70b-versatile"))
                    else:
                        print("La cle Groq trouvee est une cle Requesty (rqsty-), le test direct ne fonctionnera pas.")
                else:
                    print("Aucune cle Groq réelle (non-Requesty) trouvée dans api_keys.json pour le test.")
        else:
            print(f"Fichier {keys_file} non trouve.")
    except Exception as e:
        print(f"Erreur lors du test: {e}")
