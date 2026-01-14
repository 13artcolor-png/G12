# -*- coding: utf-8 -*-
"""
G12 - Script de recuperation des trades manquants
USAGE UNIQUE pour recuperer les trades fermes depuis MT5
qui n'ont pas ete logges a cause du bug de persistence.

Executer: python recover_session_trades.py
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mt5_connector import get_mt5
from config import DATABASE_DIR


def recover_trades():
    """Recupere les trades manquants depuis MT5 et les ajoute a la session"""

    # 1. Charger la session active
    session_file = DATABASE_DIR / "session.json"
    if not session_file.exists():
        print("[RECOVERY] ERREUR: Aucune session active (session.json non trouve)")
        return False

    with open(session_file, 'r', encoding='utf-8') as f:
        session_data = json.load(f)

    session_start = session_data.get('start_time')
    if not session_start:
        print("[RECOVERY] ERREUR: Session sans date de debut")
        return False

    # Convertir en datetime
    from_date = datetime.fromisoformat(session_start)
    to_date = datetime.now()

    print(f"[RECOVERY] Session: {session_data.get('id')}")
    print(f"[RECOVERY] Periode: {from_date} -> {to_date}")
    print()

    # 2. Recuperer les deals de chaque compte MT5
    all_deals = []

    for agent_id in ["fibo1", "fibo2", "fibo3"]:
        print(f"[RECOVERY] Connexion compte {agent_id}...")

        try:
            mt5 = get_mt5(agent_id)
            if not mt5.connect():
                print(f"[RECOVERY]   -> Echec connexion {agent_id}")
                continue

            deals = mt5.get_history_deals(from_date=from_date, to_date=to_date)
            print(f"[RECOVERY]   -> {len(deals)} deal(s) G12 trouve(s)")

            for deal in deals:
                deal['_agent_id'] = agent_id
                all_deals.append(deal)

        except Exception as e:
            print(f"[RECOVERY]   -> Erreur {agent_id}: {e}")

    if not all_deals:
        print()
        print("[RECOVERY] Aucun deal G12 trouve dans l'historique MT5")
        return True

    # 3. Filtrer les deals deja presents dans la session
    existing_tickets = set()
    for trade in session_data.get('trades', []):
        if trade.get('ticket'):
            existing_tickets.add(trade['ticket'])

    new_deals = [d for d in all_deals if d['ticket'] not in existing_tickets]

    if not new_deals:
        print()
        print("[RECOVERY] Tous les deals sont deja dans la session")
        return True

    print()
    print(f"[RECOVERY] {len(new_deals)} nouveau(x) deal(s) a ajouter:")

    # 4. Ajouter les deals manquants
    trades_to_add = []
    for deal in new_deals:
        trade = {
            'ticket': deal['ticket'],
            'agent': deal['_agent_id'],
            'direction': deal['type'],
            'volume': deal['volume'],
            'entry_price': 0,  # Non disponible dans le deal de sortie
            'exit_price': deal['price'],
            'profit': deal['profit'],
            'close_reason': 'TP' if deal['profit'] > 0 else 'SL',
            'timestamp': datetime.fromtimestamp(deal['time']).isoformat(),
            'recovered': True  # Marqueur pour indiquer que c'est une recuperation
        }
        trades_to_add.append(trade)

        profit_str = f"+{deal['profit']:.2f}" if deal['profit'] > 0 else f"{deal['profit']:.2f}"
        print(f"   - Ticket {deal['ticket']}: {deal['_agent_id']} {deal['type']} {profit_str} EUR")

    # Trier par timestamp
    trades_to_add.sort(key=lambda x: x['timestamp'])

    # 5. Mettre a jour session.json
    if 'trades' not in session_data:
        session_data['trades'] = []

    session_data['trades'].extend(trades_to_add)
    session_data['updated_at'] = datetime.now().isoformat()

    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    print()
    print(f"[RECOVERY] {len(trades_to_add)} trade(s) ajoute(s) a session.json")

    # 6. Mettre a jour trades.json aussi
    trades_file = DATABASE_DIR / "trades.json"
    trades_data = {"trades": []}

    if trades_file.exists():
        try:
            with open(trades_file, 'r', encoding='utf-8') as f:
                trades_data = json.load(f)
        except:
            pass

    # Ajouter les trades au format du logger
    for trade in trades_to_add:
        log_trade = {
            "id": f"trade_{trade['timestamp'].replace(':', '').replace('-', '').replace('T', '_')}_{trade['ticket']}",
            "ticket": trade['ticket'],
            "agent_id": trade['agent'],
            "symbol": "BTCUSD",
            "direction": trade['direction'],
            "volume": trade['volume'],
            "entry_price": trade['entry_price'],
            "exit_price": trade['exit_price'],
            "profit_eur": round(trade['profit'], 2),
            "close_reason": trade['close_reason'],
            "timestamp": trade['timestamp'],
            "won": trade['profit'] > 0,
            "recovered": True
        }
        trades_data['trades'].insert(0, log_trade)

    with open(trades_file, 'w', encoding='utf-8') as f:
        json.dump(trades_data, f, indent=2, ensure_ascii=False)

    print(f"[RECOVERY] {len(trades_to_add)} trade(s) ajoute(s) a trades.json")

    # 7. Resume
    total_profit = sum(t['profit'] for t in trades_to_add)
    wins = sum(1 for t in trades_to_add if t['profit'] > 0)

    print()
    print("=" * 50)
    print("[RECOVERY] RECUPERATION TERMINEE")
    print(f"   Trades recuperes: {len(trades_to_add)}")
    print(f"   Gagnants: {wins}")
    print(f"   Perdants: {len(trades_to_add) - wins}")
    print(f"   Profit total: {total_profit:+.2f} EUR")
    print("=" * 50)

    return True


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("G12 - RECUPERATION DES TRADES MANQUANTS")
    print("=" * 50)
    print()

    success = recover_trades()

    if success:
        print()
        print("[RECOVERY] Rafraichissez le dashboard pour voir les trades.")
    else:
        print()
        print("[RECOVERY] Echec de la recuperation.")
