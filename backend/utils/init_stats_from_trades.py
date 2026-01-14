# -*- coding: utf-8 -*-
"""
Initialise les fichiers stats_*.json depuis trades.json existant
Execute une seule fois pour recuperer les stats de la session en cours
"""

import json
from pathlib import Path
from datetime import datetime

DATABASE_DIR = Path(__file__).parent.parent / "database"
TRADES_FILE = DATABASE_DIR / "trades.json"

def init_stats_from_trades():
    """Calcule et sauvegarde les stats de chaque agent depuis trades.json"""

    # Charger trades.json
    if not TRADES_FILE.exists():
        print("trades.json non trouve")
        return

    with open(TRADES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trades = data.get('trades', [])
    print(f"Trouve {len(trades)} trades")

    # Calculer les stats par agent
    stats = {
        'momentum': {'pnl': 0, 'trades': 0, 'wins': 0},
        'fibo': {'pnl': 0, 'trades': 0, 'wins': 0},
        'liquidation': {'pnl': 0, 'trades': 0, 'wins': 0}
    }

    for trade in trades:
        agent_id = trade.get('agent_id')
        if agent_id not in stats:
            continue

        profit = trade.get('profit_eur', 0)
        won = trade.get('won', profit > 0)

        stats[agent_id]['pnl'] += profit
        stats[agent_id]['trades'] += 1
        if won:
            stats[agent_id]['wins'] += 1

    # Sauvegarder dans les fichiers stats_*.json
    for agent_id, agent_stats in stats.items():
        stats_file = DATABASE_DIR / f"stats_{agent_id}.json"
        stats_data = {
            'agent_id': agent_id,
            'updated_at': datetime.now().isoformat(),
            'session_pnl': round(agent_stats['pnl'], 2),
            'session_trades': agent_stats['trades'],
            'session_wins': agent_stats['wins']
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2)

        win_rate = (agent_stats['wins'] / agent_stats['trades'] * 100) if agent_stats['trades'] > 0 else 0
        print(f"{agent_id}: PnL={agent_stats['pnl']:.2f} EUR, Trades={agent_stats['trades']}, Wins={agent_stats['wins']} ({win_rate:.1f}%)")

    # Total
    total_pnl = sum(s['pnl'] for s in stats.values())
    total_trades = sum(s['trades'] for s in stats.values())
    total_wins = sum(s['wins'] for s in stats.values())
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print(f"\nTOTAL: PnL={total_pnl:.2f} EUR, Trades={total_trades}, Wins={total_wins} ({win_rate:.1f}%)")
    print("\nFichiers stats_*.json crees avec succes!")

if __name__ == "__main__":
    init_stats_from_trades()
