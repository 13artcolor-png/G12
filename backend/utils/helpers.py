# -*- coding: utf-8 -*-
"""
G12 - Fonctions utilitaires
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


def load_json(filepath: Path, default: Any = None) -> Any:
    """Charge un fichier JSON avec fallback"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def save_json(filepath: Path, data: Any, indent: int = 2):
    """Sauvegarde des donnees en JSON"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erreur sauvegarde JSON {filepath}: {e}")
        return False


def format_price(price: float, decimals: int = 2) -> str:
    """Formate un prix pour affichage"""
    return f"${price:,.{decimals}f}"


def format_pnl(pnl: float) -> str:
    """Formate un P&L avec couleur indicative"""
    sign = "+" if pnl >= 0 else ""
    return f"{sign}{pnl:.2f} EUR"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Formate un pourcentage"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def timestamp_now() -> str:
    """Retourne le timestamp actuel ISO"""
    return datetime.now().isoformat()


def calculate_pnl_pct(entry_price: float, current_price: float, direction: str) -> float:
    """Calcule le P&L en pourcentage"""
    if entry_price == 0:
        return 0

    if direction.upper() == "BUY":
        return ((current_price - entry_price) / entry_price) * 100
    else:
        return ((entry_price - current_price) / entry_price) * 100
