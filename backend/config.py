# -*- coding: utf-8 -*-
"""
G12 - Configuration globale
BTCUSD Trading Bot avec 3 agents IA

NOTE: Les valeurs modifiables sont dans database/*_runtime_config.json
Ce fichier contient uniquement les constantes et valeurs par defaut.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv(Path(__file__).parent.parent / ".env")

# =============================================================================
# CHEMINS (CONSTANTES)
# =============================================================================
BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = DATABASE_DIR / "logs"

DATABASE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# CONSTANTES
# =============================================================================
SYMBOL = "BTCUSD"
REQUESTY_URL = "https://router.requesty.ai/v1/chat/completions"

# =============================================================================
# MT5 - VALEURS PAR DEFAUT (runtime: mt5_accounts_runtime.json)
# =============================================================================
MT5_CONFIG = {
    "login": 0,
    "password": "",
    "server": "",
    "path": r"C:\Program Files\EightCap MetaTrader 5\terminal64.exe"
}

MT5_ACCOUNTS = {
    "fibo1": {"login": 0, "password": "", "server": "", "path": MT5_CONFIG["path"], "enabled": False},
    "fibo2": {"login": 0, "password": "", "server": "", "path": MT5_CONFIG["path"], "enabled": False},
    "fibo3": {"login": 0, "password": "", "server": "", "path": MT5_CONFIG["path"], "enabled": False}
}

# =============================================================================
# API KEYS (runtime: api_keys.json) - Charger depuis .env
# =============================================================================
API_KEYS = {
    "fibo1": {
        "provider": "anthropic",
        "key": os.getenv("MOMENTUM_API_KEY", ""),
        "model": "anthropic/claude-sonnet-4-20250514"
    },
    "fibo2": {
        "provider": "openai",
        "key": os.getenv("FIBO_API_KEY", ""),
        "model": "openai/gpt-4o"
    },
    "fibo3": {
        "provider": "alibaba",
        "key": os.getenv("LIQUIDATION_API_KEY", ""),
        "model": "alibaba/qwen3-max"
    }
}

# =============================================================================
# AGENTS - VALEURS PAR DEFAUT (runtime: agents_runtime_config.json)
# =============================================================================
AGENTS_CONFIG = {
    "fibo1": {
        "name": "FIBO1",
        "description": "Trade sur niveaux Fibonacci + ICT/SMC (Compte 1)",
        "enabled": True,
        "color": "#3b82f6",
        "fibo_level": "0.236",
        "fibo_tolerance_pct": 1.0,
        "cooldown_seconds": 120
    },
    "fibo2": {
        "name": "FIBO2",
        "description": "Trade sur niveaux Fibonacci + ICT/SMC (Compte 2)",
        "enabled": True,
        "color": "#10b981",
        "fibo_level": "0.382",
        "fibo_tolerance_pct": 1.0,
        "cooldown_seconds": 120
    },
    "fibo3": {
        "name": "FIBO3",
        "description": "Trade sur niveaux Fibonacci + ICT/SMC (Compte 3)",
        "enabled": True,
        "color": "#f59e0b",
        "fibo_level": "0.618",
        "fibo_tolerance_pct": 1.0,
        "cooldown_seconds": 120
    }
}

# =============================================================================
# RISQUE - VALEURS PAR DEFAUT (runtime: risk_runtime_config.json)
# =============================================================================
RISK_CONFIG = {
    "max_drawdown_pct": 10,
    "max_daily_loss_pct": 5,
    "max_positions_total": 3,
    "emergency_close_pct": 15,
    "winner_never_loser": True
}

# =============================================================================
# TP/SL/SPREAD - VALEURS PAR DEFAUT (runtime: spread_runtime_config.json)
# =============================================================================
SPREAD_CONFIG = {
    "max_spread_points": 2000,
    "spread_check_enabled": True,
    "tp_pct": 0.3,
    "sl_pct": 0.5
}

# Alias pour compatibilite
TPSL_CONFIG = SPREAD_CONFIG

# =============================================================================
# INTERVALLES (CONSTANTES)
# =============================================================================
INTERVALS = {
    "trading_loop": 2,
    "closer_loop": 2,
    "data_refresh": 2,
    "binance_refresh": 2,
    "sentiment_refresh": 60
}

# =============================================================================
# SOURCES DE DONNEES (CONSTANTES)
# =============================================================================
DATA_SOURCES = {
    "binance_futures": {
        "base_url": "https://fapi.binance.com",
        "symbol": "BTCUSDT",
        "enabled": True
    },
    "fear_greed": {
        "url": "https://api.alternative.me/fng/",
        "enabled": True
    }
}

# =============================================================================
# SESSIONS BTC (CONSTANTES)
# =============================================================================
SESSIONS = {
    "asia": {"name": "Asie (Tokyo)", "start": "01:00", "end": "09:00", "volatility": "low"},
    "london": {"name": "Londres", "start": "09:00", "end": "14:00", "volatility": "high"},
    "overlap": {"name": "Killzone (Ldn+NY)", "start": "14:00", "end": "18:00", "volatility": "ultra"},
    "usa": {"name": "New York", "start": "18:00", "end": "22:00", "volatility": "high"},
    "night": {"name": "Post-Market", "start": "22:00", "end": "01:00", "volatility": "low"}
}

# =============================================================================
# FASTAPI (CONSTANTES)
# =============================================================================
API_CONFIG = {
    "host": "127.0.0.1",  # Localhost uniquement (securite)
    "port": 8012,
    "reload": False
}

# =============================================================================
# LOGGING (CONSTANTES)
# =============================================================================
LOG_CONFIG = {
    "level": "INFO",
    "max_trades_history": 1000,  # NON UTILISE - trades.json conserve TOUS les trades de session
    "max_decisions_history": 5000
}
# =============================================================================
# TELEGRAM (PHASE 2) - Charger depuis .env
# =============================================================================
TELEGRAM_CONFIG = {
    "enabled": True,
    "token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    "notify_trades": True,
    "notify_errors": True,
    "notify_strategist": True
}
