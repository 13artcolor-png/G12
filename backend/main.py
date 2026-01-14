# -*- coding: utf-8 -*-
"""
G12 - API FastAPI
Backend API pour le dashboard
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import threading
import time
import json
from datetime import datetime
from pathlib import Path

# Imports G12
from config import API_CONFIG, DATABASE_DIR, AGENTS_CONFIG
from core.mt5_connector import get_mt5

# ============================================================================
# CACHE GLOBAL pour eviter les blocages MT5 sur /api/status
# Le cache est mis a jour par les boucles de trading en arriere-plan
# ============================================================================
_status_cache = {
    "timestamp": None,
    "data": None,
    "updating": False
}
_status_cache_max_age = 10  # Secondes max avant refresh
from core.trading_loop import get_trading_loop
from core.closer_loop import get_closer_loop
from data.aggregator import get_aggregator
from data.binance_data import get_binance
from data.sentiment import get_sentiment
from risk.risk_manager import get_risk_manager
from utils.logger import get_logger


def telegram_worker():
    """Worker thread pour Telegram polling"""
    from utils.telegram_service import get_telegram
    telegram = get_telegram()
    print("[Telegram] Worker demarre")
    while True:
        try:
            telegram.poll_commands()
        except Exception as e:
            print(f"[Telegram] Erreur worker: {e}")
        time.sleep(1)


@asynccontextmanager
async def lifespan(app):
    """Gestion du cycle de vie de l'application"""
    print("Demarrage services...")

    # Fonction pour demarrer les boucles avec delai (laisse le serveur HTTP demarrer d'abord)
    def start_loops_delayed():
        time.sleep(2)  # Attendre que le serveur HTTP soit pret
        print("[Lifespan] Demarrage des boucles de trading...")

        # IMPORTANT: Ne PAS auto-demarrer de session
        # La session doit PERSISTER entre les redemarrages
        # Nouvelle session uniquement quand l'utilisateur clique "Nouvelle Session"
        from session_logger import get_session_logger
        logger = get_session_logger()
        if logger.session_id:
            print(f"[Lifespan] Session existante restauree: {logger.session_id}")
        else:
            print("[Lifespan] Aucune session active - en attente de clic 'Nouvelle Session'")

        # Lancer TradingLoop
        trading_loop = get_trading_loop()
        threading.Thread(target=trading_loop.start, daemon=True).start()

        # Lancer CloserLoop
        closer_loop = get_closer_loop()
        threading.Thread(target=closer_loop.start, daemon=True).start()

    # Lancer les boucles dans un thread separe avec delai
    threading.Thread(target=start_loops_delayed, daemon=True).start()

    # Lancer Telegram Worker
    threading.Thread(target=telegram_worker, daemon=True).start()

    yield  # L'application tourne ici

    # Cleanup
    print("Arret des services...")


# FastAPI app
app = FastAPI(
    title="G12 - BTCUSD Trading Bot",
    description="API pour le bot de trading BTCUSD avec 3 agents IA",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# FONCTION HELPER - Liste dynamique des agents
# =============================================================================
def get_agent_ids() -> list:
    """
    Retourne la liste des IDs d'agents depuis agents_runtime_config.json
    Evite le hardcoding de ["fibo1", "fibo2", "fibo3"]
    """
    config_file = DATABASE_DIR / "agents_runtime_config.json"
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                agents_config = json.load(f)
            return list(agents_config.keys())
    except Exception as e:
        print(f"[CONFIG] Erreur lecture agents: {e}")
    # Fallback sur AGENTS_CONFIG si fichier non disponible
    return list(AGENTS_CONFIG.keys())


def get_first_enabled_agent() -> str:
    """
    Retourne l'ID du premier agent actif (enabled=True)
    Utilise pour les connexions MT5 generiques
    """
    config_file = DATABASE_DIR / "agents_runtime_config.json"
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                agents_config = json.load(f)
            for agent_id, config in agents_config.items():
                if config.get('enabled', False):
                    return agent_id
    except Exception as e:
        print(f"[CONFIG] Erreur lecture agent actif: {e}")
    # Fallback: premier agent de la liste
    agent_ids = get_agent_ids()
    return agent_ids[0] if agent_ids else "fibo1"


# Charger config spread au demarrage
def load_spread_config():
    """Charge la config spread depuis le fichier runtime"""
    from config import SPREAD_CONFIG
    config_file = DATABASE_DIR / "spread_runtime_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
            for key, value in saved_config.items():
                SPREAD_CONFIG[key] = value
            print(f"[CONFIG] Config spread chargee: {saved_config}")
        except Exception as e:
            print(f"[CONFIG] Erreur chargement config spread: {e}")


# Charger au demarrage
load_spread_config()


# =============================================================================
# MODELS
# =============================================================================
class AgentToggle(BaseModel):
    agent_id: str
    enabled: bool

class ConfigUpdate(BaseModel):
    key: str
    value: Any

class AgentConfigUpdate(BaseModel):
    """Validation pour les mises a jour de config agent"""
    enabled: Optional[bool] = None
    min_fibo1_pct: Optional[float] = None
    cooldown_seconds: Optional[int] = None
    fibo_level: Optional[str] = None
    fibo_tolerance_pct: Optional[float] = None
    min_fibo3_usd: Optional[int] = None
    funding_rate_threshold: Optional[float] = None
    position_size_pct: Optional[float] = None
    max_positions: Optional[int] = None

    class Config:
        extra = "allow"  # Permettre des champs supplementaires

class RiskConfigUpdate(BaseModel):
    """Validation pour les mises a jour de config risque"""
    max_drawdown_pct: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    max_positions_total: Optional[int] = None
    emergency_close_pct: Optional[float] = None
    winner_never_loser: Optional[bool] = None

class SpreadConfigUpdate(BaseModel):
    """Validation pour les mises a jour de config spread"""
    max_spread_points: Optional[int] = None
    spread_check_enabled: Optional[bool] = None
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None


# =============================================================================
# ENDPOINTS - STATUS
# =============================================================================
@app.get("/")
async def root():
    """Page d'accueil - redirige vers le dashboard"""
    response = FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _build_status_data():
    """Construit les donnees de status (appele en arriere-plan)"""
    try:
        # Utiliser le premier agent actif au lieu de hardcoder "fibo1"
        mt5 = get_mt5(get_first_enabled_agent())
        aggregator = get_aggregator()
        trading_loop = get_trading_loop()
        closer_loop = get_closer_loop()
        risk = get_risk_manager()
        logger = get_logger()

        from session_logger import get_session_logger
        session_logger = get_session_logger()
        g11_session = session_logger.get_status()

        context = aggregator.get_full_context() or {}
        global_stats = logger.get_global_stats(7)
        recent_decisions = logger.get_recent_decisions(limit=20)
        account_data = context.get("account") or {}

        return {
            "timestamp": datetime.now().isoformat(),
            "mt5": {
                "connected": mt5.connected,
                "account": None  # Ne pas appeler get_account_info() ici, trop lent
            },
            "account": account_data,
            "price": context.get("price"),
            "futures": context.get("futures"),
            "sentiment": context.get("sentiment"),
            "macro": context.get("macro"),
            "whales": context.get("whales"),
            "whale_list": context.get("whale_list"),
            "session": context.get("session"),
            "g11_session": g11_session,
            "analysis": context.get("analysis"),
            "positions": account_data.get("positions", []),
            "agents": trading_loop.get_status().get("agents", {}),
            "risk": risk.get_status(),
            "loops": {
                "trading": trading_loop.get_status(),
                "closer": closer_loop.get_status()
            },
            "stats": global_stats,
            "decisions": recent_decisions
        }
    except Exception as e:
        print(f"[API] Erreur build status: {e}")
        return None


def _update_status_cache_background():
    """Met a jour le cache status en arriere-plan"""
    global _status_cache
    if _status_cache["updating"]:
        return  # Deja en cours

    _status_cache["updating"] = True
    try:
        data = _build_status_data()
        if data:
            _status_cache["data"] = data
            _status_cache["timestamp"] = time.time()
    finally:
        _status_cache["updating"] = False


@app.get("/api/status")
async def get_status():
    """Status complet du systeme (utilise cache pour eviter blocage MT5)"""
    global _status_cache

    now = time.time()
    cache_age = now - (_status_cache["timestamp"] or 0)

    # Si le cache est recent, le retourner immediatement
    if _status_cache["data"] and cache_age < _status_cache_max_age:
        return _status_cache["data"]

    # Si pas de cache ou cache trop vieux, lancer update dans un VRAI thread separe
    if not _status_cache["updating"]:
        threading.Thread(target=_update_status_cache_background, daemon=True).start()

    # Retourner le cache existant (meme vieux) ou donnees minimales
    if _status_cache["data"]:
        return _status_cache["data"]

    # Pas de cache du tout - retourner donnees minimales
    return {
        "timestamp": datetime.now().isoformat(),
        "mt5": {"connected": False, "account": None},
        "account": {},
        "price": None,
        "futures": {},
        "sentiment": {},
        "macro": None,
        "whales": None,
        "whale_list": [],
        "session": None,
        "g11_session": {},
        "analysis": None,
        "positions": [],
        "agents": {},
        "risk": {},
        "loops": {"trading": {}, "closer": {}},
        "stats": {},
        "decisions": []
    }


@app.get("/api/context")
async def get_context():
    """Contexte complet pour debug"""
    aggregator = get_aggregator()
    return aggregator.get_full_context()


# =============================================================================
# ENDPOINTS - AGENTS
# =============================================================================
@app.get("/api/agents")
async def get_agents():
    """Liste des agents et leur status"""
    trading_loop = get_trading_loop()
    logger = get_logger()

    agents_status = {}
    for agent_id, agent in trading_loop.agents.items():
        status = agent.get_status()
        status["stats_7d"] = logger.get_agent_stats(agent_id, 7)
        agents_status[agent_id] = status

    return agents_status


@app.post("/api/agents/toggle")
async def toggle_agent(data: AgentToggle):
    """Active/desactive un agent (POST)"""
    return _toggle_agent_impl(data.agent_id, data.enabled)


@app.get("/api/agents/toggle")
async def toggle_agent_get(agent_id: str, enabled: bool):
    """Active/desactive un agent (GET - evite blocage POST)"""
    return _toggle_agent_impl(agent_id, enabled)


def _toggle_agent_impl(agent_id: str, enabled: bool):
    """Implementation commune pour toggle agent"""
    global _status_cache

    trading_loop = get_trading_loop()

    if agent_id not in trading_loop.agents:
        raise HTTPException(status_code=404, detail="Agent non trouve")

    # Mettre a jour en memoire
    trading_loop.agents[agent_id].enabled = enabled

    # IMPORTANT: Sauvegarder aussi dans le fichier de config
    try:
        config_file = DATABASE_DIR / "agents_runtime_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                all_configs = json.load(f)
            if agent_id in all_configs:
                all_configs[agent_id]['enabled'] = enabled
                with open(config_file, 'w') as f:
                    json.dump(all_configs, f, indent=2)
                print(f"[API] Agent {agent_id} {'active' if enabled else 'desactive'} et sauvegarde")
    except Exception as e:
        print(f"[API] Erreur sauvegarde toggle: {e}")

    # Mettre a jour le cache immediatement pour cet agent
    if _status_cache["data"] and "agents" in _status_cache["data"]:
        if agent_id in _status_cache["data"]["agents"]:
            _status_cache["data"]["agents"][agent_id]["enabled"] = enabled

    return {"success": True, "agent_id": agent_id, "enabled": enabled}


@app.get("/api/agents/{agent_id}/decisions")
async def get_agent_decisions(agent_id: str, limit: int = 50):
    """Decisions recentes d'un agent"""
    logger = get_logger()
    return logger.get_recent_decisions(agent_id, limit)


# =============================================================================
# ENDPOINTS - TRADING
# =============================================================================
@app.post("/api/trading/start")
async def start_trading(background_tasks: BackgroundTasks):
    """Demarre les boucles de trading"""
    global trading_thread, closer_thread
    trading_loop = get_trading_loop()
    closer_loop = get_closer_loop()

    # Si deja en cours, rien a faire
    if trading_loop.running and closer_loop.running:
        return {"success": True, "message": "Trading deja en cours"}

    # Redemarrer les boucles si elles ont ete arretees
    if not trading_loop.running:
        trading_loop.running = True
        trading_thread = threading.Thread(target=trading_loop.start, daemon=True)
        trading_thread.start()
        print("[API] TradingLoop redemarre")

    if not closer_loop.running:
        closer_loop.running = True
        closer_thread = threading.Thread(target=closer_loop.start, daemon=True)
        closer_thread.start()
        print("[API] CloserLoop redemarre")

    return {"success": True, "message": "Trading demarre"}


@app.post("/api/trading/stop")
async def stop_trading():
    """Arrete les boucles de trading (non-bloquant)"""
    trading_loop = get_trading_loop()
    closer_loop = get_closer_loop()
    trading_loop.running = False
    closer_loop.running = False
    print("[API] Trading arrete via /api/trading/stop")
    return {"success": True, "message": "Trading arrete"}


@app.post("/api/trading/close-all")
async def close_all_positions():
    """Ferme toutes les positions"""
    closer_loop = get_closer_loop()
    result = closer_loop.close_all()
    return result


# =============================================================================
# ENDPOINTS - RISK
# =============================================================================
@app.get("/api/risk")
async def get_risk_status():
    """Status du gestionnaire de risque"""
    risk = get_risk_manager()
    return risk.get_status()


@app.post("/api/risk/halt")
async def halt_trading(reason: str = "Manual halt"):
    """Arrete le trading (halt)"""
    risk = get_risk_manager()
    risk.halt_trading(reason)
    return {"success": True, "message": f"Trading halte: {reason}"}


@app.post("/api/risk/resume")
async def resume_trading():
    """Reprend le trading"""
    risk = get_risk_manager()
    risk.resume_trading()
    return {"success": True, "message": "Trading repris"}


# =============================================================================
# ENDPOINTS - DATA
# =============================================================================
@app.get("/api/data/binance")
async def get_binance_data():
    """Donnees Binance Futures"""
    binance = get_binance()
    return binance.get_all_data()


@app.get("/api/data/sentiment")
async def get_sentiment_data():
    """Donnees de sentiment"""
    sentiment = get_sentiment()
    return sentiment.get_all_sentiment()


@app.get("/api/data/price")
async def get_price_data():
    """Prix et technique MT5"""
    aggregator = get_aggregator()
    return aggregator.get_price_data()


# =============================================================================
# ENDPOINTS - LOGS
# =============================================================================
@app.get("/api/trades")
async def get_trades(limit: int = 100, agent: str = None):
    """Historique des trades - depuis session.json"""
    from session_logger import get_session_logger
    session_logger = get_session_logger()

    # Utiliser les trades de la session active
    trades = session_logger.trades[-limit:] if session_logger.trades else []

    # Filtrer par agent si specifie
    if agent and agent != 'all':
        trades = [t for t in trades if t.get('agent') == agent or t.get('agent_id') == agent]

    # Formater les trades pour compatibilite avec le frontend
    formatted_trades = []
    for t in trades:
        formatted_trades.append({
            'agent_id': t.get('agent') or t.get('agent_id'),
            'ticket': t.get('ticket'),
            'direction': t.get('direction'),
            'volume': t.get('volume'),
            'entry_price': t.get('entry_price', 0),
            'exit_price': t.get('exit_price', 0),
            'profit': t.get('profit', 0),
            'profit_eur': t.get('profit', 0),
            'close_reason': t.get('close_reason', ''),
            'timestamp': t.get('timestamp', '')
        })

    return {"trades": formatted_trades}


@app.get("/api/decisions")
async def get_decisions(limit: int = 200):
    """Historique des decisions"""
    logger = get_logger()
    return logger.get_recent_decisions(limit=limit)


@app.get("/api/stats")
async def get_stats(days: int = 7):
    """Statistiques globales"""
    logger = get_logger()
    return logger.get_global_stats(days)


# =============================================================================
# ENDPOINTS - CONFIG LABORATOIRE
# =============================================================================
@app.get("/api/config")
async def get_config():
    """Configuration actuelle"""
    config_file = DATABASE_DIR / "config.json"
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


@app.get("/api/config/all")
async def get_all_config():
    """Toutes les configurations pour le laboratoire - charge les valeurs sauvegardees"""
    from config import AGENTS_CONFIG, RISK_CONFIG, SPREAD_CONFIG
    import copy

    # Charger les configs depuis les fichiers JSON (valeurs sauvegardees)
    # Si le fichier n'existe pas, utiliser les valeurs par defaut de config.py

    # Agents config - copie profonde pour ne pas modifier l'original
    agents_config = copy.deepcopy(AGENTS_CONFIG)
    agents_runtime_file = DATABASE_DIR / "agents_runtime_config.json"
    if agents_runtime_file.exists():
        try:
            with open(agents_runtime_file, 'r') as f:
                saved_agents = json.load(f)
                # Fusionner avec les defaults (les valeurs sauvegardees ont priorite)
                for agent_id, agent_data in saved_agents.items():
                    if agent_id in agents_config:
                        agents_config[agent_id].update(agent_data)
                print(f"[CONFIG] Agents config charges depuis agents_runtime_config.json")
        except Exception as e:
            print(f"[API] Erreur chargement agents_runtime_config.json: {e}")

    # Risk config
    risk_config = copy.deepcopy(RISK_CONFIG)
    risk_runtime_file = DATABASE_DIR / "risk_runtime_config.json"
    if risk_runtime_file.exists():
        try:
            with open(risk_runtime_file, 'r') as f:
                saved_risk = json.load(f)
                risk_config.update(saved_risk)
                print(f"[CONFIG] Risk config charge depuis risk_runtime_config.json")
        except Exception as e:
            print(f"[API] Erreur chargement risk_runtime_config.json: {e}")

    # Spread config
    spread_config = copy.deepcopy(SPREAD_CONFIG)
    spread_runtime_file = DATABASE_DIR / "spread_runtime_config.json"
    if spread_runtime_file.exists():
        try:
            with open(spread_runtime_file, 'r') as f:
                saved_spread = json.load(f)
                spread_config.update(saved_spread)
                print(f"[CONFIG] Spread config charge depuis spread_runtime_config.json")
        except Exception as e:
            print(f"[API] Erreur chargement spread_runtime_config.json: {e}")

    return {
        "agents": agents_config,
        "risk": risk_config,
        "spread": spread_config
    }


@app.post("/api/config/agent/{agent_id}")
async def update_agent_config(agent_id: str, updates: Dict):
    """Met a jour la configuration d'un agent"""
    from config import AGENTS_CONFIG

    if agent_id not in AGENTS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} non trouve")

    # Sauvegarder dans un fichier JSON
    config_file = DATABASE_DIR / "agents_runtime_config.json"
    try:
        # Charger config existante
        try:
            with open(config_file, 'r') as f:
                runtime_config = json.load(f)
        except Exception:
            runtime_config = {}

        # Initialiser l'agent si pas encore present
        if agent_id not in runtime_config:
            runtime_config[agent_id] = dict(AGENTS_CONFIG[agent_id])

        # Mettre a jour TOUTES les cles (pas seulement celles existantes)
        for key, value in updates.items():
            runtime_config[agent_id][key] = value
            # Aussi mettre a jour en memoire
            AGENTS_CONFIG[agent_id][key] = value

        # Sauvegarder
        with open(config_file, 'w') as f:
            json.dump(runtime_config, f, indent=2)

        print(f"[API] Config agent {agent_id} sauvegardee: {list(updates.keys())}")
        return {"success": True, "agent_id": agent_id, "config": runtime_config[agent_id]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config/risk")
async def update_risk_config(updates: Dict):
    """Met a jour la configuration de risque"""
    from config import RISK_CONFIG
    from risk.risk_manager import get_risk_manager

    # Mettre a jour la config en memoire
    for key, value in updates.items():
        if key in RISK_CONFIG:
            RISK_CONFIG[key] = value

    # Sauvegarder
    config_file = DATABASE_DIR / "risk_runtime_config.json"
    try:
        with open(config_file, 'w') as f:
            json.dump(RISK_CONFIG, f, indent=2)

        # Recharger le RiskManager avec la nouvelle config
        risk_manager = get_risk_manager()
        risk_manager.reload_config()

        return {"success": True, "config": RISK_CONFIG}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config/spread")
async def update_spread_config(updates: Dict):
    """Met a jour la configuration spread/TP/SL"""
    from config import SPREAD_CONFIG
    from risk.risk_manager import get_risk_manager

    # Mettre a jour la config en memoire
    for key, value in updates.items():
        if key in SPREAD_CONFIG:
            SPREAD_CONFIG[key] = value

    # Sauvegarder
    config_file = DATABASE_DIR / "spread_runtime_config.json"
    try:
        with open(config_file, 'w') as f:
            json.dump(SPREAD_CONFIG, f, indent=2)

        # Recharger le RiskManager avec la nouvelle config spread
        risk_manager = get_risk_manager()
        risk_manager.reload_config()

        return {"success": True, "config": SPREAD_CONFIG}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config")
async def update_config(updates: Dict):
    """Met a jour la configuration generale"""
    config_file = DATABASE_DIR / "config.json"
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        config.update(updates)

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return {"success": True, "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENDPOINTS - MT5 CONFIG
# =============================================================================
@app.get("/api/config/mt5")
async def get_mt5_config():
    """Recupere la configuration MT5"""
    from config import MT5_CONFIG
    # Ne pas renvoyer le password
    return {
        "login": MT5_CONFIG.get("login"),
        "server": MT5_CONFIG.get("server"),
        "path": MT5_CONFIG.get("path")
    }


@app.post("/api/config/mt5")
async def update_mt5_config(updates: Dict):
    """Met a jour la configuration MT5"""
    config_file = DATABASE_DIR / "mt5_config.json"
    try:
        # Charger config existante
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception:
            config = {}

        # Mettre a jour
        for key in ['login', 'password', 'server', 'path']:
            if key in updates and updates[key]:
                config[key] = updates[key]

        # Sauvegarder
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        return {"success": True, "message": "Configuration MT5 sauvegardee"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mt5/test")
async def test_mt5_connection():
    """Teste la connexion MT5 (premier agent actif)"""
    mt5 = get_mt5(get_first_enabled_agent())
    try:
        if mt5.connect():
            account = mt5.get_account_info()
            return {
                "connected": True,
                "account": account.get("login") if account else "Unknown"
            }
        else:
            return {"connected": False, "error": "Echec connexion"}
    except Exception as e:
        return {"connected": False, "error": str(e)}


# =============================================================================
# ENDPOINTS - MULTI-COMPTE (un agent = un compte)
# =============================================================================
@app.get("/api/accounts")
async def get_all_accounts():
    """Recupere tous les comptes MT5 par agent"""
    from core.mt5_connector import get_all_mt5_accounts
    return get_all_mt5_accounts()


@app.get("/api/accounts/{agent_id}")
async def get_agent_account(agent_id: str):
    """Recupere le compte MT5 d'un agent"""
    from core.mt5_connector import get_all_mt5_accounts
    accounts = get_all_mt5_accounts()
    if agent_id not in accounts:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} non trouve")
    return accounts[agent_id]


@app.post("/api/accounts/{agent_id}")
async def update_agent_account(agent_id: str, config: Dict):
    """Met a jour le compte MT5 d'un agent"""
    from core.mt5_connector import save_mt5_account

    if agent_id not in get_agent_ids():
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} non trouve")

    if save_mt5_account(agent_id, config):
        return {"success": True, "message": f"Compte MT5 de {agent_id} mis a jour"}
    else:
        raise HTTPException(status_code=500, detail="Erreur sauvegarde config")


@app.post("/api/accounts/{agent_id}/test")
async def test_agent_account(agent_id: str):
    """Teste la connexion MT5 d'un agent"""
    from core.mt5_connector import get_mt5

    if agent_id not in get_agent_ids():
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} non trouve")

    mt5 = get_mt5(agent_id)
    try:
        if mt5.connect():
            account = mt5.get_account_info()
            return {
                "connected": True,
                "agent_id": agent_id,
                "login": account.get("login") if account else None,
                "balance": account.get("balance") if account else 0
            }
        else:
            return {"connected": False, "agent_id": agent_id, "error": "Echec connexion"}
    except Exception as e:
        return {"connected": False, "agent_id": agent_id, "error": str(e)}


@app.get("/api/accounts/status/all")
async def get_all_accounts_status():
    """Status de connexion de tous les comptes (utilise le cache pour eviter reconnexions)"""
    from core.mt5_connector import get_mt5, get_all_mt5_accounts, get_cached_account_info, is_agent_currently_connected

    accounts = get_all_mt5_accounts()
    status = {}
    first_agent = get_first_enabled_agent()
    need_reconnect_first = False

    for agent_id in get_agent_ids():
        account_cfg = accounts.get(agent_id, {})
        if account_cfg.get("enabled") and account_cfg.get("login"):
            try:
                # Utiliser le cache au lieu de forcer une reconnexion
                cached_info = get_cached_account_info(agent_id)

                if cached_info:
                    # Donnees en cache disponibles - les utiliser sans reconnexion
                    status[agent_id] = {
                        "configured": True,
                        "enabled": True,
                        "connected": True,
                        "login": account_cfg.get("login"),
                        "balance": cached_info.get("balance", 0),
                        "equity": cached_info.get("equity", 0),
                        "from_cache": True
                    }
                else:
                    # Pas de cache - connexion initiale necessaire (une seule fois)
                    mt5 = get_mt5(agent_id)
                    connected = mt5.connect()
                    account_info = mt5.get_account_info() if connected else None
                    status[agent_id] = {
                        "configured": True,
                        "enabled": True,
                        "connected": connected,
                        "login": account_cfg.get("login"),
                        "balance": account_info.get("balance") if account_info else 0,
                        "equity": account_info.get("equity") if account_info else 0,
                        "from_cache": False
                    }
                    # Marquer qu'on a du switcher de compte
                    if agent_id != first_agent:
                        need_reconnect_first = True
            except Exception as e:
                status[agent_id] = {
                    "configured": True,
                    "enabled": True,
                    "connected": False,
                    "login": account_cfg.get("login"),
                    "error": f"Erreur connexion: {e}"
                }
        else:
            status[agent_id] = {
                "configured": bool(account_cfg.get("login")),
                "enabled": account_cfg.get("enabled", False),
                "connected": False
            }

    # IMPORTANT: Toujours revenir sur le premier agent actif apres avoir peuple les caches
    # Pour eviter que trading_loop doive switcher a chaque iteration
    if need_reconnect_first:
        first_mt5 = get_mt5(first_agent)
        first_mt5.connect()

    return status


# =============================================================================
# ENDPOINTS - SESSIONS
# =============================================================================
@app.get("/api/session")
async def get_session():
    """Recupere les infos de la session actuelle"""
    from session_logger import get_session_logger
    logger = get_session_logger()
    return logger.get_status()


@app.post("/api/session/reload")
async def reload_session():
    """Recharge la session depuis session.json (utile apres modification manuelle)"""
    from session_logger import get_session_logger
    logger = get_session_logger()
    logger._load_session()
    return {"success": True, "balance_start": logger.balance_start, "session_id": logger.session_id}


@app.post("/api/session/start")
@app.get("/api/session/start")  # GET temporaire car POST bloque sous charge MT5
async def start_session():
    """Demarre une nouvelle session (non-bloquant)"""
    from session_logger import get_session_logger
    logger = get_session_logger()

    # Demarrer la session avec balance 0 (sera mise a jour en background)
    result = logger.start_session(0)

    # Recuperer la balance en background via thread separe
    def update_balance():
        try:
            aggregator = get_aggregator()
            account_data = aggregator.get_account_data()
            if account_data:
                balance = account_data.get('balance', 0)
                logger.balance_start = balance
                logger._save_session()
                print(f"[API] Session balance mise a jour: {balance} EUR")
        except Exception as e:
            print(f"[API] Erreur mise a jour balance session: {e}")

    threading.Thread(target=update_balance, daemon=True).start()
    return result


@app.post("/api/session/end")
@app.get("/api/session/end")  # GET temporaire car POST bloque sous charge MT5
async def end_session():
    """Termine la session actuelle (non-bloquant, termine en background)"""
    from session_logger import get_session_logger
    logger = get_session_logger()

    if not logger.session_id:
        return {'success': False, 'message': 'Aucune session active'}

    # Utiliser la balance de session si disponible, sinon 0
    balance = logger.balance_start or 0
    session_id = logger.session_id

    # Lancer la terminaison en background (peut prendre du temps)
    def end_in_background():
        try:
            result = logger.end_session(balance)
            print(f"[API] Session {session_id} terminee en background")
        except Exception as e:
            print(f"[API] Erreur fin session background: {e}")

    threading.Thread(target=end_in_background, daemon=True).start()

    # Retourner immediatement
    return {'success': True, 'message': f'Terminaison de la session {session_id} en cours...', 'session_id': session_id}


@app.post("/api/session/sync")
async def sync_session_with_mt5():
    """
    Synchronise les trades de la session avec l'historique MT5.
    Recupere tous les trades fermes depuis le debut de la session
    et met a jour les stats avec les VRAIS chiffres MT5.
    """
    from session_logger import get_session_logger
    logger = get_session_logger()
    return logger.sync_with_mt5_history()


@app.get("/api/session/export")
async def export_session():
    """Exporte les donnees de la session"""
    from session_logger import get_session_logger
    logger = get_session_logger()
    return logger.export_session()


@app.get("/api/session/history")
async def get_session_history(limit: int = 10):
    """Recupere l'historique des sessions"""
    from session_logger import get_session_logger
    logger = get_session_logger()
    return logger.get_session_history(limit)
@app.get("/api/session/performance")
async def get_session_performance():
    """Recupere les points de performance (equite) de la session actuelle"""
    from session_logger import get_session_logger
    logger = get_session_logger()
    return {
        "session_id": logger.session_id,
        "performance": logger.performance_history
    }


# =============================================================================
# ENDPOINTS - STRATEGIST
# =============================================================================
@app.get("/api/strategist/analyze")
async def strategist_analyze():
    """Analyse complete des performances"""
    from strategist import get_strategist
    strategist = get_strategist()
    return strategist.analyze()


@app.get("/api/strategist/insights")
async def strategist_insights():
    """Insights rapides"""
    from strategist import get_strategist
    strategist = get_strategist()
    return strategist.get_quick_insights()


@app.get("/api/strategist/logs")
async def strategist_logs(limit: int = 50):
    """Recupere les logs du Strategist"""
    from strategist import get_strategist
    strategist = get_strategist()
    return {
        'logs': strategist.get_logs(limit),
        'suggestions': strategist._load_logs()[:10] if hasattr(strategist, '_load_logs') else []
    }


@app.post("/api/strategist/execute")
async def strategist_execute():
    """Execute les suggestions critiques automatiquement"""
    from strategist import get_strategist
    strategist = get_strategist()
    result = strategist.execute_suggestions()
    return result


@app.get("/api/strategist/actions")
async def strategist_actions(limit: int = 50):
    """Recupere les actions executees par le Strategist"""
    from strategist import get_strategist
    strategist = get_strategist()
    return {
        'actions': strategist.get_executed_actions(limit)
    }


@app.get("/api/strategist/debug")
async def strategist_debug():
    """Debug: montre l'etat interne du Strategist"""
    import time
    from strategist import get_strategist
    strategist = get_strategist()
    current_time = time.time()
    inactivity_seconds = current_time - strategist._last_inactivity_reduction_time
    return {
        "trades_count": len(strategist.trades),
        "last_trades_count": strategist._last_trades_count,
        "last_inactivity_reduction_time": strategist._last_inactivity_reduction_time,
        "inactivity_seconds": int(inactivity_seconds),
        "inactivity_minutes": round(inactivity_seconds / 60, 1),
        "threshold_seconds": 900,
        "would_trigger": inactivity_seconds >= 900,
        "last_optimization_time": strategist._last_optimization_time
    }


@app.post("/api/strategist/force-inactivity-check")
async def force_inactivity_check():
    """Force l'execution du check d'inactivite (bypass threshold)"""
    from strategist import get_strategist
    strategist = get_strategist()
    # Forcer le timer a une valeur ancienne pour bypasser le seuil
    old_time = strategist._last_inactivity_reduction_time
    strategist._last_inactivity_reduction_time = 0  # Simule 50+ ans d'inactivite
    result = strategist._check_inactivity_and_correct()
    # Restaurer si aucune action n'a ete executee
    if result.get('executed_count', 0) == 0:
        strategist._last_inactivity_reduction_time = old_time
    return {
        "result": result,
        "message": f"Check execute, {result.get('executed_count', 0)} actions"
    }


# =============================================================================
# ENDPOINTS - API KEYS
# =============================================================================
@app.get("/api/keys")
async def get_api_keys():
    """Recupere les cles API enregistrees"""
    keys_file = DATABASE_DIR / "api_keys.json"
    try:
        if keys_file.exists():
            with open(keys_file, 'r') as f:
                return json.load(f)
        else:
            return {"keys": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/keys")
async def save_api_keys(data: Dict):
    """Sauvegarde les cles API"""
    keys_file = DATABASE_DIR / "api_keys.json"
    try:
        with open(keys_file, 'w') as f:
            json.dump(data, f, indent=4)
        return {"success": True, "message": "Cles API sauvegardees"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/keys/selections")
async def get_api_selections():
    """Recupere les selections de cles API par agent"""
    selections_file = DATABASE_DIR / "api_selections.json"
    try:
        if selections_file.exists():
            with open(selections_file, 'r') as f:
                return json.load(f)
        else:
            return {"selections": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/keys/selections")
async def save_api_selections(data: Dict):
    """Sauvegarde les selections de cles API par agent"""
    selections_file = DATABASE_DIR / "api_selections.json"
    try:
        with open(selections_file, 'w') as f:
            json.dump(data, f, indent=4)
        return {"success": True, "message": "Selections sauvegardees"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("G12 - BTCUSD Trading Bot")
    print("=" * 60)
    print(f"API: http://localhost:{API_CONFIG['port']}")
    print(f"Dashboard: http://localhost:{API_CONFIG['port']}/")
    print("=" * 60)

    uvicorn.run(
        app,
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG.get("reload", False),
        limit_concurrency=100,  # Limite de requetes concurrentes
        timeout_keep_alive=5    # Timeout keepalive reduit
    )
