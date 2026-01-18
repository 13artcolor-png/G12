# -*- coding: utf-8 -*-
"""
G12 - API FastAPI
Backend API pour le dashboard
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import threading
import time
import json
import secrets
import base64
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

        # ***********************************************************************
        # DEMARRAGE MANUEL DES BOUCLES DE TRADING
        # ***********************************************************************
        # Les boucles NE SONT PLUS lancées automatiquement au démarrage
        # L'utilisateur doit cliquer sur "Démarrer Trading" dans le dashboard
        # Cela permet un contrôle total sur le moment de lancement du trading
        # ***********************************************************************
        
        # Initialiser les loops SANS les démarrer
        trading_loop = get_trading_loop()
        closer_loop = get_closer_loop()
        
        # Mettre les loops en mode PAUSE par défaut
        trading_loop.running = False
        closer_loop.running = False

        # SYNCHRONISATION INITIALE MT5 meme en PAUSE
        # Cela permet d'afficher les trades fermes dans les stats meme sans demarrer le trading
        print("[Lifespan] Synchronisation initiale MT5...")
        try:
            session_logger = get_session_logger()
            sync_result = session_logger.sync_with_mt5_history()
            if sync_result.get('success'):
                total_trades = sync_result.get('total_trades', 0)
                total_pnl = sync_result.get('total_synced_pnl', 0)
                print(f"[Lifespan] Sync MT5 OK: {total_trades} trades, P&L={total_pnl:+.2f} EUR")
            else:
                print(f"[Lifespan] Sync MT5: {sync_result.get('message', 'pas de session active')}")
        except Exception as e:
            print(f"[Lifespan] Erreur sync MT5: {e}")

        print("[Lifespan] Services initialises - Trading en PAUSE")
        print("[Lifespan] Cliquez sur 'Demarrer Trading' dans le dashboard pour lancer")

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

# =============================================================================
# SECURITE - HTTP Basic Authentication
# =============================================================================
# Charger les credentials depuis .env (avec fallback)
import os
AUTH_USERNAME = os.getenv("API_AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("API_AUTH_PASSWORD", "G12_secure_2026")

# =============================================================================
# CORS - Restreint au localhost uniquement
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8012", "http://127.0.0.1:8012"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MIDDLEWARE - Authentification globale pour tous les endpoints /api/*
# =============================================================================
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    Middleware d'authentification HTTP Basic pour tous les endpoints /api/*
    Exclut la page d'accueil "/" pour permettre le chargement du dashboard
    """
    # Laisser passer la page d'accueil et les fichiers statiques
    if request.url.path == "/" or not request.url.path.startswith("/api/"):
        return await call_next(request)

    # Vérifier l'authentification HTTP Basic pour tous les endpoints /api/*
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Basic "):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Authentification requise"},
            headers={"WWW-Authenticate": "Basic"},
        )

    try:
        # Decoder les credentials
        encoded_credentials = auth_header.split(" ")[1]
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        username, password = decoded_credentials.split(":", 1)

        # Vérifier les credentials (constant-time comparison)
        correct_username = secrets.compare_digest(username, AUTH_USERNAME)
        correct_password = secrets.compare_digest(password, AUTH_PASSWORD)

        if not (correct_username and correct_password):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Identifiants incorrects"},
                headers={"WWW-Authenticate": "Basic"},
            )

    except Exception:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Format d'authentification invalide"},
            headers={"WWW-Authenticate": "Basic"},
        )

    # Authentification réussie - continuer
    return await call_next(request)


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
        # Recuperer les composants avec gestion d'erreur individuelle
        aggregator = get_aggregator()
        context = aggregator.get_full_context() or {}
        account_data = context.get("account") or {}

        # MT5 connection status
        mt5_connected = False
        try:
            mt5 = get_mt5(get_first_enabled_agent())
            mt5_connected = mt5.connected if mt5 else False
        except Exception as e:
            print(f"[API] Erreur MT5 status: {e}")

        # Session logger
        g11_session = {}
        try:
            from session_logger import get_session_logger
            session_logger = get_session_logger()
            g11_session = session_logger.get_status()
        except Exception as e:
            print(f"[API] Erreur session logger: {e}")

        # Trading loop status
        agents_status = {}
        trading_status = {}
        try:
            trading_loop = get_trading_loop()
            trading_status = trading_loop.get_status()
            agents_status = trading_status.get("agents", {})
        except Exception as e:
            print(f"[API] Erreur trading loop: {e}")

        # Closer loop status
        closer_status = {}
        try:
            closer_loop = get_closer_loop()
            closer_status = closer_loop.get_status()
        except Exception as e:
            print(f"[API] Erreur closer loop: {e}")

        # Risk manager
        risk_status = {}
        try:
            risk = get_risk_manager()
            risk_status = risk.get_status()
        except Exception as e:
            print(f"[API] Erreur risk manager: {e}")

        # Session stats (depuis stats_fibo*.json de la session actuelle)
        session_stats = {}
        recent_decisions = []
        try:
            from session_logger import get_session_logger
            session_logger_inst = get_session_logger()
            session_stats = session_logger_inst.get_session_stats()

            # Decisions de la session (toutes)
            logger = get_logger()
            recent_decisions = logger.get_recent_decisions(limit=5000)  # Augmenté pour toute la session
        except Exception as e:
            print(f"[API] Erreur session stats: {e}")

        return {
            "timestamp": datetime.now().isoformat(),
            "mt5": {
                "connected": mt5_connected,
                "account": None
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
            "agents": agents_status,
            "risk": risk_status,
            "loops": {
                "trading": trading_status,
                "closer": closer_status
            },
            "stats": session_stats,
            "decisions": recent_decisions
        }
    except Exception as e:
        import traceback
        print(f"[API] ERREUR CRITIQUE build status: {e}")
        traceback.print_exc()
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
    """Status complet du systeme - utilise directement le context de l'aggregator"""
    global _status_cache

    now = time.time()
    cache_age = now - (_status_cache["timestamp"] or 0)

    # Si le cache est recent, le retourner immediatement
    if _status_cache["data"] and cache_age < _status_cache_max_age:
        return _status_cache["data"]

    # Lancer update en arriere-plan pour le prochain appel
    if not _status_cache["updating"]:
        threading.Thread(target=_update_status_cache_background, daemon=True).start()

    # Retourner le cache existant s'il y a
    if _status_cache["data"]:
        return _status_cache["data"]

    # PAS DE CACHE - Construire une reponse directement avec le context
    try:
        aggregator = get_aggregator()
        context = aggregator.get_full_context() or {}
        account_data = context.get("account") or {}

        # Session G12
        g11_session = {}
        try:
            from session_logger import get_session_logger
            session_logger = get_session_logger()
            g11_session = session_logger.get_status()
        except Exception:
            pass

        return {
            "timestamp": datetime.now().isoformat(),
            "mt5": {"connected": bool(account_data), "account": None},
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
            "agents": {},
            "risk": {},
            "loops": {"trading": {}, "closer": {}},
            "stats": {},
            "decisions": []
        }
    except Exception as e:
        print(f"[API] Erreur fallback status: {e}")

    # Fallback ultime
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


@app.get("/api/agents/reload")
async def reload_agents_config():
    """Force le rechargement immediat de la config de tous les agents"""
    trading_loop = get_trading_loop()
    reloaded = []
    
    for agent_id, agent in trading_loop.agents.items():
        # Forcer le rechargement en resetant le timestamp
        agent._last_config_load = 0
        agent.reload_config()
        reloaded.append({
            "agent_id": agent_id,
            "enabled": agent.enabled,
            "name": agent.name
        })
    
    print(f"[API] Config rechargee pour {len(reloaded)} agents")
    return {"success": True, "agents": reloaded}


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

    # IMPORTANT: Reinitialiser le timer d'inactivite du Strategist au demarrage du trading
    # Ceci evite les fausses alertes basees sur une session precedente
    import time
    from strategist import get_strategist
    strategist = get_strategist()
    strategist._last_inactivity_reduction_time = time.time()
    strategist._save_inactivity_state()
    print(f"[API] Timer d'inactivite reinitialise au demarrage du trading")

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
@app.get("/api/trading/stop")
async def stop_trading():
    """Arrete les boucles de trading (non-bloquant, GET ou POST)"""
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


@app.post("/api/position/close/{ticket}")
async def close_position_by_ticket(ticket: int):
    """Ferme une position specifique par son ticket"""
    try:
        from core.mt5_connector import get_mt5
        from core.trading_loop import get_trading_loop

        closer_loop = get_closer_loop()
        trading_loop = get_trading_loop()

        # Chercher la position sur tous les comptes agents
        for agent_id in ["fibo1", "fibo2", "fibo3"]:
            try:
                agent_mt5 = get_mt5(agent_id)
                if agent_mt5.connect():
                    positions = agent_mt5.get_positions()
                    agent = trading_loop.agents.get(agent_id)

                    for position in positions:
                        if position.get("ticket") == ticket:
                            # Position trouvée - la fermer
                            closer_loop._close_position(position, "MANUAL_CLOSE", trading_loop, agent)
                            return {
                                "success": True,
                                "message": f"Position #{ticket} fermée manuellement",
                                "ticket": ticket,
                                "agent": agent_id
                            }
            except Exception as e:
                print(f"[API] Erreur fermeture position {ticket} sur {agent_id}: {e}")

        return {
            "success": False,
            "message": f"Position #{ticket} non trouvée",
            "ticket": ticket
        }

    except Exception as e:
        print(f"[API] Erreur fermeture position: {e}")
        return {
            "success": False,
            "message": str(e),
            "ticket": ticket
        }


@app.post("/api/restart")
async def restart_backend():
    """Redémarre le backend en exécutant le script restart_backend.bat"""
    import subprocess
    import os

    try:
        # Chemin du script de redémarrage
        script_path = os.path.join(os.path.dirname(__file__), "..", "restart_backend.bat")

        # Lancer le script en arrière-plan (détaché)
        subprocess.Popen(
            [script_path],
            shell=True,
            stdin=None,
            stdout=None,
            stderr=None,
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )

        print("[API] Script de redémarrage lancé")
        return {"success": True, "message": "Redémarrage du backend en cours..."}
    except Exception as e:
        print(f"[API] Erreur redémarrage: {e}")
        return {"success": False, "message": f"Erreur: {str(e)}"}


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
    """Historique des trades - depuis session.json (rapide, synchronise avec MT5)

    OPTIMISATION: Utilise les trades deja synchronises dans session.json
    au lieu d'interroger MT5 a chaque requete (10+ secondes de latence)
    """
    from session_logger import get_session_logger

    logger = get_session_logger()

    # Recuperer tous les trades de la session
    all_trades = []
    for trade in logger.trades:
        trade_agent = trade.get('agent', '')
        all_trades.append({
            'agent_id': trade_agent,
            'ticket': trade.get('ticket', 0),
            'direction': trade.get('direction', 'UNKNOWN'),
            'volume': trade.get('volume', 0),
            'entry_price': trade.get('entry_price', 0),
            'exit_price': trade.get('exit_price', 0),
            'profit': trade.get('profit', 0),
            'profit_eur': trade.get('profit', 0),
            'close_reason': trade.get('close_reason', ''),
            'timestamp': trade.get('timestamp', '')
        })

    # Filtrer par agent si necessaire
    if agent and agent != 'all':
        all_trades = [t for t in all_trades if t['agent_id'] == agent]

    # Trier par timestamp descending et limiter
    all_trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    all_trades = all_trades[:limit]

    return {"trades": all_trades}


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


# =============================================================================
# ENDPOINTS - STRATEGIST CONFIG
# =============================================================================
@app.get("/api/config/strategist")
async def get_strategist_config():
    """Recupere la configuration du Strategist"""
    config_file = DATABASE_DIR / "strategist_runtime_config.json"
    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Config par defaut
            return {"use_session_analysis": True}
    except Exception as e:
        return {"use_session_analysis": True}


@app.post("/api/config/strategist")
async def update_strategist_config(updates: Dict):
    """Met a jour la configuration du Strategist"""
    config_file = DATABASE_DIR / "strategist_runtime_config.json"
    try:
        # Charger config existante
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = {"use_session_analysis": True}

        # Mettre a jour
        for key, value in updates.items():
            config[key] = value

        # Sauvegarder
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"[API] Config Strategist mise a jour: {config}")
        return {"success": True, "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/accounts")
async def get_mt5_accounts():
    """Retourne les configurations MT5 de tous les agents (pour remplir les champs du frontend)"""
    from core.mt5_connector import get_all_mt5_accounts

    accounts = get_all_mt5_accounts()

    # Retourner les configs AVEC les vraies valeurs (sauf password masque)
    return accounts


@app.post("/api/accounts/{agent_id}")
async def save_mt5_account_config(agent_id: str, config: Dict):
    """Sauvegarde la config MT5 d'un agent"""
    from core.mt5_connector import save_mt5_account, get_all_mt5_accounts

    if agent_id not in ["fibo1", "fibo2", "fibo3"]:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} invalide")

    # IMPORTANT: Si password est vide ou "***", ne PAS l'ecraser
    # Recuperer le password existant
    existing_accounts = get_all_mt5_accounts()
    existing_password = existing_accounts.get(agent_id, {}).get("password", "")

    # Si le password recu est vide ou masque, garder l'existant
    received_password = config.get("password", "")
    if not received_password or received_password == "***":
        config["password"] = existing_password
        print(f"[API] {agent_id}: Password non fourni, conservation du password existant")

    # Sauvegarder
    success = save_mt5_account(agent_id, config)

    if success:
        return {"success": True, "message": f"Config {agent_id} sauvegardee"}
    else:
        raise HTTPException(status_code=500, detail="Erreur sauvegarde config")


@app.post("/api/accounts/{agent_id}/test")
async def test_mt5_connection(agent_id: str):
    """Teste la connexion MT5 d'un agent"""
    from core.mt5_connector import get_mt5

    if agent_id not in ["fibo1", "fibo2", "fibo3"]:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} invalide")

    try:
        mt5 = get_mt5(agent_id)
        connected = mt5.connect()

        if connected:
            account_info = mt5.get_account_info()
            if account_info:
                return {
                    "connected": True,
                    "login": account_info.get("login"),
                    "balance": account_info.get("balance", 0),
                    "server": mt5.server
                }
            else:
                return {"connected": False, "error": "Connexion OK mais impossible de recuperer les infos compte"}
        else:
            return {"connected": False, "error": "Echec connexion MT5 - Verifier login/password/server"}

    except Exception as e:
        return {"connected": False, "error": f"Exception: {str(e)}"}


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


@app.post("/api/session/start")
@app.get("/api/session/start")  # GET temporaire car POST bloque sous charge MT5
async def start_session():
    """Demarre une nouvelle session avec la vraie balance MT5"""
    from session_logger import get_session_logger
    logger = get_session_logger()

    # IMPORTANT: Recuperer la balance AVANT de demarrer la session
    # Ne PAS demarrer avec balance 0 car cela donne des stats incorrectes
    try:
        aggregator = get_aggregator()
        account_data = aggregator.get_account_data()

        if not account_data:
            # Impossible de recuperer les donnees compte - probablement echec connexion MT5
            return {
                'success': False,
                'error': 'Impossible de se connecter aux comptes MT5. Verifiez les mots de passe dans la section Comptes MT5.'
            }

        balance = account_data.get('balance', 0)
        connected_accounts = len([k for k, v in account_data.get('accounts', {}).items() if v])

        if balance == 0:
            return {
                'success': False,
                'error': f'Balance totale = 0 EUR. Seulement {connected_accounts}/3 comptes connectes. Verifiez les configurations MT5.'
            }

        print(f"[API] Demarrage session avec balance: {balance:.2f} EUR ({connected_accounts}/3 comptes)")

        # Demarrer la session avec la vraie balance
        result = logger.start_session(balance)

        return result

    except Exception as e:
        print(f"[API] Erreur demarrage session: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'Erreur: {str(e)}'
        }


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


@app.get("/api/session/export")
async def export_session():
    """Exporte les donnees de la session"""
    from session_logger import get_session_logger
    logger = get_session_logger()
    return logger.export_session()


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
    try:
        from strategist import get_strategist
        strategist = get_strategist()
        result = strategist.analyze()
        return result
    except Exception as e:
        logger.error(f"[Strategist] Erreur lors de l'analyse: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'suggestions': [],
            'patterns': [],
            'by_agent': {},
            'by_session': {},
            'global': None
        }


@app.get("/api/strategist/insights")
async def strategist_insights():
    """Insights rapides"""
    try:
        from strategist import get_strategist
        strategist = get_strategist()
        result = strategist.get_quick_insights()
        return result
    except Exception as e:
        logger.error(f"[Strategist] Erreur lors de get_quick_insights: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'summary': 'Erreur lors de la recuperation des insights',
            'suggestions': []
        }


@app.get("/api/strategist/logs")
async def strategist_logs(limit: int = 50):
    """Recupere les logs du Strategist"""
    try:
        from strategist import get_strategist
        strategist = get_strategist()

        # Charger les logs une seule fois (pas deux fois comme avant)
        logs = strategist.get_logs(limit) if hasattr(strategist, 'get_logs') else []

        return {
            'logs': logs,
            'total_logs': len(logs)
        }
    except Exception as e:
        logger.error(f"[Strategist] Erreur lors de get_logs: {e}", exc_info=True)
        return {
            'logs': [],
            'total_logs': 0,
            'error': str(e)
        }


@app.post("/api/strategist/execute")
async def strategist_execute():
    """Execute les suggestions critiques automatiquement"""
    try:
        from strategist import get_strategist
        strategist = get_strategist()
        result = strategist.execute_suggestions()
        return result
    except Exception as e:
        logger.error(f"[Strategist] Erreur lors de execute_suggestions: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'executed_count': 0,
            'actions': []
        }


@app.get("/api/strategist/debug")
async def strategist_debug():
    """Debug complet: montre l'etat interne du Strategist avec profit factor et actions"""
    try:
        import time
        from datetime import datetime
        from strategist import get_strategist
        from data.aggregator import get_aggregator

        strategist = get_strategist()
        aggregator = get_aggregator()
        current_time = time.time()
        inactivity_seconds = current_time - strategist._last_inactivity_reduction_time
        optim_seconds = current_time - strategist._last_optimization_time

        # Charger les stats globales
        global_stats = strategist._analyze_global()

        # Recuperer l'analyse complete des positions (G12 vs MT5)
        positions_analysis = strategist._analyze_open_positions()

        # Reformater pour l'API
        open_positions = []
        try:
            account_data = aggregator.get_account_data()
            if account_data and account_data.get('positions'):
                for pos in account_data['positions']:
                    open_positions.append({
                        'ticket': pos.get('ticket'),
                        'agent': pos.get('_agent_id', 'unknown'),
                        'direction': 'BUY' if pos.get('type', 0) == 0 else 'SELL',
                        'symbol': pos.get('symbol', 'BTCUSD'),
                        'entry_price': round(pos.get('price_open', 0), 2),
                        'current_price': round(pos.get('price_current', 0), 2),
                        'volume': pos.get('volume', 0),
                        'sl': round(pos.get('sl', 0), 2),
                        'tp': round(pos.get('tp', 0), 2),
                        'floating_pnl': round(pos.get('profit', 0), 2)
                    })
        except Exception as e:
            print(f"[Strategist Debug] Erreur recuperation positions: {e}")

        total_floating_pnl = positions_analysis.get('total_floating_pnl', 0)
        positions_inconsistencies = positions_analysis.get('inconsistencies', [])
        positions_alerts = positions_analysis.get('alerts', [])

        # Charger les actions recentes depuis action_history.json
        recent_actions = []
        try:
            import json
            from pathlib import Path
            action_file = Path(__file__).parent / "database" / "action_history.json"
            if action_file.exists():
                with open(action_file, 'r') as f:
                    data = json.load(f)
                    recent_actions = data.get('actions', [])[-10:]  # 10 dernieres
        except Exception:
            pass

        # Calculer profit_factor par agent
        agents_pf = {}
        for agent_id in ['fibo1', 'fibo2', 'fibo3']:
            agent_trades = [t for t in strategist.trades if t.get('agent_id') == agent_id or t.get('agent', '') == agent_id]
            wins = [t for t in agent_trades if t.get('profit', 0) > 0]
            losses = [t for t in agent_trades if t.get('profit', 0) < 0]
            total_win = sum(t.get('profit', 0) for t in wins)
            total_loss = abs(sum(t.get('profit', 0) for t in losses))
            pf = round(total_win / total_loss, 2) if total_loss > 0 else 999
            win_rate = round(len(wins) / len(agent_trades) * 100, 1) if agent_trades else 0
            agents_pf[agent_id] = {
                'trades': len(agent_trades),
                'profit_factor': pf,
                'win_rate': win_rate,
                'total_pnl': round(sum(t.get('profit', 0) for t in agent_trades), 2)
            }

        # Determiner prochaine action potentielle
        next_action = "Aucune correction necessaire"
        if global_stats.get('profit_factor', 999) < 1.0 and global_stats.get('total_trades', 0) >= 10:
            next_action = f"AJUSTER_TPSL (profit_factor={global_stats.get('profit_factor')} < 1.0)"
        elif any(agents_pf[a]['win_rate'] < 30 and agents_pf[a]['trades'] >= 5 for a in agents_pf):
            losing_agents = [a for a in agents_pf if agents_pf[a]['win_rate'] < 30 and agents_pf[a]['trades'] >= 5]
            next_action = f"AUGMENTER_SEUILS pour {', '.join(losing_agents)}"
        elif inactivity_seconds >= 900:
            next_action = "CHECK_INACTIVITE (15min+ sans action)"

        return {
            "status": "actif" if strategist._last_optimization_time > 0 else "en_attente",
            "trades_count": len(strategist.trades),
            "last_optimization": {
                "timestamp": datetime.fromtimestamp(strategist._last_optimization_time).isoformat() if strategist._last_optimization_time > 0 else None,
                "seconds_ago": int(optim_seconds),
                "minutes_ago": round(optim_seconds / 60, 1)
            },
            "inactivity": {
                "seconds": int(inactivity_seconds),
                "minutes": round(inactivity_seconds / 60, 1),
                "threshold_minutes": 15,
                "would_trigger": inactivity_seconds >= 900
            },
            "global_stats": {
                "profit_factor": global_stats.get('profit_factor', 0),
                "win_rate": global_stats.get('win_rate', 0),
                "total_trades": global_stats.get('total_trades', 0),
                "total_pnl": global_stats.get('total_profit', 0)
            },
            "agents": agents_pf,
            "next_potential_action": next_action,
            "recent_actions": recent_actions,
            "thresholds": {
                "profit_factor_min": 1.0,
                "win_rate_min": 30,
                "min_trades_for_correction": 10
            },
            "open_positions": open_positions,
            "open_positions_count": len(open_positions),
            "total_floating_pnl": total_floating_pnl,
            "positions_inconsistencies": positions_inconsistencies,
            "positions_alerts": positions_alerts,
            "positions_g12_count": positions_analysis.get('positions_count', {}).get('g12', 0),
            "positions_mt5_count": positions_analysis.get('positions_count', {}).get('mt5', 0)
        }
    except Exception as e:
        logger.error(f"[Strategist] Erreur dans /api/strategist/debug: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "trades_count": 0,
            "global_stats": {"profit_factor": 0, "win_rate": 0, "total_trades": 0, "total_pnl": 0},
            "agents": {},
            "next_potential_action": "Erreur - Strategist non disponible",
            "recent_actions": [],
            "open_positions": []
        }


@app.post("/api/strategist/test")
async def test_strategist():
    """Test manuel du Strategist - execute une analyse complete"""
    try:
        from strategist import get_strategist
        from session_logger import get_session_logger
        strategist = get_strategist()

        # 1. Recuperer les stats actuelles
        logger = get_session_logger()
        stats = logger.sync_with_mt5_history()

        # 2. Calculer les stats globales
        total_profit = 0
        total_loss = 0
        total_trades = 0

        for agent_id in ['fibo1', 'fibo2', 'fibo3']:
            agent_stats = stats.get(agent_id, {})
            pnl = agent_stats.get('session_pnl', 0)
            trades = agent_stats.get('session_trades', 0)
            total_trades += trades
            if pnl > 0:
                total_profit += pnl
            else:
                total_loss += abs(pnl)

        # 3. Calculer le profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else (999 if total_profit > 0 else 0)

        # 4. Determiner si une correction serait declenchee
        would_correct = profit_factor < 1.0 and total_trades >= 10

        # 5. Verifier l'inactivite
        import time
        current_time = time.time()
        last_check = getattr(strategist, '_last_inactivity_reduction_time', 0)
        inactivity_minutes = (current_time - last_check) / 60 if last_check > 0 else 0

        return {
            "success": True,
            "message": f"Test Strategist: {total_trades} trades analyses, PF={profit_factor:.2f}",
            "details": {
                "total_trades": total_trades,
                "total_profit": round(total_profit, 2),
                "total_loss": round(total_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "would_trigger_correction": would_correct,
                "inactivity_minutes": round(inactivity_minutes, 1)
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Erreur lors du test: {str(e)}"
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
# ENDPOINTS - CHARTS EXPORT (VISION AI)
# =============================================================================
@app.get("/api/charts/generate")
async def generate_chart(
    timeframe: str = "15m",
    candle_count: int = 100,
    agent_id: str = "fibo1",
    format: str = "base64"
):
    """
    Genere un graphique BTCUSD

    Args:
        timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d
        candle_count: Nombre de bougies (10-500)
        agent_id: Agent MT5 (fibo1, fibo2, fibo3)
        format: base64 (JSON) ou file (PNG direct)

    Returns:
        JSON avec base64 ou fichier PNG
    """
    try:
        from core.chart_generator import ChartGenerator

        # Validation
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if timeframe not in valid_timeframes:
            raise HTTPException(400, f"Timeframe invalide. Valeurs: {', '.join(valid_timeframes)}")

        if candle_count < 10 or candle_count > 500:
            raise HTTPException(400, "candle_count doit etre entre 10 et 500")

        if agent_id not in ["fibo1", "fibo2", "fibo3"]:
            raise HTTPException(400, "agent_id invalide (fibo1, fibo2 ou fibo3)")

        # Generer graphique
        generator = ChartGenerator(agent_id=agent_id)

        if format == "base64":
            # Retour base64
            img_base64 = generator.generate_candlestick_chart(
                timeframe=timeframe,
                candle_count=candle_count,
                return_base64=True
            )

            if not img_base64:
                raise HTTPException(500, "Erreur generation graphique")

            return {
                "success": True,
                "timeframe": timeframe,
                "candle_count": candle_count,
                "format": "base64",
                "image": img_base64,
                "size_kb": round(len(img_base64) / 1024, 1)
            }

        elif format == "file":
            # Retour fichier PNG
            filepath = generator.generate_candlestick_chart(
                timeframe=timeframe,
                candle_count=candle_count,
                return_base64=False
            )

            if not filepath:
                raise HTTPException(500, "Erreur generation graphique")

            # Lire et retourner PNG
            with open(filepath, 'rb') as f:
                img_data = f.read()

            return Response(
                content=img_data,
                media_type="image/png",
                headers={
                    "Content-Disposition": f"inline; filename=BTCUSD_{timeframe}.png"
                }
            )

        else:
            raise HTTPException(400, "format doit etre 'base64' ou 'file'")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Erreur /api/charts/generate: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Erreur serveur: {str(e)}")


@app.get("/api/charts/multi")
async def generate_multi_charts(
    timeframes: str = "1m,15m,1h",
    candle_count: int = 100,
    agent_id: str = "fibo1"
):
    """
    Genere plusieurs graphiques en une requete

    Args:
        timeframes: Timeframes separes par virgule (ex: "1m,5m,15m")
        candle_count: Nombre de bougies par graphique
        agent_id: Agent MT5 (fibo1, fibo2, fibo3)

    Returns:
        JSON avec dict {timeframe: base64}
    """
    try:
        from core.chart_generator import ChartGenerator

        # Parser timeframes
        tf_list = [tf.strip() for tf in timeframes.split(",") if tf.strip()]

        if len(tf_list) == 0:
            raise HTTPException(400, "Aucun timeframe specifie")

        if len(tf_list) > 5:
            raise HTTPException(400, "Maximum 5 timeframes par requete")

        # Generer
        generator = ChartGenerator(agent_id=agent_id)
        charts = generator.generate_multi_timeframe_charts(
            timeframes=tf_list,
            candle_count=candle_count
        )

        return {
            "success": True,
            "timeframes": list(charts.keys()),
            "charts": charts,
            "total_size_kb": round(sum(len(img)/1024 for img in charts.values()), 1)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Erreur /api/charts/multi: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Erreur serveur: {str(e)}")


# =============================================================================
# PIPELINE - Tracabilite complete du processus de decision
# =============================================================================

@app.get("/api/pipeline/latest")
def get_latest_pipeline(agent_id: Optional[str] = None):
    """
    Recupere le dernier pipeline enregistre

    Args:
        agent_id: Filtrer par agent (optionnel)

    Returns:
        JSON du dernier pipeline ou None
    """
    try:
        from utils.pipeline_logger import get_latest_pipelines

        pipelines = get_latest_pipelines(agent_id=agent_id, limit=1)

        if not pipelines:
            return {"success": True, "pipeline": None}

        return {"success": True, "pipeline": pipelines[0]}

    except Exception as e:
        print(f"[API] Erreur /api/pipeline/latest: {e}")
        raise HTTPException(500, f"Erreur serveur: {str(e)}")


@app.get("/api/pipeline/history")
def get_pipeline_history(agent_id: Optional[str] = None, limit: int = 10):
    """
    Recupere l'historique des pipelines

    Args:
        agent_id: Filtrer par agent (optionnel)
        limit: Nombre max de resultats (defaut: 10, max: 50)

    Returns:
        JSON avec liste des pipelines
    """
    try:
        from utils.pipeline_logger import get_latest_pipelines

        # Limiter a 50 max
        limit = min(limit, 50)

        pipelines = get_latest_pipelines(agent_id=agent_id, limit=limit)

        return {
            "success": True,
            "count": len(pipelines),
            "pipelines": pipelines
        }

    except Exception as e:
        print(f"[API] Erreur /api/pipeline/history: {e}")
        raise HTTPException(500, f"Erreur serveur: {str(e)}")



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
