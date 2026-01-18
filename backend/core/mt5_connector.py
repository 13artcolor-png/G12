# -*- coding: utf-8 -*-
"""
G12 - Connecteur MetaTrader 5
Gere la connexion et les operations MT5 pour BTCUSD
Support multi-compte (un compte par agent)
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time
import threading
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append('..')
from config import MT5_CONFIG, MT5_ACCOUNTS, SYMBOL, SPREAD_CONFIG, DATABASE_DIR

# Lock global pour serialiser tous les acces MT5 (evite le chaos des switches de terminal)
# Utilise RLock pour permettre les appels reentrants (ensure_connected -> connect)
_mt5_lock = threading.RLock()

# Timeout pour eviter les blocages infinis (en secondes)
MT5_LOCK_TIMEOUT = 10  # 10 secondes max pour acquerir le verrou
MT5_OPERATION_TIMEOUT = 5  # 5 secondes max pour une operation MT5

# ThreadPoolExecutor pour operations MT5 avec timeout
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
_mt5_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MT5Op")


def mt5_with_timeout(func, timeout: float = MT5_OPERATION_TIMEOUT, default=None):
    """Execute une fonction MT5 avec timeout. Retourne default si timeout."""
    try:
        future = _mt5_executor.submit(func)
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        print(f"[MT5] TIMEOUT: Operation {func.__name__ if hasattr(func, '__name__') else 'MT5'} apres {timeout}s")
        return default
    except Exception as e:
        print(f"[MT5] Erreur operation: {e}")
        return default


class MT5LockContext:
    """Context manager pour le verrou MT5 avec timeout"""
    def __init__(self, timeout: float = MT5_LOCK_TIMEOUT):
        self.timeout = timeout
        self.acquired = False

    def __enter__(self):
        self.acquired = _mt5_lock.acquire(timeout=self.timeout)
        if not self.acquired:
            print(f"[MT5] WARN: Impossible d'acquerir le verrou apres {self.timeout}s")
        return self.acquired

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            _mt5_lock.release()
        return False  # Ne pas supprimer les exceptions


def load_spread_runtime_config() -> dict:
    """Charge la config spread depuis le fichier runtime (prioritaire sur config.py)"""
    try:
        config_file = DATABASE_DIR / "spread_runtime_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[MT5] Erreur chargement spread_runtime_config: {e}")
    return SPREAD_CONFIG


def to_python(val: Any) -> Any:
    """Convertit les types numpy en types Python natifs"""
    if isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, dict):
        return {k: to_python(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_python(v) for v in val]
    return val


class MT5Connector:
    """Connecteur MT5 pour G12 - BTCUSD uniquement"""

    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id
        self.connected = False
        self.current_login = None
        self.symbol = SYMBOL

        # Charger la config depuis le fichier ou utiliser la config par defaut
        self._load_config()

    def _load_config(self):
        """Charge la configuration MT5 pour l'agent"""
        # Essayer de charger depuis le fichier de config runtime
        config_file = DATABASE_DIR / "mt5_accounts_runtime.json"
        runtime_config = {}

        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    runtime_config = json.load(f)
        except Exception:
            pass

        # Si agent specifie, utiliser sa config
        if self.agent_id and self.agent_id in runtime_config:
            cfg = runtime_config[self.agent_id]
        elif self.agent_id and self.agent_id in MT5_ACCOUNTS:
            cfg = MT5_ACCOUNTS[self.agent_id]
        else:
            cfg = MT5_CONFIG

        self.login = cfg.get("login", MT5_CONFIG["login"])
        self.password = cfg.get("password", MT5_CONFIG["password"])
        self.server = cfg.get("server", MT5_CONFIG["server"])
        self.path = cfg.get("path", MT5_CONFIG["path"])
        self.account_enabled = cfg.get("enabled", True)

    def connect(self) -> bool:
        """Connexion a MT5"""
        global _current_global_login, _current_global_path

        # Recharger la config pour avoir le bon path
        self._load_config()

        # Verifier si le compte est configure
        if not self.login or self.login == 0:
            agent_str = f" pour {self.agent_id}" if self.agent_id else ""
            print(f"[MT5] Compte non configure{agent_str}")
            return False

        # Lock global avec timeout pour eviter les blocages infinis
        with MT5LockContext() as lock_acquired:
            if not lock_acquired:
                print(f"[MT5] Timeout verrou pour connexion {self.agent_id}")
                return False
            try:
                # MT5 est GLOBAL - verifier si on est deja connecte au BON compte ET au BON terminal
                if _current_global_login == self.login and _current_global_path == self.path:
                    # Verifier que MT5 repond toujours (avec timeout)
                    info = mt5_with_timeout(mt5.account_info, timeout=5, default=None)
                    if info is not None and info.login == self.login:
                        self.connected = True
                        self.current_login = self.login
                        return True

                # Si on change de terminal, il faut shutdown d'abord
                if _current_global_path and _current_global_path != self.path:
                    print(f"[MT5] Switch terminal: {_current_global_path} -> {self.path}")
                    mt5_with_timeout(mt5.shutdown, timeout=3, default=None)
                    time.sleep(1.0)  # Attendre que le shutdown soit complet

                # Initialiser MT5 avec le bon terminal ET credentials (requis pour certains terminaux)
                init_result = mt5_with_timeout(
                    lambda: mt5.initialize(
                        path=self.path,
                        login=self.login,
                        password=self.password,
                        server=self.server
                    ),
                    timeout=15,
                    default=False
                )
                if not init_result:
                    err = mt5.last_error()
                    print(f"[MT5] Erreur/Timeout initialisation {self.agent_id} ({self.path}): {err}")
                    return False

                # Attendre la synchronisation des positions apres connexion
                time.sleep(1.0)

                # Mettre a jour l'etat global ET l'etat de l'instance
                _current_global_login = self.login
                _current_global_path = self.path
                self.connected = True
                self.current_login = self.login
                # Note: pas de log ici pour eviter le spam (reconnexions frequentes entre comptes)
                return True

            except Exception as e:
                print(f"[MT5] Exception connexion: {e}")
            return False

    def disconnect(self):
        """Deconnexion MT5"""
        global _current_global_login, _current_global_path
        if self.connected:
            mt5.shutdown()
            self.connected = False
            _current_global_login = None
            _current_global_path = None
            print("[MT5] Deconnecte")

    def ensure_connected(self) -> bool:
        """S'assure que la connexion est active ET sur le bon compte ET le bon terminal"""
        global _current_global_login, _current_global_path

        # Recharger la config pour avoir le bon path
        self._load_config()

        # Lock global avec timeout pour eviter les blocages
        with MT5LockContext() as lock_acquired:
            if not lock_acquired:
                return False
            # Verifier si MT5 est connecte au BON compte ET au BON terminal
            if _current_global_login != self.login or _current_global_path != self.path:
                return self.connect()

            # Verifier que MT5 repond (avec timeout)
            info = mt5_with_timeout(mt5.account_info, timeout=5, default=None)
            if info is None or info.login != self.login:
                self.connected = False
                _current_global_login = None
                _current_global_path = None
                return self.connect()

            return True

    def get_account_info(self) -> Optional[Dict]:
        """Recupere les infos du compte"""
        global _account_cache

        # Lock global avec timeout pour eviter les blocages
        with MT5LockContext() as lock_acquired:
            if not lock_acquired:
                # Retourner le cache si disponible
                if self.agent_id and self.agent_id in _account_cache:
                    return _account_cache[self.agent_id]
                return None

            if not self.ensure_connected():
                return None

            # Avec timeout pour eviter blocage
            info = mt5_with_timeout(mt5.account_info, timeout=5, default=None)
            if info is None:
                return None

            result = to_python({
                "login": info.login,
                "balance": info.balance,
                "equity": info.equity,
                "margin": info.margin,
                "margin_free": info.margin_free,
                "profit": info.profit,
                "currency": info.currency
            })

            # Mettre en cache pour eviter reconnexions inutiles
            if self.agent_id:
                _account_cache[self.agent_id] = result

            return result

    def is_currently_connected(self) -> bool:
        """Verifie si CE compte est actuellement le compte connecte (sans forcer reconnexion)"""
        global _current_global_login, _current_global_path
        return _current_global_login == self.login and _current_global_path == self.path

    def get_symbol_info(self) -> Optional[Dict]:
        """Recupere les infos du symbole BTCUSD"""
        # Lock global avec timeout pour eviter les blocages
        with MT5LockContext() as lock_acquired:
            if not lock_acquired:
                return None

            if not self.ensure_connected():
                return None

            info = mt5_with_timeout(lambda: mt5.symbol_info(self.symbol), timeout=5, default=None)
            if info is None:
                # Essayer d'activer le symbole
                select_result = mt5_with_timeout(lambda: mt5.symbol_select(self.symbol, True), timeout=3, default=False)
                if not select_result:
                    print(f"[MT5] Impossible d'activer {self.symbol}")
                    return None
                info = mt5_with_timeout(lambda: mt5.symbol_info(self.symbol), timeout=5, default=None)

            if info is None:
                return None

            tick = mt5_with_timeout(lambda: mt5.symbol_info_tick(self.symbol), timeout=5, default=None)
            if tick is None:
                return None

            spread = tick.ask - tick.bid
            spread_points = int(spread / info.point)

            return to_python({
                "symbol": self.symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": spread,
                "spread_points": spread_points,
                "point": info.point,
                "digits": info.digits,
                "volume_min": info.volume_min,
                "volume_max": info.volume_max,
                "volume_step": info.volume_step,
                "trade_mode": info.trade_mode,
                "time": datetime.fromtimestamp(tick.time).isoformat()
            })

    def get_price(self) -> Optional[float]:
        """Recupere le prix actuel (mid)"""
        info = self.get_symbol_info()
        if info:
            return (info["bid"] + info["ask"]) / 2
        return None

    def get_spread_points(self) -> Optional[int]:
        """Recupere le spread en points"""
        info = self.get_symbol_info()
        if info:
            return info["spread_points"]
        return None

    def get_candles(self, timeframe: str, count: int = 100) -> Optional[List[Dict]]:
        """Recupere les bougies historiques"""
        if not self.ensure_connected():
            return None

        # Mapping timeframe
        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1
        }

        tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)

        rates = mt5_with_timeout(lambda: mt5.copy_rates_from_pos(self.symbol, tf, 0, count), timeout=5, default=None)
        if rates is None or len(rates) == 0:
            return None

        candles = []
        for r in rates:
            candles.append(to_python({
                "time": datetime.fromtimestamp(r['time']).isoformat(),
                "open": r['open'],
                "high": r['high'],
                "low": r['low'],
                "close": r['close'],
                "volume": r['tick_volume']
            }))

        return candles

    def get_positions(self) -> List[Dict]:
        """Recupere les positions ouvertes (toutes, car le symbole peut varier)"""
        # Lock global avec timeout pour eviter les blocages
        with MT5LockContext() as lock_acquired:
            if not lock_acquired:
                return []

            if not self.ensure_connected():
                return []

            # Recuperer TOUTES les positions avec timeout pour eviter blocage
            positions = mt5_with_timeout(mt5.positions_get, timeout=5, default=None)
            if positions is None:
                return []

            # Filtrer seulement les positions BTC (contient "BTC" dans le nom)
            positions = [p for p in positions if "BTC" in p.symbol.upper()]

            result = []
            for pos in positions:
                result.append(to_python({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "time": datetime.fromtimestamp(pos.time).isoformat(),
                    "comment": pos.comment,
                    "magic": pos.magic
                }))

            return result

    def open_position(self, direction: str, volume: float, sl: float = 0, tp: float = 0,
                      comment: str = "") -> Optional[Dict]:
        """Ouvre une position sur BTCUSD"""
        if not self.ensure_connected():
            return None

        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            return None

        # Verifier le spread
        spread_config = load_spread_runtime_config()
        if symbol_info["spread_points"] > spread_config.get("max_spread_points", 2000):
            print(f"[MT5] Spread trop eleve: {symbol_info['spread_points']} > {spread_config.get('max_spread_points', 2000)}")
            return None

        # Prix d'entree
        if direction.upper() == "BUY":
            price = symbol_info["ask"]
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = symbol_info["bid"]
            order_type = mt5.ORDER_TYPE_SELL

        # Ajuster le volume
        volume = max(symbol_info["volume_min"],
                    min(volume, symbol_info["volume_max"]))
        volume = round(volume / symbol_info["volume_step"]) * symbol_info["volume_step"]

        # Creer la requete
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 11,  # Magic number pour G12
            "comment": comment[:31] if comment else "G12",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Envoyer l'ordre (avec timeout pour eviter blocage)
        result = mt5_with_timeout(lambda: mt5.order_send(request), timeout=10, default=None)

        if result is None:
            print(f"[MT5] Erreur/Timeout order_send: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[MT5] Ordre rejete: {result.retcode} - {result.comment}")
            return None

        print(f"[MT5] Position ouverte: {direction} {volume} {self.symbol} @ {result.price}")

        ticket = result.order

        # Modifier SL/TP apres ouverture (certains brokers n'acceptent pas SL/TP dans la requete initiale)
        if sl > 0 or tp > 0:
            sltp_set = self.modify_position(ticket, sl=sl if sl > 0 else None, tp=tp if tp > 0 else None)
            if sltp_set:
                print(f"[MT5] SL/TP definis: SL={sl:.2f}, TP={tp:.2f}")
            else:
                print(f"[MT5] ATTENTION: Echec definition SL/TP pour position {ticket}")

        return to_python({
            "ticket": ticket,
            "direction": direction,
            "volume": volume,
            "price": result.price,
            "sl": sl,
            "tp": tp,
            "comment": comment
        })

    def close_position(self, ticket: int) -> Optional[Dict]:
        """Ferme une position par son ticket"""
        if not self.ensure_connected():
            return None

        # Trouver la position (avec timeout)
        position = mt5_with_timeout(lambda: mt5.positions_get(ticket=ticket), timeout=5, default=None)
        if position is None or len(position) == 0:
            print(f"[MT5] Position {ticket} non trouvee")
            return None

        pos = position[0]

        # Prix de fermeture (avec timeout)
        tick = mt5_with_timeout(lambda: mt5.symbol_info_tick(self.symbol), timeout=5, default=None)
        if tick is None:
            print(f"[MT5] Impossible d'obtenir le tick pour fermeture")
            return None

        if pos.type == mt5.ORDER_TYPE_BUY:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY

        # Creer la requete de fermeture
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 11,
            "comment": "G12_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Envoyer l'ordre (avec timeout)
        result = mt5_with_timeout(lambda: mt5.order_send(request), timeout=10, default=None)

        if result is None:
            print(f"[MT5] Erreur fermeture: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[MT5] Fermeture rejetee: {result.retcode} - {result.comment}")
            return None

        print(f"[MT5] Position {ticket} fermee @ {result.price}, profit: {pos.profit}")

        return to_python({
            "ticket": ticket,
            "close_price": result.price,
            "profit": pos.profit,
            "volume": pos.volume
        })

    def close_all_positions(self) -> List[Dict]:
        """Ferme toutes les positions BTCUSD"""
        positions = self.get_positions()
        results = []

        for pos in positions:
            result = self.close_position(pos["ticket"])
            if result:
                results.append(result)

        return results

    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        """Modifie SL/TP d'une position"""
        if not self.ensure_connected():
            print(f"[MT5] Echec SL/TP: non connecte")
            return False

        # Trouver la position - plusieurs tentatives car le broker peut avoir du délai
        position = None
        for attempt in range(3):
            position = mt5_with_timeout(lambda: mt5.positions_get(ticket=ticket), timeout=5, default=None)
            if position is not None and len(position) > 0:
                break
            print(f"[MT5] Tentative {attempt+1}/3: position {ticket} pas encore visible...")
            time.sleep(1.0)  # Attendre 1 seconde entre chaque tentative

        if position is None or len(position) == 0:
            print(f"[MT5] Position {ticket} non trouvee apres 3 tentatives")
            return False

        pos = position[0]
        print(f"[MT5] Position trouvee: {pos.symbol}, prix={pos.price_open}, SL actuel={pos.sl}, TP actuel={pos.tp}")

        # IMPORTANT: Utiliser le symbole REEL de la position, pas self.symbol
        # Car le broker peut utiliser un nom different (BTCUSDm, BTCUSD.a, etc.)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,  # Symbole reel de la position
            "position": ticket,
            "sl": sl if sl is not None else pos.sl,
            "tp": tp if tp is not None else pos.tp,
        }

        print(f"[MT5] Envoi modification SL/TP: SL={request['sl']}, TP={request['tp']}")
        result = mt5_with_timeout(lambda: mt5.order_send(request), timeout=10, default=None)

        if result is None:
            error = mt5.last_error()
            print(f"[MT5] Erreur modification SL/TP: {error}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[MT5] Modification SL/TP rejetee: code={result.retcode}, comment={result.comment}")
            return False

        print(f"[MT5] SL/TP appliques avec succes!")
        return True

    def calculate_momentum(self, timeframe: str = "1m", periods: int = 5) -> Optional[float]:
        """Calcule le momentum en pourcentage"""
        candles = self.get_candles(timeframe, periods + 1)
        if candles is None or len(candles) < 2:
            return None

        old_close = candles[0]["close"]
        new_close = candles[-1]["close"]

        if old_close == 0:
            return None

        momentum = ((new_close - old_close) / old_close) * 100
        return round(momentum, 4)

    def calculate_volatility(self, timeframe: str = "1h", periods: int = 24) -> Optional[float]:
        """Calcule la volatilite (ATR simplifie)"""
        candles = self.get_candles(timeframe, periods)
        if candles is None or len(candles) < periods:
            return None

        ranges = []
        for c in candles:
            ranges.append(c["high"] - c["low"])

        atr = sum(ranges) / len(ranges)
        current_price = self.get_price()

        if current_price and current_price > 0:
            volatility_pct = (atr / current_price) * 100
            return round(volatility_pct, 4)

        return None

    def get_history_deals(self, from_date: datetime = None, to_date: datetime = None) -> List[Dict]:
        """Recupere l'historique des deals (trades fermes) depuis MT5"""
        if not self.ensure_connected():
            return []

        # Par defaut, recuperer les deals des dernieres 24h
        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(hours=24)

        # Recuperer TOUS les deals avec timeout pour eviter blocage
        # NOTE: history_deals_get() n'accepte PAS de parametre 'group' ou 'symbol'
        # Il faut filtrer manuellement apres recuperation
        deals = mt5_with_timeout(
            lambda: mt5.history_deals_get(from_date, to_date),
            timeout=10, default=None
        )

        if deals is None or len(deals) == 0:
            return []

        result = []
        for deal in deals:
            # Filtrer par symbole BTCUSD
            if deal.symbol != self.symbol:
                continue

            # Filtrer: ne garder que les deals de sortie (DEAL_ENTRY_OUT) avec magic 11 ou comment G12
            if deal.entry == 1:  # DEAL_ENTRY_OUT = sortie de position
                if deal.magic == 11 or "G12" in (deal.comment or ""):
                    result.append(to_python({
                        "ticket": deal.ticket,
                        "order": deal.order,
                        "position_id": deal.position_id,
                        "time": deal.time,
                        "type": "BUY" if deal.type == 0 else "SELL",
                        "volume": deal.volume,
                        "price": deal.price,
                        "profit": deal.profit,
                        "commission": deal.commission,
                        "swap": deal.swap,
                        "comment": deal.comment,
                        "magic": deal.magic
                    }))

        return result


# Instances par agent (multi-compte)
_mt5_instances: Dict[str, MT5Connector] = {}
_mt5_default = None
_current_global_login = None  # Compte actuellement connecte (MT5 est global)
_current_global_path = None  # Terminal MT5 actuellement utilise
_account_cache: Dict[str, Dict] = {}  # Cache des infos compte pour eviter reconnexions


def get_mt5(agent_id: str = None) -> MT5Connector:
    """Retourne l'instance MT5 pour un agent ou l'instance par defaut"""
    global _mt5_instances, _mt5_default

    if agent_id:
        if agent_id not in _mt5_instances:
            _mt5_instances[agent_id] = MT5Connector(agent_id)
        return _mt5_instances[agent_id]
    else:
        if _mt5_default is None:
            _mt5_default = MT5Connector()
        return _mt5_default


def get_cached_account_info(agent_id: str) -> Optional[Dict]:
    """
    Retourne les infos du compte en cache SANS forcer de reconnexion.
    Utilise pour afficher le status sans faire clignoter MT5.
    Retourne None si aucune donnee en cache.
    """
    global _account_cache
    return _account_cache.get(agent_id)


def is_agent_currently_connected(agent_id: str) -> bool:
    """Verifie si un agent est actuellement le compte connecte (sans reconnexion)"""
    global _current_global_login, _current_global_path, _mt5_instances

    if agent_id not in _mt5_instances:
        return False

    mt5_instance = _mt5_instances[agent_id]
    return _current_global_login == mt5_instance.login and _current_global_path == mt5_instance.path


def get_all_mt5_accounts() -> Dict:
    """Retourne le status de tous les comptes MT5"""
    # Charger la config runtime
    config_file = DATABASE_DIR / "mt5_accounts_runtime.json"
    runtime_config = {}

    try:
        if config_file.exists():
            with open(config_file, 'r') as f:
                runtime_config = json.load(f)
    except Exception:
        pass

    # Fusionner avec la config par defaut
    accounts = {}
    for agent_id in ["fibo1", "fibo2", "fibo3"]:
        if agent_id in runtime_config:
            accounts[agent_id] = runtime_config[agent_id]
        elif agent_id in MT5_ACCOUNTS:
            accounts[agent_id] = MT5_ACCOUNTS[agent_id].copy()
            # Ne pas exposer le password
            if "password" in accounts[agent_id]:
                accounts[agent_id]["password"] = "***" if accounts[agent_id]["password"] else ""
        else:
            accounts[agent_id] = {"login": 0, "server": "", "path": "", "enabled": False}

    return accounts


def save_mt5_account(agent_id: str, config: Dict) -> bool:
    """Sauvegarde la config MT5 pour un agent"""
    config_file = DATABASE_DIR / "mt5_accounts_runtime.json"

    try:
        # Charger config existante
        runtime_config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                runtime_config = json.load(f)

        # Mettre a jour
        if agent_id not in runtime_config:
            runtime_config[agent_id] = {}

        for key in ['login', 'password', 'server', 'path', 'enabled']:
            if key in config:
                runtime_config[agent_id][key] = config[key]

        # Sauvegarder
        with open(config_file, 'w') as f:
            json.dump(runtime_config, f, indent=2)

        # Recharger l'instance si elle existe
        if agent_id in _mt5_instances:
            _mt5_instances[agent_id]._load_config()

        return True

    except Exception as e:
        print(f"[MT5] Erreur sauvegarde config: {e}")
        return False
