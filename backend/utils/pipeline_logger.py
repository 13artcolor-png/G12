# -*- coding: utf-8 -*-
"""
G12 - Pipeline Logger
Enregistre toutes les etapes du processus de decision/ouverture/fermeture
pour visibilite complete du systeme
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DATABASE_DIR = Path(__file__).parent.parent / "database"
PIPELINE_LOGS_FILE = DATABASE_DIR / "pipeline_logs.json"
MAX_LOGS_IN_FILE = 100  # Garder les 100 dernieres actions


class PipelineLogger:
    """Logger pour le pipeline complet de decision"""

    def __init__(self, agent_id: str, action_type: str):
        """
        Args:
            agent_id: ID de l'agent (fibo1, fibo2, fibo3)
            action_type: Type d'action (open, close, hold)
        """
        self.agent_id = agent_id
        self.action_type = action_type
        self.pipeline_id = f"{agent_id}_{int(time.time() * 1000)}"
        self.start_time = time.time()
        self.steps = []
        self.result = {}
        self.current_step_start = None

    def start_step(self, step_name: str):
        """Demarre un chrono pour une etape"""
        self.current_step_start = time.time()
        self.current_step_name = step_name

    def end_step(self, status: str = "success", details: Dict = None):
        """Termine une etape et enregistre sa duree"""
        if not self.current_step_start:
            return

        duration_ms = int((time.time() - self.current_step_start) * 1000)

        step_data = {
            "step": len(self.steps) + 1,
            "name": self.current_step_name,
            "status": status,  # success, warning, error
            "duration_ms": duration_ms,
            "details": details or {}
        }

        self.steps.append(step_data)
        self.current_step_start = None

    def set_result(self, result: Dict):
        """Definit le resultat final de l'action"""
        self.result = result

    def save(self):
        """Sauvegarde le pipeline dans le fichier JSON"""
        try:
            # Charger les logs existants
            if PIPELINE_LOGS_FILE.exists():
                with open(PIPELINE_LOGS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logs = data.get('logs', [])
            else:
                logs = []

            # Calculer duree totale
            total_duration_ms = int((time.time() - self.start_time) * 1000)

            # Construire l'entree de log
            log_entry = {
                "pipeline_id": self.pipeline_id,
                "timestamp": datetime.now().isoformat(),
                "agent": self.agent_id,
                "action_type": self.action_type,
                "total_duration_ms": total_duration_ms,
                "steps": self.steps,
                "result": self.result,
                "success": all(s.get("status") == "success" for s in self.steps)
            }

            # Ajouter au debut de la liste (plus recent en premier)
            logs.insert(0, log_entry)

            # Limiter a MAX_LOGS_IN_FILE
            logs = logs[:MAX_LOGS_IN_FILE]

            # Sauvegarder
            with open(PIPELINE_LOGS_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "updated_at": datetime.now().isoformat(),
                    "total_logs": len(logs),
                    "logs": logs
                }, f, indent=2, ensure_ascii=False)

            print(f"[PipelineLogger] Log sauvegarde: {self.pipeline_id} ({total_duration_ms}ms)")

        except Exception as e:
            print(f"[PipelineLogger] Erreur sauvegarde: {e}")


def get_latest_pipelines(agent_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Recupere les derniers pipelines enregistres

    Args:
        agent_id: Filtrer par agent (optionnel)
        limit: Nombre max de resultats

    Returns:
        Liste des pipelines
    """
    try:
        if not PIPELINE_LOGS_FILE.exists():
            return []

        with open(PIPELINE_LOGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logs = data.get('logs', [])

        # Filtrer par agent si specifie
        if agent_id:
            logs = [log for log in logs if log.get('agent') == agent_id]

        # Limiter
        return logs[:limit]

    except Exception as e:
        print(f"[PipelineLogger] Erreur lecture logs: {e}")
        return []


def get_pipeline_by_id(pipeline_id: str) -> Optional[Dict]:
    """
    Recupere un pipeline specifique par son ID

    Args:
        pipeline_id: ID du pipeline

    Returns:
        Pipeline ou None
    """
    try:
        if not PIPELINE_LOGS_FILE.exists():
            return None

        with open(PIPELINE_LOGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logs = data.get('logs', [])

        for log in logs:
            if log.get('pipeline_id') == pipeline_id:
                return log

        return None

    except Exception as e:
        print(f"[PipelineLogger] Erreur lecture pipeline: {e}")
        return None
