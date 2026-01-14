from .base_agent import BaseAgent
from typing import Dict, Tuple

class SimpleAgent(BaseAgent):
    """
    Agent minimal pour les tâches utilitaires (Analyse News, Self-Learning, etc.)
    qui ne nécessitent pas de logique de trading complète.
    """
    
    def _load_api_config(self):
        """Surcharge pour fallback sur strategist si pas de clé spécifique"""
        super()._load_api_config()
        # Si pas de clé trouvée, on essaie celle du strategist ou momentum
        if not self.api_config or not self.api_config.get('key'):
            try:
                # Hack: on change temporairement l'ID pour charger la config du strategist
                original_id = self.agent_id
                self.agent_id = "strategist" 
                super()._load_api_config()
                self.agent_id = original_id # Restore
                
                # Si toujours rien, on essaie momentum
                if not self.api_config or not self.api_config.get('key'):
                    self.agent_id = "fibo1"
                    super()._load_api_config()
                    self.agent_id = original_id
            except Exception as e:
                print(f"[SimpleAgent] Erreur fallback API: {e}")

    def get_opener_prompt(self, context: Dict) -> str:
        """Non utilisé pour les tâches utilitaires"""
        return ""

    def get_closer_prompt(self, context: Dict, position: Dict) -> str:
        """Non utilisé pour les tâches utilitaires"""
        return ""

    def should_consider_trade(self, context: Dict) -> Tuple[bool, str]:
        """Ne trade jamais"""
        return False, "SimpleAgent ne trade pas"
