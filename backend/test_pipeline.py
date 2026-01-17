# -*- coding: utf-8 -*-
"""
Test du systeme Pipeline Logger
"""

from utils.pipeline_logger import PipelineLogger
import time

def test_pipeline_logger():
    """Test simple du PipelineLogger"""
    print("=== Test Pipeline Logger ===\n")

    # Creer un pipeline de test
    pipeline = PipelineLogger("fibo1", "open")

    # Etape 1: Verifications
    pipeline.start_step("Verifications preliminaires")
    time.sleep(0.02)  # Simuler le temps
    pipeline.end_step("success", {
        "spread_check": True,
        "budget_check": True,
        "cooldown_check": True
    })

    # Etape 2: Vision AI
    pipeline.start_step("Vision AI - Generation graphiques")
    time.sleep(0.8)  # Simuler generation
    pipeline.end_step("success", {
        "charts_generated": 2,
        "timeframes": ["1m", "15m"],
        "total_bytes": 73248
    })

    # Etape 3: Appel IA
    pipeline.start_step("Appel API IA")
    time.sleep(2.3)  # Simuler appel API
    pipeline.end_step("success", {
        "api_duration_ms": 2300,
        "model": "claude-sonnet-4-5",
        "action": "BUY",
        "confidence": 75
    })

    # Etape 4: Filtre
    pipeline.start_step("Filtre consensus")
    time.sleep(0.01)
    pipeline.end_step("success", {
        "aligned": True,
        "consensus_bias": "bullish",
        "consensus_confidence": 68
    })

    # Resultat final
    pipeline.set_result({
        "action": "BUY",
        "ticket": 231975430,
        "price": 95356,
        "agent": "fibo1"
    })

    # Sauvegarder
    pipeline.save()

    print(f"Pipeline ID: {pipeline.pipeline_id}")
    print(f"Duree totale: {len(pipeline.steps)} etapes")
    print("\nEtapes:")
    for step in pipeline.steps:
        print(f"  {step['step']}. {step['name']}: {step['duration_ms']}ms - {step['status']}")

    print(f"\nResultat: {pipeline.result}")
    print("\n=== Test termine ===")

    # Tester la recuperation
    print("\n=== Test recuperation ===")
    from utils.pipeline_logger import get_latest_pipelines

    latest = get_latest_pipelines(agent_id="fibo1", limit=1)
    if latest:
        print(f"Dernier pipeline recupere: {latest[0]['pipeline_id']}")
        print(f"Nombre d'etapes: {len(latest[0]['steps'])}")
        print(f"Succes: {latest[0]['success']}")
    else:
        print("Aucun pipeline trouve")

if __name__ == "__main__":
    test_pipeline_logger()
