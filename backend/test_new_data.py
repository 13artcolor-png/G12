# -*- coding: utf-8 -*-
"""
G12 - Test des nouvelles sources de donnees
"""

import sys
from pathlib import Path

# Ajouter le chemin backend pour les imports
sys.path.append(str(Path(__file__).parent))

from data.macro_engine import get_macro_engine
from data.whale_tracker import get_whale_tracker
from data.sentiment import get_sentiment

def test_sources():
    print("--- TEST MACRO ENGINE ---")
    macro = get_macro_engine()
    data = macro.get_macro_data()
    print(f"DXY: {data.get('dxy')}")
    print(f"S&P 500: {data.get('sp500')}")
    print(f"Bias Macro: {data.get('macro_bias')}")
    
    print("\n--- TEST WHALE TRACKER ---")
    whale = get_whale_tracker()
    bias = whale.get_whale_bias()
    print(f"Whale Bias: {bias}")
    print(f"Nb Whales detectees: {len(whale.cache)}")
    
    print("\n--- TEST BTC DOMINANCE ---")
    sent = get_sentiment()
    dom = sent.get_btc_dominance()
    print(f"BTC Dominance: {dom}%")

if __name__ == "__main__":
    test_sources()
