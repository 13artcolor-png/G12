# -*- coding: utf-8 -*-
"""
G12 - Générateur de graphiques pour analyse visuelle IA
Génère des graphiques BTCUSD aux timeframes pertinents et les encode en base64
"""

import io
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import matplotlib
matplotlib.use('Agg')  # Backend sans interface graphique
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

from core.mt5_connector import get_mt5

# Chemins
DATABASE_DIR = Path(__file__).parent.parent / "database"
CHARTS_CACHE_DIR = DATABASE_DIR / "charts_cache"
CHARTS_CACHE_DIR.mkdir(exist_ok=True)


class ChartGenerator:
    """Générateur de graphiques BTCUSD pour analyse IA"""

    def __init__(self, agent_id: str = "fibo1"):
        self.mt5 = get_mt5(agent_id)
        self.cache_dir = CHARTS_CACHE_DIR

    def generate_candlestick_chart(
        self,
        timeframe: str = "15m",
        candle_count: int = 100,
        width: int = 800,
        height: int = 600,
        show_volume: bool = True,
        return_base64: bool = True
    ) -> Optional[str]:
        """
        Génère un graphique en chandeliers japonais

        Args:
            timeframe: "1m", "5m", "15m", "1h", etc.
            candle_count: Nombre de bougies à afficher
            width: Largeur en pixels
            height: Hauteur en pixels
            show_volume: Afficher les volumes
            return_base64: Retourner en base64 (True) ou sauvegarder fichier (False)

        Returns:
            String base64 de l'image ou chemin fichier
        """
        try:
            # Récupérer les bougies
            candles = self.mt5.get_candles(timeframe, candle_count)
            if not candles or len(candles) < 10:
                print(f"[ChartGen] Pas assez de données pour {timeframe}")
                return None

            # Convertir en arrays
            times = [datetime.fromisoformat(c['time']) for c in candles]
            opens = [c['open'] for c in candles]
            highs = [c['high'] for c in candles]
            lows = [c['low'] for c in candles]
            closes = [c['close'] for c in candles]
            volumes = [c['volume'] for c in candles]

            # Créer la figure
            dpi = 100
            fig_width = width / dpi
            fig_height = height / dpi

            if show_volume:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1,
                    figsize=(fig_width, fig_height),
                    gridspec_kw={'height_ratios': [3, 1]},
                    dpi=dpi
                )
            else:
                fig, ax1 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

            # Style
            fig.patch.set_facecolor('#1a1a1a')
            ax1.set_facecolor('#0d0d0d')

            # Chandeliers japonais
            for i, (time, o, h, l, c) in enumerate(zip(times, opens, highs, lows, closes)):
                color = '#26a69a' if c >= o else '#ef5350'  # Vert si hausse, rouge si baisse

                # Mèche (high-low)
                ax1.plot([i, i], [l, h], color=color, linewidth=1, solid_capstyle='round')

                # Corps (open-close)
                body_height = abs(c - o)
                body_bottom = min(o, c)
                rect = Rectangle(
                    (i - 0.4, body_bottom),
                    0.8,
                    body_height if body_height > 0 else h * 0.0001,  # Éviter hauteur 0
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0
                )
                ax1.add_patch(rect)

            # Axes
            ax1.set_xlim(-0.5, len(candles) - 0.5)
            ax1.set_ylim(min(lows) * 0.999, max(highs) * 1.001)

            # Labels
            ax1.set_ylabel('Prix (USD)', color='#ffffff', fontsize=10)
            ax1.tick_params(colors='#888888', labelsize=8)
            ax1.grid(True, alpha=0.1, color='#444444')

            # Titre
            current_price = closes[-1]
            change_pct = ((closes[-1] - closes[0]) / closes[0]) * 100
            change_color = '#26a69a' if change_pct >= 0 else '#ef5350'

            title = f"BTCUSD {timeframe.upper()} | ${current_price:,.0f} ({change_pct:+.2f}%)"
            ax1.set_title(title, color='#ffffff', fontsize=12, pad=10)

            # Volumes (si activé)
            if show_volume:
                ax2.set_facecolor('#0d0d0d')
                colors_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(closes, opens)]
                ax2.bar(range(len(volumes)), volumes, color=colors_vol, alpha=0.5, width=0.8)
                ax2.set_xlim(-0.5, len(candles) - 0.5)
                ax2.set_ylabel('Volume', color='#ffffff', fontsize=9)
                ax2.tick_params(colors='#888888', labelsize=7)
                ax2.grid(True, alpha=0.1, color='#444444')

            # Format temps sur l'axe X
            x_labels_count = min(8, len(times))
            x_indices = np.linspace(0, len(times) - 1, x_labels_count, dtype=int)

            if show_volume:
                ax2.set_xticks(x_indices)
                ax2.set_xticklabels([times[i].strftime('%H:%M') for i in x_indices], rotation=45)
            else:
                ax1.set_xticks(x_indices)
                ax1.set_xticklabels([times[i].strftime('%H:%M') for i in x_indices], rotation=45)

            plt.tight_layout()

            # Sauvegarder ou retourner base64
            if return_base64:
                # Convertir en base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', facecolor='#1a1a1a', edgecolor='none')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close(fig)
                return img_base64
            else:
                # Sauvegarder fichier
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"BTCUSD_{timeframe}_{timestamp}.png"
                filepath = self.cache_dir / filename
                plt.savefig(filepath, facecolor='#1a1a1a', edgecolor='none')
                plt.close(fig)
                print(f"[ChartGen] Graphique sauvegardé: {filename}")
                return str(filepath)

        except Exception as e:
            print(f"[ChartGen] Erreur génération graphique {timeframe}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_multi_timeframe_charts(
        self,
        timeframes: List[str] = ["1m", "5m", "15m", "1h"],
        candle_count: int = 100
    ) -> Dict[str, str]:
        """
        Génère plusieurs graphiques (multi-timeframe)

        Args:
            timeframes: Liste des timeframes à générer
            candle_count: Nombre de bougies par graphique

        Returns:
            Dict {timeframe: base64_image}
        """
        results = {}

        print(f"[ChartGen] Génération de {len(timeframes)} graphiques...")

        for tf in timeframes:
            img_base64 = self.generate_candlestick_chart(
                timeframe=tf,
                candle_count=candle_count,
                return_base64=True
            )

            if img_base64:
                results[tf] = img_base64
                print(f"[ChartGen] OK {tf} genere ({len(img_base64)} bytes)")
            else:
                print(f"[ChartGen] ERREUR {tf} echec")

        return results

    def cleanup_old_charts(self, keep_last: int = 20):
        """Nettoie les anciens graphiques du cache"""
        try:
            charts = sorted(self.cache_dir.glob("BTCUSD_*.png"))

            if len(charts) > keep_last:
                to_delete = charts[:-keep_last]
                for f in to_delete:
                    f.unlink()
                print(f"[ChartGen] {len(to_delete)} anciens graphiques supprimés")
        except Exception as e:
            print(f"[ChartGen] Erreur nettoyage: {e}")


# Singleton
_chart_generator = None

def get_chart_generator(agent_id: str = "fibo1") -> ChartGenerator:
    """Retourne l'instance ChartGenerator singleton"""
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = ChartGenerator(agent_id)
    return _chart_generator


def generate_charts_for_context(timeframes: List[str] = None) -> Dict[str, str]:
    """
    Fonction helper pour générer rapidement des graphiques pour le contexte IA

    Args:
        timeframes: Liste des timeframes (défaut: ["1m", "15m"])

    Returns:
        Dict {timeframe: base64_image}
    """
    if timeframes is None:
        timeframes = ["1m", "15m"]  # Timeframes par défaut pour l'IA

    generator = get_chart_generator()
    return generator.generate_multi_timeframe_charts(timeframes)
