"""
Configurações globais do dashboard.
Contém constantes de estilo, cores, features e configurações gerais.
"""

from typing import Dict

# ============================================================================
# ESTILO E CORES
# ============================================================================

# Gradiente de fundo principal
BACKGROUND_GRADIENT = (
    "radial-gradient(circle at 20% 20%, rgba(76, 201, 240, 0.14), transparent 46%)"
    ", radial-gradient(circle at 80% -10%, rgba(255, 109, 196, 0.16), transparent 50%)"
    ", linear-gradient(135deg, #05070d 0%, #0b101a 55%, #05070d 100%)"
)

# Cores dos cards
CARD_BACKGROUND = "rgba(16, 20, 26, 0.78)"
CARD_BORDER_COLOR = "rgba(255, 255, 255, 0.08)"
CARD_BORDER_STYLE = f"1px solid {CARD_BORDER_COLOR}"
CARD_SHADOW = "0 32px 65px rgba(0, 0, 0, 0.45)"

# Cores de texto
PRIMARY_TEXT = "#f8f9fa"
SECONDARY_TEXT = "#aeb6c4"

# Cores de destaque
ACCENT_COLOR = "#4cc9f0"
ACCENT_SECONDARY = "#ff6dc4"
ACCENT_TERTIARY = "#ffd60a"

# Fonte padrão
FONT_FAMILY = "'Poppins', 'Inter', sans-serif"

# Grade
GRID_COLOR = "rgba(255, 255, 255, 0.05)"

# ============================================================================
# ESTILOS DE TABELAS
# ============================================================================

TABLE_HEADER_STYLE: Dict[str, str] = {
    "backgroundColor": "rgba(33, 37, 41, 0.85)",
    "color": PRIMARY_TEXT,
    "fontWeight": "600",
    "border": CARD_BORDER_STYLE,
}

TABLE_CELL_STYLE: Dict[str, str] = {
    "backgroundColor": "rgba(12, 16, 22, 0.75)",
    "color": PRIMARY_TEXT,
    "border": CARD_BORDER_STYLE,
}

# ============================================================================
# ESTILOS DAS ABAS
# ============================================================================

TAB_STYLE = {
    "padding": "14px",
    "fontWeight": "500",
    "backgroundColor": "transparent",
    "color": SECONDARY_TEXT,
    "border": "0",
}

TAB_SELECTED_STYLE = TAB_STYLE | {
    "background": "linear-gradient(135deg, rgba(76, 201, 240, 0.3), rgba(255, 109, 196, 0.25))",
    "color": PRIMARY_TEXT,
}

# ============================================================================
# FEATURES DE MACHINE LEARNING
# ============================================================================

CLASSIFIER_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "track_popularity",
]

CLUSTER_FEATURES = [
    "valence",
    "energy",
    "danceability",
    "tempo",
    "acousticness",
]

# ============================================================================
# LABELS DE TONALIDADE E MODO
# ============================================================================

KEY_LABELS_MAP = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

MODE_LABELS = {0: "Menor", 1: "Maior"}

# ============================================================================
# CONFIGURAÇÕES DO DASH
# ============================================================================

EXTERNAL_STYLESHEETS = [
    "https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/darkly/bootstrap.min.css"
]

# URL do Google Fonts
GOOGLE_FONTS_URL = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap"

DASH_CONFIG = {
    "suppress_callback_exceptions": True,
    "title": "Spotify Insights Dashboard",
    "update_title": None,
}
