"""
Configuration settings for the Research Network Analysis Dashboard.
"""

# Application settings
APP_TITLE = "Unified Research Dashboard"
DEFAULT_PORT = 8000
DEBUG_MODE = False

# Network Analysis Configuration
NETWORK_CONFIG = {
    'default_min_publications': 10,
    'default_min_co_publications': 3,
    'max_publications': 100,
    'max_co_publications': 50,
    'network_layout_iterations': 50,
    'network_layout_k': 1,
    'node_size': 20,
    'edge_width': 1,
    'plot_height': 600
}

# Institution Analysis Configuration
INSTITUTION_CONFIG = {
    'plot_height': 800,
    'max_subplot_cols': 3,
    'top_keywords_limit': 10,
    'top_authors_limit': 7,
    'top_collaborations_limit': 7,
    'plot_template': 'plotly_dark',
    'font_size': 12
}

# Color schemes
COLORS = {
    'network_nodes': [
        "gold", "#004949", "#009292", "#ff6db6", "#ffb6db",
        "#490092", "#006ddb", "#b66dff", "#6db6ff", "#b6dbff",
        "#920000", "#924900", "#db6d00", "#24ff24", "#ffff6d"
    ],
    'institution_bars': 'Set1'  # Plotly color scheme
}

# Data generation settings (for synthetic data)
DATA_CONFIG = {
    'num_authors': 100,
    'num_collaborations': 300,
    'min_keywords_per_author': 2,
    'max_keywords_per_author': 5,
    'min_publications': 1,
    'max_publications': 100,
    'min_co_publications': 1,
    'max_co_publications': 20,
    'random_seed': 42
}

# UI Configuration
UI_CONFIG = {
    'sidebar_width': 300,
    'enable_full_screen': True,
    'show_download_buttons': True
}
