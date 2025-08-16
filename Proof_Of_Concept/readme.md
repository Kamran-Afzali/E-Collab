# E-Collab Network Analysis Dashboard

A comprehensive interactive dashboard for analyzing research collaborations and institutional networks. This application provides two main analysis modules: Network Analysis for exploring author collaborations, and Institution Analysis for examining research patterns across academic institutions.

## Features

### Network Analysis
- **Interactive Network Visualization**: Explore collaboration networks between researchers
- **Advanced Filtering**: Filter by affiliations, keywords, publication counts, and co-publication thresholds
- **Data Export**: Download filtered network data, metadata, and visualizations
- **Summary Statistics**: View dataset overview and affiliation distributions
- **Multi-tab Interface**: Separate views for network visualization, author data, collaboration data, and statistics

### Institution Analysis
- **Keyword Analysis**: Visualize top research keywords by institution
- **Publication Analysis**: Compare top authors by publication count across institutions
- **Collaboration Analysis**: Examine collaboration patterns within institutions
- **Multi-institution Comparison**: Select and compare multiple institutions simultaneously
- **Dark Theme Visualization**: Professional dark-themed charts for better presentation

## Installation

1. **Clone or download the project files**
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to the URL displayed in the terminal (typically `http://127.0.0.1:8000`)

3. **Explore the dashboard**:
   - Use the **Network Analysis** tab to explore research collaboration networks
   - Use the **Institution Analysis** tab to compare institutional research patterns
   - Adjust filters and selections to customize your analysis

## Project Structure

```
research-dashboard/
├── app.py              # Main application entry point
├── ui.py               # User interface definitions
├── server.py           # Server logic and data processing
├── data_generator.py   # Synthetic data generation functions
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Key Components

- **Data Generation**: Synthetic research data with realistic author names, affiliations, keywords, and collaboration patterns
- **Interactive Filtering**: Real-time data filtering based on user selections
- **Network Visualization**: Graph-based visualization using NetworkX and Plotly
- **Institutional Comparisons**: Multi-panel visualizations for comparing research metrics
- **Data Export**: CSV downloads and HTML plot exports

## Technology Stack

- **Frontend**: Shiny for Python (UI framework)
- **Visualization**: Plotly (interactive charts and network graphs)
- **Data Processing**: Pandas, NumPy
- **Network Analysis**: NetworkX
- **Backend**: Python

## Sample Data

The application generates synthetic research data including:
- 100 authors with realistic names and affiliations
- 64 research keywords across AI/ML domains
- 300+ collaboration relationships
- 10 major academic institutions
- Publication counts and co-publication metrics

## Customization

To use your own data:
1. Modify the `data_generator.py` file to load your actual datasets
2. Ensure data follows the expected column structure:
   - **Author metadata**: Author, Nb_publications, Affiliation_clean, Keywords
   - **Network data**: Author1, Author2, Nb_co_publications
   - **Institution keywords**: Affiliations, Keywords, Count

## Requirements

- Python 3.8+
- Modern web browser with JavaScript enabled
- 512MB+ RAM recommended for large datasets

## License

This project is provided as-is for educational and research purposes.
