"""
Main application entry point for the Research Network Analysis Dashboard.
"""

from shiny import App
from ui import app_ui
from server import create_server

# Create the app
app = App(app_ui, create_server())

if __name__ == "__main__":
    app.run()
