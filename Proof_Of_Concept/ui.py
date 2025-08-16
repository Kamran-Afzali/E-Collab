"""
User Interface definitions for the Research Network Analysis Dashboard.
"""

from shiny import ui
from data_generator import generate_network_data, generate_institution_data

# Generate data for UI choices
(_, _, affs_net, keywords_list_net) = generate_network_data()
(_, _, _, affiliations_inst) = generate_institution_data()

app_ui = ui.page_navbar(
    # Network Analysis Tab
    ui.nav_panel(
        "Network Analysis",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Filters"),
                ui.input_slider("bins", "Minimum Publications", min=1, max=100, value=10),
                ui.input_slider("bins2", "Minimum Co-Publications", min=1, max=50, value=3),
                ui.input_selectize(
                    "AffInput",
                    "Affiliation",
                    choices=affs_net,
                    selected=affs_net[:3],
                    multiple=True
                ),
                ui.input_selectize(
                    "search",
                    "Keyword Search",
                    choices=keywords_list_net,
                    selected=["Artificial intelligence", "Machine learning", "Deep learning", "Neural network"],
                    multiple=True
                ),
                ui.br(),
                ui.h4("Downloads"),
                ui.download_button("download_net_data", "Download Net Data", class_="btn-primary"),
                ui.br(),
                ui.download_button("download_meta_data", "Download Meta Data", class_="btn-primary"),
                ui.br(),
                ui.download_button("download_plot", "Download Plot", class_="btn-primary"),
                width=300
            ),
            ui.navset_tab(
                ui.nav_panel(
                    "Network Visualization",
                    ui.card(
                        ui.card_header("Network Analysis Dashboard"),
                        ui.output_ui("network_plot"),
                        full_screen=True
                    )
                ),
                ui.nav_panel(
                    "Author Data",
                    ui.card(
                        ui.card_header("Author Metadata"),
                        ui.output_data_frame("meta_data_table")
                    )
                ),
                ui.nav_panel(
                    "Collaboration Data",
                    ui.card(
                        ui.card_header("Collaboration Network Data"),
                        ui.output_data_frame("network_data_table")
                    )
                ),
                ui.nav_panel(
                    "Summary Statistics",
                    ui.card(
                        ui.card_header("Dataset Summary"),
                        ui.output_ui("summary_stats")
                    )
                )
            )
        )
    ),
    
    # Institution Analysis Tab
    ui.nav_panel(
        "Institution Analysis",
        ui.div(
            {"style": "background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px;"},
            ui.h1("Institution Analysis Dashboard", style="text-align: center; margin: 0;")
        ),
        ui.div(
            {"style": "background-color: #34495e; padding: 15px; margin-bottom: 20px;"},
            ui.input_selectize(
                "ia_institution_input",  # Namespaced ID
                "Select Institution(s):",
                choices=affiliations_inst,
                selected=affiliations_inst[:3],
                multiple=True,
                width="100%"
            )
        ),
        ui.div(
            {"style": "background-color: #ecf0f1; padding: 20px;"},
            ui.navset_tab(
                ui.nav_panel("By Keyword", ui.output_ui("ia_keyword_plot", height="800px")),
                ui.nav_panel("By Publication", ui.output_ui("ia_publication_plot", height="800px")),
                ui.nav_panel("By Collaboration", ui.output_ui("ia_collaboration_plot", height="800px")),
                id="ia_main_tabs"
            )
        )
    ),
    title="Unified Research Dashboard",
    id="main_navbar"
)
