"""
Server logic for the Research Network Analysis Dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from shiny import render, reactive
from plotly.subplots import make_subplots
import plotly.io as pio
from data_generator import generate_network_data, generate_institution_data
from shiny import ui

def create_server():
    """Create and return the server function."""
    
    # Generate all datasets
    (meta_data_net, net_data_net, affs_net, keywords_list_net) = generate_network_data()
    (kwss2_inst, meta_data_inst, net_data_inst, affiliations_inst) = generate_institution_data()
    
    def server(input, output, session):
        # Shared plot configuration for Institution Analysis
        PLOT_CONFIG = {
            'height': 800,
            'template': 'plotly_dark',
            'font': {'size': 12}
        }

        # --------------------------
        # Network Analysis Functions
        # --------------------------
        @reactive.calc
        def filtered_data():
            """Filter data for Network Analysis"""
            if not input.search():
                search_pattern = ""
            else:
                search_pattern = "|".join(input.search())
            
            if search_pattern:
                meta_filtered = meta_data_net[
                    meta_data_net['Keywords'].str.contains(search_pattern, case=False, na=False)
                ]
            else:
                meta_filtered = meta_data_net.copy()
            
            if input.AffInput():
                meta_filtered = meta_filtered[
                    meta_filtered['Affiliation_clean'].isin(input.AffInput())
                ]
            
            meta_filtered = meta_filtered[
                meta_filtered['Nb_publications'] >= input.bins()
            ]
            
            net_filtered = net_data_net[net_data_net['Nb_co_publications'] >= input.bins2()]
            meta_authors = set(meta_filtered['Author'].tolist())
            net_authors = set(net_filtered['Author1'].tolist() + net_filtered['Author2'].tolist())
            common_authors = meta_authors.intersection(net_authors)
            net_final = net_filtered[
                (net_filtered['Author1'].isin(common_authors)) &
                (net_filtered['Author2'].isin(common_authors))
            ]
            connected_authors = set(net_final['Author1'].tolist() + net_final['Author2'].tolist())
            meta_network = meta_filtered[meta_filtered['Author'].isin(connected_authors)]
            return meta_filtered, meta_network, net_final

        @render.ui
        def network_plot():
            """Network visualization"""
            meta_filtered, meta_network, net_df = filtered_data()
            
            if len(meta_network) == 0 or len(net_df) == 0:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data matches the current filters",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                fig.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='white',
                    height=600
                )
                return ui.HTML(fig.to_html(include_plotlyjs=True, div_id="network_plot"))
            
            G = nx.Graph()
            for _, row in meta_network.iterrows():
                G.add_node(row['Author'], 
                          nb_pubs=row['Nb_publications'],
                          affiliation=row['Affiliation_clean'],
                          keywords=row['Keywords'])
            
            for _, row in net_df.iterrows():
                if row['Author1'] in G.nodes and row['Author2'] in G.nodes:
                    G.add_edge(row['Author1'], row['Author2'], 
                              weight=row['Nb_co_publications'])
            
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Edge traces
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Node traces
            node_x = []
            node_y = []
            node_info = []
            node_colors = []
            colors = ["gold", "#004949", "#009292", "#ff6db6", "#ffb6db",
                     "#490092", "#006ddb", "#b66dff", "#6db6ff", "#b6dbff",
                     "#920000", "#924900", "#db6d00", "#24ff24", "#ffff6d"]
            
            unique_affs = meta_network['Affiliation_clean'].unique()
            aff_colors = {aff: colors[i % len(colors)] for i, aff in enumerate(unique_affs)}
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_data = meta_network[meta_network['Author'] == node].iloc[0]
                info = f"Author: {node}<br>Affiliation: {node_data['Affiliation_clean']}<br>Publications: {node_data['Nb_publications']}"
                node_info.append(info)
                node_colors.append(aff_colors[node_data['Affiliation_clean']])
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[node.split()[-1] for node in G.nodes()],
                textposition="middle center",
                hovertext=node_info,
                marker=dict(
                    showscale=False,
                    color=node_colors,
                    size=20,
                    line=dict(width=2, color='black')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title=f'Network Analysis - {len(G.nodes())} authors, {len(G.edges())} collaborations',
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text="Hover over nodes for details",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor='left', yanchor='bottom',
                                   font=dict(color='gray', size=12)
                               )],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               plot_bgcolor='white',
                               height=600
                           ))
            
            return ui.HTML(fig.to_html(include_plotlyjs=True, div_id="network_plot"))

        @render.data_frame
        def meta_data_table():
            meta_filtered, _, _ = filtered_data()
            return render.DataGrid(meta_filtered, row_selection_mode='multiple')

        @render.data_frame
        def network_data_table():
            _, _, net_filtered = filtered_data()
            return render.DataGrid(net_filtered, row_selection_mode='multiple')

        @render.ui
        def summary_stats():
            meta_filtered, meta_network, net_filtered = filtered_data()
            total_authors = len(meta_filtered)
            connected_authors = len(meta_network)
            total_collaborations = len(net_filtered)
            avg_publications = meta_filtered['Nb_publications'].mean() if len(meta_filtered) > 0 else 0
            avg_co_publications = net_filtered['Nb_co_publications'].mean() if len(net_filtered) > 0 else 0
            
            if len(meta_filtered) > 0:
                aff_counts = meta_filtered['Affiliation_clean'].value_counts()
                aff_html = "<br>".join([f"{aff}: {count}" for aff, count in aff_counts.items()])
            else:
                aff_html = "No data available"
            
            summary_html = f"""
            <div class="row">
                <div class="col-md-6">
                    <h5>Dataset Overview</h5>
                    <p><strong>Total Authors (filtered):</strong> {total_authors}</p>
                    <p><strong>Connected Authors:</strong> {connected_authors}</p>
                    <p><strong>Total Collaborations:</strong> {total_collaborations}</p>
                    <p><strong>Average Publications per Author:</strong> {avg_publications:.1f}</p>
                    <p><strong>Average Co-publications:</strong> {avg_co_publications:.1f}</p>
                </div>
                <div class="col-md-6">
                    <h5>Affiliation Distribution</h5>
                    <p>{aff_html}</p>
                </div>
            </div>
            """
            return ui.HTML(summary_html)

        @render.download(filename="network_data.csv")
        def download_net_data():
            _, _, net_df = filtered_data()
            return net_df.to_csv(index=False)

        @render.download(filename="meta_data.csv")
        def download_meta_data():
            meta_df, _, _ = filtered_data()
            return meta_df.to_csv(index=False)

        @render.download(filename="network_plot.html")
        def download_plot():
            meta_filtered, meta_network, net_df = filtered_data()
            if len(meta_network) == 0 or len(net_df) == 0:
                return "<html><body><h1>No data to display</h1></body></html>"
            
            # Recreate the plot for download (same logic as network_plot)
            G = nx.Graph()
            for _, row in meta_network.iterrows():
                G.add_node(row['Author'], 
                          nb_pubs=row['Nb_publications'],
                          affiliation=row['Affiliation_clean'],
                          keywords=row['Keywords'])
            
            for _, row in net_df.iterrows():
                if row['Author1'] in G.nodes and row['Author2'] in G.nodes:
                    G.add_edge(row['Author1'], row['Author2'], 
                              weight=row['Nb_co_publications'])
            
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Edge traces
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Node traces
            node_x = []
            node_y = []
            node_info = []
            node_colors = []
            colors = ["gold", "#004949", "#009292", "#ff6db6", "#ffb6db",
                     "#490092", "#006ddb", "#b66dff", "#6db6ff", "#b6dbff",
                     "#920000", "#924900", "#db6d00", "#24ff24", "#ffff6d"]
            
            unique_affs = meta_network['Affiliation_clean'].unique()
            aff_colors = {aff: colors[i % len(colors)] for i, aff in enumerate(unique_affs)}
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_data = meta_network[meta_network['Author'] == node].iloc[0]
                info = f"Author: {node}<br>Affiliation: {node_data['Affiliation_clean']}<br>Publications: {node_data['Nb_publications']}"
                node_info.append(info)
                node_colors.append(aff_colors[node_data['Affiliation_clean']])
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[node.split()[-1] for node in G.nodes()],
                textposition="middle center",
                hovertext=node_info,
                marker=dict(
                    showscale=False,
                    color=node_colors,
                    size=20,
                    line=dict(width=2, color='black')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title=f'Network Analysis - {len(G.nodes())} authors, {len(G.edges())} collaborations',
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               plot_bgcolor='white',
                               height=600
                           ))
            
            return fig.to_html(include_plotlyjs=True)

        # --------------------------
        # Institution Analysis Functions
        # --------------------------
        @reactive.Calc
        def filtered_institutions():
            selected = input.ia_institution_input()
            return selected if selected and all(s in affiliations_inst for s in selected) else []

        @render.ui
        def ia_keyword_plot():
            selected_institutions = filtered_institutions()
            if not selected_institutions:
                return ui.HTML(
                    '<div style="text-align: center; padding-top: 300px;">'
                    'Please select at least one valid institution'
                    '</div>'
                )
            
            filtered_data = kwss2_inst[kwss2_inst['Affiliations'].isin(selected_institutions)].copy()
            n_institutions = len(selected_institutions)
            cols = min(3, n_institutions)
            rows = (n_institutions + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=selected_institutions,
                specs=[[{"type": "xy"}] * cols for _ in range(rows)]
            )
            
            for i, institution in enumerate(selected_institutions):
                row = i // cols + 1
                col = i % cols + 1
                inst_data = filtered_data[filtered_data['Affiliations'] == institution]
                inst_data = inst_data.nlargest(10, 'Count')
                
                fig.add_trace(
                    go.Bar(
                        x=inst_data['Count'],
                        y=inst_data['Keywords'].str.title(),
                        orientation='h',
                        name=institution,
                        showlegend=False,
                        marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Top Keywords by Institution",
                **PLOT_CONFIG
            )
            return ui.HTML(pio.to_html(fig, include_plotlyjs='cdn'))

        @render.ui
        def ia_publication_plot():
            selected_institutions = filtered_institutions()
            if not selected_institutions:
                return ui.HTML(
                    '<div style="text-align: center; padding-top: 300px;">'
                    'Please select at least one valid institution'
                    '</div>'
                )
            
            filtered_data = []
            for institution in selected_institutions:
                inst_data = meta_data_inst[meta_data_inst['Affiliation'] == institution]
                top_authors = inst_data.nlargest(7, 'Nb_publications')
                if not top_authors.empty:
                    filtered_data.append(top_authors)
            
            if not filtered_data:
                return ui.HTML(
                    '<div style="text-align: center; padding-top: 300px;">'
                    'No publication data available'
                    '</div>'
                )
            
            plot_data = pd.concat(filtered_data)
            n_institutions = len(selected_institutions)
            cols = min(3, n_institutions)
            rows = (n_institutions + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=selected_institutions,
                specs=[[{"type": "xy"}] * cols for _ in range(rows)]
            )
            
            for i, institution in enumerate(selected_institutions):
                row = i // cols + 1
                col = i % cols + 1
                inst_data = plot_data[plot_data['Affiliation'] == institution]
                
                fig.add_trace(
                    go.Bar(
                        x=inst_data['Nb_publications'],
                        y=inst_data['Author'],
                        orientation='h',
                        name=institution,
                        showlegend=False,
                        marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Top Authors by Number of Publications",
                **PLOT_CONFIG
            )
            return ui.HTML(pio.to_html(fig, include_plotlyjs='cdn'))

        @render.ui
        def ia_collaboration_plot():
            selected_institutions = filtered_institutions()
            if not selected_institutions:
                return ui.HTML(
                    '<div style="text-align: center; padding-top: 300px;">'
                    'Please select at least one valid institution'
                    '</div>'
                )
            
            collaboration_data = []
            for institution in selected_institutions:
                inst_authors = meta_data_inst[meta_data_inst['Affiliation'] == institution][['Author', 'Nb_publications']]
                inst_net_data = net_data_inst[
                    net_data_inst['Author1'].isin(inst_authors['Author']) & 
                    net_data_inst['Author2'].isin(inst_authors['Author'])
                ]
                
                for _, row in inst_net_data.iterrows():
                    author1_pubs = inst_authors[inst_authors['Author'] == row['Author1']]['Nb_publications'].iloc[0]
                    author2_pubs = inst_authors[inst_authors['Author'] == row['Author2']]['Nb_publications'].iloc[0]
                    collab_str = f"{row['Author1']} -- {row['Author2']}" if author1_pubs >= author2_pubs else f"{row['Author2']} -- {row['Author1']}"
                    
                    collaboration_data.append({
                        'Institution': institution,
                        'Collaboration': collab_str,
                        'Nb_co_publications': row['Nb_co_publications']
                    })
            
            if not collaboration_data:
                return ui.HTML(
                    '<div style="text-align: center; padding-top: 300px;">'
                    'No collaboration data available'
                    '</div>'
                )
            
            collab_df = pd.DataFrame(collaboration_data)
            top_collaborations = []
            for institution in selected_institutions:
                inst_collabs = collab_df[collab_df['Institution'] == institution]
                top_7 = inst_collabs.nlargest(7, 'Nb_co_publications')
                if not top_7.empty:
                    top_collaborations.append(top_7)
            
            if not top_collaborations:
                return ui.HTML(
                    '<div style="text-align: center; padding-top: 300px;">'
                    'No collaboration data available'
                    '</div>'
                )
            
            plot_data = pd.concat(top_collaborations)
            n_institutions = len(selected_institutions)
            cols = min(3, n_institutions)
            rows = (n_institutions + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=selected_institutions,
                specs=[[{"type": "xy"}] * cols for _ in range(rows)]
            )
            
            for i, institution in enumerate(selected_institutions):
                row = i // cols + 1
                col = i % cols + 1
                inst_data = plot_data[plot_data['Institution'] == institution]
                
                fig.add_trace(
                    go.Bar(
                        x=inst_data['Nb_co_publications'],
                        y=inst_data['Collaboration'],
                        orientation='h',
                        name=institution,
                        showlegend=False,
                        marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Top Collaborations by Institution",
                **PLOT_CONFIG
            )
            return ui.HTML(pio.to_html(fig, include_plotlyjs='cdn'))

    return server
            