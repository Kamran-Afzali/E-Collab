import pandas as pd
import numpy as np
import random
from collections import defaultdict, Counter
from itertools import combinations
import re

def generate_unified_academic_data(n_papers=800, seed=42):
    """
    Generate a unified academic dataset where all components are connected
    based on a consistent set of publications.
    
    Parameters:
    - n_papers: Number of papers to generate
    - seed: Random seed for reproducibility
    
    Returns:
    - Dictionary containing all interconnected datasets
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Core reference data
    affiliations = [
        "McGill University", "ETS Montreal", "Concordia University", "University of Quebec",
        "Mila - Quebec AI Institute", "University of Montreal", "Polytechnique Montreal", 
        "HEC Montreal", "National Research Council Canada", "Private Research Lab",
        "Harvard University", "Stanford University", "MIT", "UC Berkeley",
        "Yale University", "Princeton University", "Columbia University"
    ]
    
    keywords = [
        "Active Learning", "Adversarial Learning", "Artificial Intelligence", "Autoencoders",
        "Bayesian Learning", "Computer Vision", "Convolutional Neural Networks", "Deep Learning",
        "Ensemble Methods", "Federated Learning", "Game Theory", "Graph Neural Networks",
        "Knowledge Graphs", "Machine Learning", "Machine Translation", "Meta Learning",
        "Multi-Agent Systems", "Natural Language Processing", "Neural Networks", "Object Detection",
        "Quantum Machine Learning", "Random Forest", "Reinforcement Learning", "Representation Learning",
        "Self-Supervised Learning", "Semi-Supervised Learning", "Sentiment Analysis", 
        "Support Vector Machine", "Transfer Learning", "Transformer Models", "Unsupervised Learning"
    ]
    
    first_names = [
        "John", "Sarah", "Michael", "Emily", "David", "Jessica", "Robert", "Ashley",
        "James", "Jennifer", "William", "Amanda", "Richard", "Lisa", "Thomas", "Michelle",
        "Christopher", "Patricia", "Daniel", "Susan", "Matthew", "Elizabeth", "Anthony", "Linda"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"
    ]
    
    # Generate unique authors with affiliations
    authors_data = {}
    used_names = set()
    n_authors = 150
    
    for i in range(n_authors):
        while True:
            first = random.choice(first_names)
            last = random.choice(last_names)
            full_name = f"{first} {last}"
            if full_name not in used_names:
                used_names.add(full_name)
                break
        
        # Assign affiliation and research profile
        affiliation = random.choice(affiliations)
        expertise_keywords = random.sample(keywords, random.randint(3, 8))
        productivity = random.randint(1, 50)  # Base productivity
        
        authors_data[full_name] = {
            'affiliation': affiliation,
            'expertise': expertise_keywords,
            'productivity': productivity,
            'papers': []
        }
    
    author_list = list(authors_data.keys())
    
    # Title templates for realistic paper titles
    title_templates = [
        "Deep Learning Approaches for {field} in {application}",
        "Novel {method} Techniques in {domain}",
        "Optimization of {algorithm} for {problem} Analysis",
        "A Comprehensive Study on {technique} Applications",
        "Advanced {method} for Real-time {application}",
        "{algorithm}-based {approach} in {field}",
        "Improving {technique} Performance using {method}",
        "Comparative Analysis of {method1} and {method2} in {domain}",
        "Automated {process} with {technique} and {algorithm}",
        "Scalable {method} for Large-scale {application}"
    ]
    
    fields = ["Healthcare", "Finance", "Robotics", "Cybersecurity", "Education", "Transportation", "Climate"]
    domains = ["Medical Imaging", "Financial Markets", "Social Networks", "IoT", "Smart Cities", "Autonomous Systems"]
    applications = ["Image Recognition", "Text Analysis", "Prediction", "Classification", "Optimization", "Detection"]
    
    # Generate papers with realistic author collaborations
    papers_data = []
    author_collaborations = defaultdict(list)
    author_paper_counts = defaultdict(int)
    
    for paper_id in range(n_papers):
        # Select lead author based on productivity
        weights = [authors_data[author]['productivity'] for author in author_list]
        lead_author = random.choices(author_list, weights=weights)[0]
        
        # Determine number of co-authors (most papers have 2-5 authors)
        n_coauthors = min(random.choices([0, 1, 2, 3, 4, 5, 6], weights=[5, 15, 25, 25, 15, 10, 5])[0], len(author_list)-1)
        
        # Select co-authors preferentially from same affiliation or similar expertise
        potential_coauthors = [a for a in author_list if a != lead_author]
        if n_coauthors > 0:
            # Bias towards same affiliation (50% chance) or similar keywords
            same_affil = [a for a in potential_coauthors 
                         if authors_data[a]['affiliation'] == authors_data[lead_author]['affiliation']]
            similar_expertise = [a for a in potential_coauthors 
                               if len(set(authors_data[a]['expertise']) & set(authors_data[lead_author]['expertise'])) > 0]
            
            if same_affil and random.random() < 0.3:
                coauthors = random.sample(same_affil, min(n_coauthors, len(same_affil)))
            elif similar_expertise and random.random() < 0.4:
                coauthors = random.sample(similar_expertise, min(n_coauthors, len(similar_expertise)))
            else:
                coauthors = random.sample(potential_coauthors, n_coauthors)
        else:
            coauthors = []
        
        all_authors = [lead_author] + coauthors
        
        # Generate paper keywords based on authors' expertise
        paper_keywords = set()
        for author in all_authors:
            # Each author contributes some of their expertise keywords
            contrib_keywords = random.sample(
                authors_data[author]['expertise'], 
                random.randint(1, min(3, len(authors_data[author]['expertise'])))
            )
            paper_keywords.update(contrib_keywords)
        
        # Add some random keywords (interdisciplinary work)
        if random.random() < 0.3:
            paper_keywords.update(random.sample(keywords, random.randint(1, 2)))
        
        paper_keywords = list(paper_keywords)[:6]  # Limit to 6 keywords max
        
        # Generate realistic title
        template = random.choice(title_templates)
        title = template.format(
            field=random.choice(fields),
            method=random.choice([k for k in paper_keywords if 'Learning' in k or 'Network' in k] or keywords),
            domain=random.choice(domains),
            application=random.choice(applications),
            algorithm=random.choice([k for k in paper_keywords if any(x in k for x in ['Forest', 'SVM', 'Neural'])] or keywords),
            problem=random.choice(['Pattern Recognition', 'Anomaly Detection', 'Clustering', 'Classification']),
            technique=random.choice(paper_keywords or keywords),
            approach=random.choice(['Supervised Learning', 'Unsupervised Learning', 'Reinforcement Learning']),
            process=random.choice(['Feature Selection', 'Data Preprocessing', 'Model Selection']),
            method1=random.choice(paper_keywords or keywords),
            method2=random.choice(paper_keywords or keywords)
        )
        
        # Store paper data
        paper_data = {
            'paper_id': paper_id,
            'title': title,
            'authors': all_authors,
            'lead_author': lead_author,
            'keywords': paper_keywords,
            'affiliations': [authors_data[a]['affiliation'] for a in all_authors],
            'year': random.randint(2015, 2024)
        }
        
        papers_data.append(paper_data)
        
        # Update author records
        for author in all_authors:
            authors_data[author]['papers'].append(paper_id)
            author_paper_counts[author] += 1
        
        # Track collaborations
        if len(all_authors) > 1:
            for author_pair in combinations(all_authors, 2):
                author_collaborations[tuple(sorted(author_pair))].append(paper_id)
    
    # Generate all output datasets
    
    # 1. Publications dataset (similar to data_smltar)
    publications_df = []
    for paper in papers_data:
        main_author = paper['lead_author']
        main_affiliation = authors_data[main_author]['affiliation']
        publications_df.append({
            'TI': paper['title'],
            'AU': main_author,
            'Author': main_author,
            'DE': '; '.join(paper['keywords']),
            'Affiliation_clean': main_affiliation,
            'Year': paper['year'],
            'All_Authors': '|'.join(paper['authors'])
        })
    data_smltar = pd.DataFrame(publications_df)
    
    # 2. Author metadata
    meta_data_list = []
    for author, data in authors_data.items():
        nb_pubs = len(data['papers'])
        keywords_str = '|'.join(data['expertise'])
        meta_data_list.append({
            'Author': author,
            'Nb_publications': nb_pubs,
            'Affiliation_clean': data['affiliation'],
            'Keywords': keywords_str
        })
    meta_data = pd.DataFrame(meta_data_list)
    
    # 3. Collaboration network data
    net_data_list = []
    for (author1, author2), shared_papers in author_collaborations.items():
        net_data_list.append({
            'Author1': author1,
            'Author2': author2,
            'Nb_co_publications': len(shared_papers),
            'Shared_papers': shared_papers
        })
    net_data = pd.DataFrame(net_data_list)
    
    # 4. Author-keyword matrix (for recommender system)
    author_keyword_matrix = defaultdict(lambda: defaultdict(float))
    for paper in papers_data:
        for author in paper['authors']:
            for keyword in paper['keywords']:
                author_keyword_matrix[author][keyword] += 1.0
    
    # Normalize by author's total publications and add some noise
    data_recommender_list = []
    for author in author_list:
        author_row = {'Author': author}
        total_pubs = len(authors_data[author]['papers'])
        for keyword in keywords:
            if keyword in author_keyword_matrix[author]:
                # Normalize and add small random component
                score = (author_keyword_matrix[author][keyword] / max(total_pubs, 1)) + random.uniform(0, 0.1)
                author_row[keyword] = round(min(score, 1.0), 3)
            else:
                # Small random score for keywords not directly associated
                author_row[keyword] = round(random.uniform(0, 0.05), 3)
        data_recommender_list.append(author_row)
    data_recommender = pd.DataFrame(data_recommender_list)
    
    # 5. Collaboration frequency data (for word cloud style visualization)
    collabss_data = []
    for author in author_list:
        collaborator_counts = Counter()
        for (a1, a2), shared_papers in author_collaborations.items():
            if author == a1:
                collaborator_counts[a2] += len(shared_papers)
            elif author == a2:
                collaborator_counts[a1] += len(shared_papers)
        
        for collaborator, freq in collaborator_counts.items():
            collabss_data.append({
                'Authors': author,
                'word': collaborator,
                'freq': freq
            })
    Collabss_b = pd.DataFrame(collabss_data)
    
    # 6. Author-keyword edges (bipartite network)
    edge_data = []
    for author in author_list:
        for keyword in keywords:
            if keyword in author_keyword_matrix[author]:
                weight = author_keyword_matrix[author][keyword] / max(len(authors_data[author]['papers']), 1)
                if weight > 0.1:  # Only include significant associations
                    edge_data.append({
                        'from': author,
                        'to': keyword,
                        'weight': round(weight, 3)
                    })
    edge_2 = pd.DataFrame(edge_data)
    
    # 7. Institution-keyword analysis
    kwss2_data = []
    affiliation_keywords = defaultdict(Counter)
    for paper in papers_data:
        for affiliation in set(paper['affiliations']):
            for keyword in paper['keywords']:
                affiliation_keywords[affiliation][keyword] += 1
    
    for affiliation, keyword_counts in affiliation_keywords.items():
        for keyword, count in keyword_counts.items():
            kwss2_data.append({
                'Affiliations': affiliation,
                'Keywords': keyword,
                'Count': count
            })
    kwss2 = pd.DataFrame(kwss2_data)
    
    # Return all datasets in a structured format
    return {
        'publications': data_smltar,
        'author_metadata': meta_data,
        'collaboration_network': net_data,
        'author_keyword_matrix': data_recommender,
        'collaboration_frequency': Collabss_b,
        'author_keyword_edges': edge_2,
        'institution_keywords': kwss2,
        'raw_papers': papers_data,
        'authors_info': authors_data,
        'reference_data': {
            'affiliations': affiliations,
            'keywords': keywords,
            'authors': author_list
        }
    }

unified_data = generate_unified_academic_data(n_papers=800, seed=42)
data_smltar = unified_data['publications']
meta_data = unified_data['author_metadata'] 
meta_data_inst = unified_data['author_metadata'] 
net_data_net = unified_data['collaboration_network']
net_data_inst = unified_data['collaboration_network']
data_recommender = unified_data['author_keyword_matrix']
Collabss_b = unified_data['collaboration_frequency']
edge_2 = unified_data['author_keyword_edges']
kwss2_inst = unified_data['institution_keywords']
affs_net = unified_data['reference_data']['affiliations']    
affiliations_inst = unified_data['reference_data']['affiliations']
keywords_list_net = unified_data['reference_data']['keywords']
my_autocomplete_list = unified_data['reference_data']['keywords']
author_list = unified_data['reference_data']['authors']
    
