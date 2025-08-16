"""
Data generation functions for the Research Network Analysis Dashboard.
"""

import pandas as pd
import numpy as np
import random

def generate_network_data():
    """Generate synthetic data for network analysis."""
    affs = ["McGill", "ETS", "Concordia", "Other", "Univ Quebec", "Mila", 
            "Univ Montreal", "Private", "Polytech Montreal", "Natl Res Council Canada", "HEC Montreal"]
    
    # Full list of 64 keywords
    keywords_list = [
        "Active Learning", "Adaboost", "Adaptive Behavior", "Adversarial Agents",
        "Adversarial Attacks", "Adversarial Learning", "Adversarial Robustness",
        "Artificial intelligence", "Artificial intelligence bias", "Artificial intelligence equity",
        "Artificial intelligence Explainability", "Artificial intelligence fairness",
        "Autoencoders", "Bayesian Learning", "Bayesian Networks", "Behavioral Game Theory",
        "Constraint Learning", "Constraint Optimization", "Conversational AI",
        "convolutional neural network", "Cooperative Game Theory", "decision trees",
        "Deep Generative Models", "Deep learning", "Deep Neural Architectures",
        "Distributed Machine Learning", "Ensemble Methods", "Ensemble learning",
        "Ethics of AI", "Explainable artificial AI", "Feature Selection",
        "Federated Learning", "Game Theory", "Gradient boosting", "Graph neural networks",
        "Image recognition", "Inverse Reinforcement Learning", "K Nearest Neighbors",
        "Kernel Methods", "Knowledge Capture", "Knowledge Graphs", "Knowledge Reasoning",
        "Long Short Term Memory networks", "Machine learning", "Machine Translation",
        "Markov Models", "Meta Learning", "Metaheuristics", "Multi-class Classification",
        "Multi-instance Learning", "Multi-label Classification", "Multi-Task Learning",
        "Multiagent Learning", "Multiagent Systems", "Multimodal Learning",
        "Naive Bayes", "Natural language processing", "Neural network",
        "Object Detection", "Quantum Machine Learning", "random forest",
        "Recurrent neural network", "Reinforcement learning", "Representation Learning",
        "image segmentation", "Self-Supervised Learning", "Semi-Supervised Learning",
        "Sentiment Analysis", "Supervised learning", "Support vector machine",
        "Transfer Learning", "Unsupervised learning", "transformers model",
        "feedforward neural network", "generative adversarial networks"
    ]
    
    first_names = ["John", "Mary", "David", "Sarah", "Michael", "Lisa", "Robert", "Jennifer",
                   "William", "Patricia", "James", "Elizabeth", "Christopher", "Linda",
                   "Daniel", "Barbara", "Matthew", "Susan", "Anthony", "Jessica"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
                  "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
                  "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
    
    authors = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(100)]
    
    # Generate meta_data
    meta_data = []
    for author in authors:
        nb_pubs = random.randint(1, 100)
        # Safe sampling to prevent requesting more keywords than available
        n_keywords = min(5, len(keywords_list))
        k = random.randint(2, n_keywords)
        author_keywords = random.sample(keywords_list, k)
        keywords_str = "|".join(author_keywords)
        
        meta_data.append({
            'Author': author,
            'Nb_publications': nb_pubs,
            'Affiliation_clean': random.choice(affs),
            'Keywords': keywords_str
        })
    
    meta_df = pd.DataFrame(meta_data)
    
    # Generate net_data
    net_data = []
    for _ in range(300):
        author1, author2 = random.sample(authors, 2)
        nb_co_pubs = random.randint(1, 20)
        net_data.append({
            'Author1': author1,
            'Author2': author2,
            'Nb_co_publications': nb_co_pubs
        })
    
    net_df = pd.DataFrame(net_data)
    return meta_df, net_df, affs, keywords_list


def generate_institution_data():
    """Generate synthetic data for institution analysis."""
    np.random.seed(42)
    random.seed(42)
    affiliations = [
        "Harvard University", "Stanford University", "MIT", "UC Berkeley",
        "Yale University", "Princeton University", "Columbia University",
        "University of Chicago", "Caltech", "Johns Hopkins University"
    ]
    
    keywords = [
        "machine learning", "artificial intelligence", "deep learning", "neural networks",
        "data science", "computer vision", "natural language processing", "robotics",
        "quantum computing", "bioinformatics", "computational biology", "statistics",
        "optimization", "algorithms", "software engineering", "cybersecurity"
    ]
    
    # Generate keywords data
    kwss2_data = []
    for affil in affiliations:
        selected_keywords = random.sample(keywords, random.randint(8, 12))
        for keyword in selected_keywords:
            kwss2_data.append({
                'Affiliations': affil,
                'Keywords': keyword,
                'Count': random.randint(5, 50)
            })
    
    kwss2 = pd.DataFrame(kwss2_data)
    
    # Generate author data
    first_names = ["John", "Sarah", "Michael", "Emily", "David", "Jessica", "Robert", "Ashley",
               "James", "Jennifer", "William", "Amanda", "Richard", "Lisa", "Thomas", "Michelle"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas"]
    
    meta_data_list = []
    for affil in affiliations:
        num_authors = random.randint(15, 25)
        used_names = set()
        for _ in range(num_authors):
            while True:
                first = random.choice(first_names)
                last = random.choice(last_names)
                full_name = f"{first} {last}"
                if full_name not in used_names:
                    used_names.add(full_name)
                    break
            meta_data_list.append({
                'Author': full_name,
                'Affiliation': affil,
                'Nb_publications': random.randint(1, 100)
            })
    
    meta_data = pd.DataFrame(meta_data_list)
    
    # Generate collaboration data
    net_data_list = []
    for affil in affiliations:
        affil_authors = meta_data[meta_data['Affiliation'] == affil]['Author'].tolist()
        num_collaborations = random.randint(10, 20)
        for _ in range(num_collaborations):
            if len(affil_authors) >= 2:
                author1, author2 = random.sample(affil_authors, 2)
                net_data_list.append({
                    'Author1': author1,
                    'Author2': author2,
                    'Nb_co_publications': random.randint(1, 15)
                })
    
    net_data = pd.DataFrame(net_data_list)
    return kwss2, meta_data, net_data, affiliations
