import time

# plotting
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import itertools

import argparse
import os
import random
import re
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import Counter
from itertools import product



nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Constants
DEFAULT_NUM_ARTICLES = 3

def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        output = func(*args, **kwargs)
        end = time.perf_counter()
        return end-start, output

    return wrapper

def hyperparameter_sweep(X, eps_values, min_samples_values):
    """
    Sweep over DBSCAN parameters and return results.
    """
    results = []

    for eps, min_samples in product(eps_values, min_samples_values):
        try:
            time, labels = cluster_texts(X, eps=eps, min_samples=min_samples)
            val_ind = np.where(labels != -1)[0]
            if len(set(labels)) > 1 and len(val_ind) > 1:
                score = silhouette_score(X[val_ind], labels[val_ind])
            else:
                score = -1  # Poor clustering
        except Exception as e:
            print(f"Error for eps={eps}, min_samples={min_samples}: {e}")
            score = -1

        results.append({
            'time': time, 
            'eps': eps,
            'min_samples': min_samples,
            'silhouette_score': score,
            'num_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'num_noise_points': list(labels).count(-1)
        })

    return results

@time_func
def preprocess_text(text):
    """Clean and preprocess text data."""
    tokens = nltk.word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', token.lower()) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

@time_func
def vectorize_text(news_text, n_components=100):
    """
    TF-IDF vectorization and dimension reducation of text data.
    """
    print("Preprocessing text data...")
    pre_time, preprocessed_text = 0, []
    for text in news_text:
        p_time, p_text = preprocess_text(text)
        pre_time += p_time
        preprocessed_text.append(p_text)
    
    print("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_text)
    
    if n_components and len(news_text) >= n_components:
        print("Reducing dimensionality with PCA...")
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X.toarray())
    
    return X, pre_time



@time_func
def cluster_texts(X, eps=0.5, min_samples=3):
    """
    Cluster vectorized news articles using DBSCAN.
    """
    print("Clustering text data using DBSCAN...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    dbscan.fit(X)
    return dbscan.labels_

@time_func
def select_top_articles(data, labels, X, avg_distance_threshold=0.6):
    """
    Select top articles - one from each top-k valid clusters with largest cluster size.
    Parameters:
        - data (pd.DataFrame): The article dataset.
        - labels (array): Cluster labels from DBSCAN.
        - X (array): Cluster data points.
        - avg_distance_threshold (float): Max average pairwise distance for valid clusters.
    Returns:
        - selected_articles (pd.DataFrame): Selected top articles.
    """
    # Group indices by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    valid_clusters = []
    valid_cluster_inds = {}
    # Check for validity of each cluster based on average pairwise distance
    for cluster_id, indices in clusters.items():
        if cluster_id == -1 or len(indices) < 2:
            continue  # Skip noise and tiny clusters
        
        cluster_points = X[indices]
        avg_distance = np.mean(pairwise_distances(cluster_points, metric='cosine'))
        
        # Only consider clusters with avg_distance <= avg_distance_threshold
        if avg_distance <= avg_distance_threshold:
            valid_clusters.append((cluster_id, len(indices), avg_distance))
            valid_cluster_inds[cluster_id] = indices
    
    # Sort valid clusters by size (descending)
    valid_clusters = sorted(valid_clusters, key=lambda x: x[1], reverse=True)
    
    # Sanity check
    print(f"--- Titles in Top-{DEFAULT_NUM_ARTICLES} Valid Clusters ---")
    for i, (cluster_id, size, avg_distance) in enumerate(valid_clusters[:DEFAULT_NUM_ARTICLES], start=1):
        cluster_indices = valid_cluster_inds[cluster_id]
        cluster_titles = data.iloc[cluster_indices]['title'].tolist()
        print(f"\nCluster {i} (ID: {cluster_id}, Size: {size}, Avg Distance: {avg_distance:.4f}) Titles:")
        for title in cluster_titles:
            print(f"- {title}")
    print("--------------------------------------\n")
    
    selected_indices = set()
    selected_articles = []
    
    # Pick one random article from each of the top-K valid clusters
    for cluster_id, _, _ in valid_clusters[:DEFAULT_NUM_ARTICLES]:
        idx = random.choice(clusters[cluster_id])
        selected_indices.add(idx)
        selected_articles.append(data.iloc[[idx]])
    
    # Add random articles if fewer than DEFAULT_NUM_ARTICLES are selected
    while len(selected_articles) < DEFAULT_NUM_ARTICLES:
        idx = random.randint(0, len(data) - 1)
        if idx not in selected_indices:
            selected_indices.add(idx)
            selected_articles.append(data.iloc[[idx]])
    
    return pd.concat(selected_articles, ignore_index=True)

@time_func
def process_date(date, data, output_path, sweep=False, eps_values=None, min_samples_values=None):
    """
    Process and save selected articles for a specific date.
    """
    daily_data = data[data['date'] == date]
    if daily_data.empty:
        print(f"No data for date: {date}")
        return
    
    news_text = daily_data['text'].tolist()
    vec_time, outputs = vectorize_text(news_text)
    X, pre_time = outputs

    if sweep and eps_values and min_samples_values:
        sweep_results = hyperparameter_sweep(X, eps_values, min_samples_values)
        results_df = pd.DataFrame(sweep_results)
        results_df.to_csv(os.path.join(output_path, date, 'dbscan_sweep_results.csv'), index=False)
        best_params = max(sweep_results, key=lambda x: x['silhouette_score'])
        eps, min_samples = best_params['eps'], best_params['min_samples']
        print(f"Best parameters for {date}: eps={eps}, min_samples={min_samples}")
    else:
        eps, min_samples = 0.5, 3

        cluster_time, labels = cluster_texts(X, eps=eps, min_samples=min_samples)
    
    # Sanity check - cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_points = list(labels).count(-1)
    n_grouped_points = len(labels) - n_noise_points
    total_samples = len(labels)
    val_ind = np.where(labels != -1)[0]
    sil_score = silhouette_score(X[val_ind], labels[val_ind])
    
    print(f"\n--- Cluster Statistics ---")
    print(f"Number of clusters: {n_clusters}")
    print(f"Total number of samples: {total_samples}")
    print(f"Number of grouped points: {n_grouped_points}")
    print(f"Number of noise points: {n_noise_points}")
    print(f"Mean Silhouette Coefficient: {sil_score:.3}")
    print("---------------------------\n")

@time_func
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    df.drop_duplicates(subset=['title'], keep='first', inplace=True)
    # df = df[df['text'].apply(len) > 800]
    # df = df[df['text'].apply(len) < 10000]
    # df = df[df['title'].str.contains('Opinion:') == False]
    df = df.reset_index(drop = True)
    return df


@time_func
def main(args):
    load_time, data = load_data(args.input_path)
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    date_range = pd.date_range(start=start_date, end=end_date)
    
    process_date_times = []
    preprocess_times = []
    vectorize_times = []
    cluster_times = []
    select_topk_times = []
    for current_date in tqdm(date_range, desc="Processing Dates"):
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing date: {date_str}")

        process_time, times = process_date(
            date_str,
            data,
            args.output_path,
            sweep=args.sweep,
            eps_values=args.eps_values,
            min_samples_values=args.min_samples_values
        )

        process_date_times.append(process_time)
        preprocess_times.append(times[0])
        vectorize_times.append(times[1])
        cluster_times.append(times[2])
        select_topk_times.append(times[3])


    return (load_time,
            sum(process_date_times),
            sum(preprocess_times),
            sum(vectorize_times),
            sum(cluster_times),
            sum(select_topk_times))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hot Topic Selection Pipeline")
    
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    parser.add_argument('--sweep', action='store_true', help='Run DBSCAN hyperparameter sweep')
    parser.add_argument('--eps_values', nargs='+', type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument('--min_samples_values', nargs='+', type=int, default=[2, 3, 5])
    
    args = parser.parse_args()
    total_time, times = main(args)

    with open(args.output_path + "/times.txt", "w") as f:
        print(f'Elapsed Time = {total_time} s', file=f)
        print(f'-- Loading Data = {times[0]} s | {times[0]/total_time:.3%} total', file=f)
        print(f'-- Processing Dates = {times[1]} s | {times[1]/total_time:.3%} total', file=f)
        print(f'   -- Vectorizing Texts = {times[3]} s | {times[3]/times[1]:.3%} Processing Dates', file=f)
        print(f'      -- Pre-processing Texts = {times[2]} s | {times[2]/times[3]:.3%} Vectorizing Texts', file=f)
        print(f'   -- Clustering Texts = {times[4]} s | {times[4]/times[1]:.3%} Processing Dates', file=f)
        print(f'   -- Selecting Top Articles = {times[5]} s | {times[5]/times[1]:.3%} Processing Dates', file=f)
        print(f'      -- Plotting = {times[6]} s | {times[6]/times[5]:.3%} Selecting Top Articles\n', file=f)