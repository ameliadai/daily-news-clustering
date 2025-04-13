"""
Pipeline: News Article Preprocessing - Vectorization - Clustering - Article Selection

To run:
python main.py \
  --start_date 2025-03-01 \
  --end_date 2025-03-03 \
  --input_path ./news/news_2025_03.csv \
  --output_path ./result
"""

import time

# plotting
from sklearn.manifold import TSNE
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
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import Counter


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Constants
DEFAULT_NUM_ARTICLES = 3

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    df.drop_duplicates(subset=['title'], keep='first', inplace=True)
    # df = df[df['text'].apply(len) > 800]
    # df = df[df['text'].apply(len) < 10000]
    # df = df[df['title'].str.contains('Opinion:') == False]
    df = df.reset_index(drop = True)
    return df

def preprocess_text(text):
    """Clean and preprocess text data."""
    tokens = nltk.word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z]', '', token.lower()) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def vectorize_text(news_text, n_components=100):
    """
    TF-IDF vectorization and dimension reducation of text data.
    """
    print("Preprocessing text data...")
    start = time.perf_counter()
    preprocessed_text = [preprocess_text(text) for text in news_text]
    end = time.perf_counter()
    pre_time = end-start
    
    print("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_text)
    
    if n_components and len(news_text) >= n_components:
        print("Reducing dimensionality with PCA...")
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X.toarray())
    
    return X, pre_time


def adjust_dbscan_params(X, k=5):
    """
    Adjust DBSCAN parameters `eps` and `min_samples` based on the dataset X.
    """
    if X.shape[0] < k:
        return 0.5, 2  # Default fallback for small datasets
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    sorted_distances = np.sort(distances[:, k - 1], axis=0)
    
    eps = np.percentile(sorted_distances, 90)
    min_samples = max(2, int(np.log(len(X))))
    
    print(f"Adjusted DBSCAN Params → eps: {eps:.4f}, min_samples: {min_samples}")
    return eps, min_samples


def cluster_texts(X, eps=0.5, min_samples=3):
    """
    Cluster vectorized news articles using DBSCAN.
    """
    print("Clustering text data using DBSCAN...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    dbscan.fit(X)
    return dbscan.labels_


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

    # plot valid clusters
    start = time.perf_counter()
    cluster_fig = plot_cluster(valid_cluster_inds, X, data)
    end = time.perf_counter()
    plot_time = end-start
    
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
    
    return pd.concat(selected_articles, ignore_index=True), cluster_fig, plot_time

def plot_cluster(valid_cluster_inds, X, data):
    """
    Plot Cluster Labels in 2D
    Plot Cluster WordClouds
    """
    # reduce to 2D
    perplexity = len(X)-1 if len(X) < 31.0 else 30.0
    x_2d = TSNE(n_components=2, random_state=0, perplexity=perplexity).fit_transform(X)

    # plot 2D with cluster labels
    colors = plt.cm.rainbow(np.linspace(0, 1, len(valid_cluster_inds)))
    cluster_fig, ax = plt.subplots(figsize=(8,5))
    ax.set_title('TSNE Cluster Visualization')
    i = 0
    for label, indices in valid_cluster_inds.items():
        ax.scatter(x_2d[indices, 0], x_2d[indices, 1], color=colors[i], label=i+1)
        i += 1
    ax.legend(loc='center right', bbox_to_anchor = (1.0, 0.5))
    return cluster_fig

def process_date(date, data, output_path):
    """
    Process and save selected articles for a specific date.
    """
    daily_data = data[data['date'] == date]
    if daily_data.empty:
        print(f"No data for date: {date}")
        return
    
    news_text = daily_data['text'].tolist()
    start = time.perf_counter()
    X, pre_time = vectorize_text(news_text)
    end = time.perf_counter()
    vec_time = end-start
    # eps, min_samples = adjust_dbscan_params(X)
    eps, min_samples = 0.5, 3

    start = time.perf_counter()
    labels = cluster_texts(X, eps=eps, min_samples=min_samples)
    end = time.perf_counter()
    cluster_time = end-start
    
    # Sanity check - cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_points = list(labels).count(-1)
    n_grouped_points = len(labels) - n_noise_points
    total_samples = len(labels)
    
    print(f"\n--- Cluster Statistics ---")
    print(f"Number of clusters: {n_clusters}")
    print(f"Total number of samples: {total_samples}")
    print(f"Number of grouped points: {n_grouped_points}")
    print(f"Number of noise points: {n_noise_points}")
    print("---------------------------\n")
    
    start = time.perf_counter()
    selected_articles, cluster_fig, plot_time = select_top_articles(
        daily_data,
        labels,
        X,
        avg_distance_threshold=0.7
    )
    end = time.perf_counter()
    select_time = end-start
    
    save_path = os.path.join(output_path, date)
    os.makedirs(save_path, exist_ok=True)
    selected_articles.to_csv(os.path.join(save_path, 'articles_selected.csv'), index=False)
    cluster_fig.savefig(os.path.join(save_path, 'clusters.png'))
    print(f"Articles for {date} saved successfully!")

    return pre_time, vec_time, cluster_time, select_time, plot_time


def main(args):
    start = time.perf_counter()
    data = load_data(args.input_path)
    end = time.perf_counter()
    load_time = end-start
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    date_range = pd.date_range(start=start_date, end=end_date)
    
    process_date_times = []
    preprocess_times = []
    vectorize_times = []
    cluster_times = []
    select_topk_times = []
    plot_times = []
    for current_date in tqdm(date_range, desc="Processing Dates"):
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing date: {date_str}")

        start = time.perf_counter()
        times = process_date(date_str, data, args.output_path) 
        end = time.perf_counter()

        process_date_times.append(end-start)
        preprocess_times.append(times[0])
        vectorize_times.append(times[1])
        cluster_times.append(times[2])
        select_topk_times.append(times[3])
        plot_times.append(times[4])


    return (load_time,
            sum(process_date_times),
            sum(preprocess_times),
            sum(vectorize_times),
            sum(cluster_times),
            sum(select_topk_times),
            sum(plot_times))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hot Topic Selection Pipeline")
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    
    args = parser.parse_args()
    start = time.perf_counter()
    times = main(args)
    end = time.perf_counter()
    total_time = end-start

    with open(args.output_path + "/times.txt", "a") as f:
        print(f'Elapsed Time = {total_time} ns', file=f)
        print(f'-- Loading Data = {times[0]} ns | {times[0]/total_time:.3%} total', file=f)
        print(f'-- Processing Dates = {times[1]} ns | {times[1]/total_time:.3%} total', file=f)
        print(f'   -- Vectorizing Texts = {times[3]} ns | {times[3]/times[1]:.3%} Processing Dates', file=f)
        print(f'      -- Pre-processing Texts = {times[2]} ns | {times[2]/times[3]:.3%} Vectorizing Texts', file=f)
        print(f'   -- Clustering Texts = {times[4]} ns | {times[4]/times[1]:.3%} Processing Dates', file=f)
        print(f'   -- Selecting Top Articles = {times[5]} ns | {times[5]/times[1]:.3%} Processing Dates', file=f)
        print(f'      -- Plotting = {times[6]} ns | {times[6]/times[5]:.3%} Selecting Top Articles\n', file=f)