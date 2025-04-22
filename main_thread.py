"""
Pipeline: News Article Preprocessing - Vectorization - Clustering - Article Selection

To run:
python main_thread.py \
  --start_date 2025-03-01 \
  --end_date 2025-03-31 \
  --input_path ./news/news_2025_03_update.csv \
  --output_path ./result/thread
"""

import time

# threading
from threading import Thread
from threading import Lock
from functools import partial

# plotting
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
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

    # plot valid clusters
    plot_time, figs = plot_cluster(valid_cluster_inds, X, data)
    
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
    
    return pd.concat(selected_articles, ignore_index=True), figs, plot_time


@time_func
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

    # plot word cloud (article titles and texts)
    top_valid = sorted(valid_cluster_inds, 
                        key=lambda x: len(valid_cluster_inds[x]),
                        reverse=True)[:DEFAULT_NUM_ARTICLES]
    wc_title_fig, axs_title = plt.subplots(1, DEFAULT_NUM_ARTICLES,
                                           figsize=(10*DEFAULT_NUM_ARTICLES, 5))
    wc_text_fig, axs_text = plt.subplots(1, DEFAULT_NUM_ARTICLES,
                                         figsize=(10*DEFAULT_NUM_ARTICLES, 5))
    for i in range(DEFAULT_NUM_ARTICLES):
        titles = data.iloc[valid_cluster_inds[top_valid[i]]]['title']
        texts = data.iloc[valid_cluster_inds[top_valid[i]]]['text']

        wc_title = WordCloud(
            background_color='white',
            stopwords=stopwords.words('english'),
            min_font_size=10,
            width=1000,
            height=600
        ).generate(' '.join(titles).lower())
        wc_text = WordCloud(
            background_color='white',
            stopwords=stopwords.words('english'),
            min_font_size=10,
            width=1000,
            height=600
        ).generate(' '.join(texts).lower())

        axs_title[i].imshow(wc_title)
        axs_title[i].set_title(f'Titles of Cluster {top_valid[i]}',fontsize=20)
        axs_title[i].axis('off')

        axs_text[i].imshow(wc_text)
        axs_text[i].set_title(f'Articles of Cluster {top_valid[i]}',fontsize=20)
        axs_text[i].axis('off')

    return cluster_fig, wc_title_fig, wc_text_fig


@time_func
def process_date(date, data, output_path):
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

    # adjust_time, eps, min_samples = adjust_dbscan_params(X)
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
    
    select_time, outputs = select_top_articles(
        daily_data,
        labels,
        X,
        avg_distance_threshold=0.7
    )
    selected_articles, figs, plot_time = outputs
    
    save_path = os.path.join(output_path, date)
    os.makedirs(save_path, exist_ok=True)

    # saving articles
    selected_articles.to_csv(os.path.join(save_path, 'articles_selected.csv'), index=False)

    # saving plots
    cluster_fig, wc_title_fig, wc_text_fig = figs
    cluster_fig.savefig(os.path.join(save_path, 'clusters.png'))
    wc_title_fig.savefig(os.path.join(save_path, 'titles.png'), bbox_inches='tight')
    wc_text_fig.savefig(os.path.join(save_path, 'texts.png'), bbox_inches='tight')

    print(f"Articles for {date} saved successfully!")

    return pre_time, vec_time, cluster_time, select_time, plot_time


@time_func
def main(args):
    load_time, data = load_data(args.input_path)
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    date_range = pd.date_range(start=start_date, end=end_date)
    date_strs = [d.strftime("%Y-%m-%d") for d in date_range]

    # chunk dates
    N, date_N = args.N, len(date_strs)
    if N > len(date_strs):
        N = len(date_strs)

    date_strs = np.array_split(date_strs, N)
    print(date_strs, len(date_strs))
    
    process_date_times = []
    preprocess_times = []
    vectorize_times = []
    cluster_times = []
    select_topk_times = []
    plot_times = []

    func = partial(process_date, data=data, output_path=args.output_path)
    lock = Lock()
    def process_thread(dates):
        with lock:
            for date in dates:
                process_time, times = func(date)
                process_date_times.append(process_time)
                preprocess_times.append(times[0])
                vectorize_times.append(times[1])
                cluster_times.append(times[2])
                select_topk_times.append(times[3])
                plot_times.append(times[4])

    threads = []
    for i in range(N):
        worker = Thread(target=process_thread, args=(date_strs[i],), 
                        name='thread_' + str(i))
        worker.setDaemon(True) 
        worker.start()
        threads.append(worker)

    # join threads
    for thread in threads:
        thread.join()

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
    parser.add_argument('--N', type=int, default=10, help='Number of workers to use')
    
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