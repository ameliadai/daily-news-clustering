"""
Pipeline: News Article Preprocessing - Vectorization - Clustering - Article Selection

To run:
python main_pyopt.py \
  --start_date 2025-03-01 \
  --end_date 2025-03-03 \
  --input_path ./news/news_2025_03.csv \
  --output_path ./result/pyopt
"""

import time

from collections import defaultdict

# plotting
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
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


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        output = func(*args, **kwargs)
        end = time.perf_counter()
        return end-start, output

    return wrapper


@time_func
def vectorize_text(news_text, n_components=100):
    """
    TF-IDF vectorization and dimension reducation of text data.
    """
    print("Preprocessing text data...")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    ts = time.perf_counter()
    preprocessed_text = []
    for text in news_text:
        tokens = nltk.word_tokenize(text)
        tokens = [re.sub(r'[^a-zA-Z]', '', token.lower()) for token in tokens]
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        preprocessed_text.append(' '.join(tokens))
    pre_time = time.perf_counter() - ts
    
    print("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_text)
    
    if n_components and len(news_text) >= n_components:
        print("Reducing dimensionality with PCA...")
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X.toarray())
    
    return X, pre_time


@time_func
def select_top_articles(data, labels, X, topk, avg_distance_threshold=0.6):
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
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
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
    plot_time, figs = plot_cluster(valid_cluster_inds, X, data, topk)
    
    # Sort valid clusters by size (descending)
    valid_clusters = sorted(valid_clusters, key=lambda x: x[1], reverse=True)
    
    # Sanity check
    print(f"--- Titles in Top-{topk} Valid Clusters ---")
    for i, (cluster_id, size, avg_distance) in enumerate(valid_clusters[:topk], start=1):
        cluster_indices = valid_cluster_inds[cluster_id]
        cluster_titles = data.iloc[cluster_indices]['title'].tolist()
        print(f"\nCluster {i} (ID: {cluster_id}, Size: {size}, Avg Distance: {avg_distance:.4f}) Titles:")
        for title in cluster_titles:
            print(f"- {title}")
    print("--------------------------------------\n")
    
    selected_indices = set()
    selected_articles = []
    
    # Pick one random article from each of the top-K valid clusters
    for cluster_id, _, _ in valid_clusters[:topk]:
        idx = random.choice(clusters[cluster_id])
        selected_indices.add(idx)
        selected_articles.append(data.iloc[[idx]])
    
    # Add random articles if fewer than topk are selected
    while len(selected_articles) < topk:
        idx = random.randint(0, len(data) - 1)
        if idx not in selected_indices:
            selected_indices.add(idx)
            selected_articles.append(data.iloc[[idx]])
    
    return pd.concat(selected_articles, ignore_index=True), figs, plot_time


@time_func
def plot_cluster(valid_cluster_inds, X, data, topk):
    """
    Plot Cluster Labels in 2D
    Plot Cluster WordClouds
    """
    # reduce to 2D
    perplexity = len(X)-1 if len(X) < 31.0 else 30.0
    x_2d = TSNE(n_components=2, random_state=0, perplexity=perplexity).fit_transform(X)

    # UMAP might be faster for larger datasets
    # x_2d = umap.UMAP().fit_transform(X)

    # plot 2D with cluster labels
    colors = plt.cm.rainbow(np.linspace(0, 1, len(valid_cluster_inds)))
    cluster_fig, ax = plt.subplots(figsize=(8,5))
    ax.set_title('UMAP Cluster Visualization')
    i = 0
    for label, indices in valid_cluster_inds.items():
        ax.scatter(x_2d[indices, 0], x_2d[indices, 1], color=colors[i], label=i+1)
        i += 1
    ax.legend(loc='center right', bbox_to_anchor = (1.0, 0.5))

    # plot word cloud (article titles and texts)
    top_valid = sorted(valid_cluster_inds, 
                        key=lambda x: len(valid_cluster_inds[x]),
                        reverse=True)[:topk]
    wc_title_fig, axs_title = plt.subplots(1, topk,
                                           figsize=(10*topk, 5))
    wc_text_fig, axs_text = plt.subplots(1, topk,
                                         figsize=(10*topk, 5))
    stop = stopwords.words('english')
    for i in range(topk):
        cluster_data = data.iloc[valid_cluster_inds[top_valid[i]]]
        titles = cluster_data.title
        texts = cluster_data.text

        wc_title = WordCloud(
            background_color='white',
            stopwords=stop,
            min_font_size=10,
            width=1000,
            height=600
        ).generate(' '.join(titles).lower())
        wc_text = WordCloud(
            background_color='white',
            stopwords=stop,
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
def process_date(date, data, output_path, topk, eps=0.5, min_samples=3, k=5):
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

    ts = time.perf_counter()
    print("Clustering text data using DBSCAN...")
    '''
    if X.shape[0] < k:
        return 0.5, 2  # Default fallback for small datasets
    
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    sorted_distances = np.sort(distances[:, k - 1], axis=0)
    
    eps = np.percentile(sorted_distances, 90)
    min_samples = max(2, int(np.log(len(X))))
    
    print(f"Adjusted DBSCAN Params → eps: {eps:.4f}, min_samples: {min_samples}")
    '''
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    dbscan.fit(X)
    labels = dbscan.labels_
    cluster_time = time.perf_counter() - ts
    
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
        topk,
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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hot Topic Selection Pipeline")
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    parser.add_argument('--topk', type=int, help='Number of articles to select', default=3)

    args = parser.parse_args()

    total_ts = time.perf_counter()

    # loading data
    ts = time.perf_counter()
    data = pd.read_csv(args.input_path)
    data.drop_duplicates(subset=['text'], keep='first', inplace=True)
    data.drop_duplicates(subset=['title'], keep='first', inplace=True)
    # data = data[data['text'].apply(len) > 800]
    # data = data[data['text'].apply(len) < 10000]
    # data = data[data['title'].str.contains('Opinion:') == False]
    data = data.reset_index(drop = True)
    load_time = time.perf_counter() - ts
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # processing dates
    process_date_times = []
    preprocess_times = []
    vectorize_times = []
    cluster_times = []
    select_topk_times = []
    plot_times = []
    for current_date in tqdm(date_range, desc="Processing Dates"):
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing date: {date_str}")

        process_time, times = process_date(date_str, data, 
                                           args.output_path, args.topk) 

        process_date_times.append(process_time)
        preprocess_times.append(times[0])
        vectorize_times.append(times[1])
        cluster_times.append(times[2])
        select_topk_times.append(times[3])
        plot_times.append(times[4])
    
    total_time = time.perf_counter() - total_ts

    process_time = sum(process_date_times)
    preprocess_time = sum(preprocess_times)
    vec_time = sum(vectorize_times)
    cluster_time = sum(cluster_times)
    select_time = sum(select_topk_times)
    plot_time = sum(plot_times)

    with open(args.output_path + "/times.txt", "w") as f:
        print(f'Elapsed Time = {total_time} s', file=f)
        print(f'-- Loading Data = {load_time} s | {load_time/total_time:.3%} total', file=f)
        print(f'-- Processing Dates = {process_time} s | {process_time/total_time:.3%} total', file=f)
        print(f'   -- Vectorizing Texts = {vec_time} s | {vec_time/process_time:.3%} Processing Dates', file=f)
        print(f'      -- Pre-processing Texts = {preprocess_time} s | {preprocess_time/vec_time:.3%} Vectorizing Texts', file=f)
        print(f'   -- Clustering Texts = {cluster_time} s | {cluster_time/process_time:.3%} Processing Dates', file=f)
        print(f'   -- Selecting Top Articles = {select_time} s | {select_time/process_time:.3%} Processing Dates', file=f)
        print(f'      -- Plotting = {plot_time} s | {plot_time/select_time:.3%} Selecting Top Articles\n', file=f)