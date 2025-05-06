# daily-news-clustering

Amelia Dai, Dan Pechi, Marina Castaño Ishizaka, Simone Rittenhouse

This repository corresponds to the final project for the Spring 2025 semester of Advanced Python for Data Science (DS-GA 1019). The aim of this project is to automatically and efficiently track the world’s biggest news events every single day. To do so, we build a pipeline for automated news article clustering. Our algorithm takes in a date range and returns the top $k$ clusters of topics for each date in the range.

# Repository Contents

## Data Scraping

The data scraping and data processing scripts are found in the directory `scraping`. The final dataset covering March 2025 is `news_2025_03_update.csv` found in the `news` directory.

## Clustering

### Initial Implementation

Our initial clustering script is `main_timed.py` which takes input `news_2025_03_update.csv`. This script's output is:
- `result/times.txt`: Contains the execution times for each function in `main_timed.py` as well as the overall script runtime.
- `result/result_summary.csv`: Contains performance metrics for each date in the input range.
- For each subdirectory `result/2025-03-01` through `result/2025-03-31`:
    - `result/[DATE]/articles_selected.csv`: Contains one randomly selected news article from each of the final top $k$ clusters.
    - `result/[DATE]/clusters.png`: Contains a scatter plot of all the valid clusters found for that date.
    - `result/[DATE]/texts.png`: Contains a word cloud summarizing article texts for each of the final top $k$ clusters.
    - `result/[DATE]/titles.png`: Contains a word cloud summarizing article titles for each of the final top $k$ clusters.

**Note**: The notebook `plotting.ipynb` uses the results stored in `result/result_summary.csv` to create performance summary plots and metrics.

### Performance Optimization

The script `dbscan_params.py` optimizes our initial implmentation for clustering performance using DBSCAN hyperparameter tuning.

### Time Optimization

Our time optimization scripts each modify `main_timed.py` and are as follows:
  
|Strategy|Script|Output Directory|
|---|---|---|
|Python Optimization| `main_pyopt.py` | `result/pyopt/` |
|Numba | `main_numba.py` | `result/numba/` |
|Cython | `main_cython.py` with files `utils_cython.pyx`, `utils_cython.c`, `utils_cython.cpython-312-darwin.so`, and `setup_cython.py` | `result/cython/` |
|Multiprocessing | `main_mp.py` | `result/multiprocessing/` | 
|Threading | `main_thread.py` | `result/thread/` | 

For each of these scripts, the output is stored in the same format as the output of `main_timed.py` within their output subdirectories. Note that these time optimizations do not produce a `result_summary.csv` file as the clustering algorithm is left unchanged across the implementations.
