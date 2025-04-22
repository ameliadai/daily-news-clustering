"""
Pipeline: News Article Preprocessing - Vectorization - Clustering - Article Selection

To run:
python main_cython.py \
  --start_date 2025-03-01 \
  --end_date 2025-03-31 \
  --input_path ./news/news_2025_03_update.csv \
  --output_path ./result/cython
"""

import argparse
from utils_cython import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hot Topic Selection Pipeline")
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output data')
    
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