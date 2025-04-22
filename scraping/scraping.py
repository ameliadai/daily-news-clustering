"""
We’re not optimizing this file because its performance can’t be reliably measured,
i.e., the total runtime fluctuates based on how many articles are available at any given time.
"""

import argparse
import datetime
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import newspaper
from newspaper import Article, Source


@dataclass
class NewsArticle:
    date: str
    text: str
    url: str
    title: str
    source: str


def build_source(url: str, language: str = 'en') -> Source:
    """
    a newspaper Source for the given url
    """
    return newspaper.build(url, language=language, memoize_articles=False)


def filter_by_date(articles: List[Article], date: datetime.date) -> List[Article]:
    """
    filter articles whose URL contains the given date (YYYY/MM/DD)
    """
    date_str = date.strftime('%Y/%m/%d')
    return [article for article in articles if date_str in article.url]


def download_and_parse(articles: List[Article]) -> List[Article]:
    """
    download and parse each Article, returning successfully processed ones
    """
    parsed: List[Article] = []
    for article in articles:
        try:
            article.download()
            article.parse()
            parsed.append(article)
        except Exception as e:
            logging.warning(f"Failed to process {article.url}: {e}")
    return parsed


def to_news_article(article: Article, source_url: str) -> NewsArticle:
    """
    convert a newspaper Article into a NewsArticle dataclass
    """
    date_str = (
        article.publish_date.strftime('%Y-%m-%d')
        if article.publish_date else ''
    )
    return NewsArticle(
        date=date_str,
        text=article.text,
        url=article.url,
        title=article.title,
        source=source_url,
    )


def scrape_source(url: str, date: datetime.date) -> List[NewsArticle]:
    """
    run the full scraping pipeline for one source URL.
    """
    source = build_source(url)
    candidates = filter_by_date(source.articles, date)
    parsed = download_and_parse(candidates)
    articles = [to_news_article(a, url) for a in parsed]
    # Exclude non-English or subdomains if needed
    return [
        art for art in articles
        if 'arabic.cnn.com' not in art.url and 'cnnespanol' not in art.url
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='News Article Scraper'
    )
    parser.add_argument(
        '--date',
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date(),
        default=datetime.date.today(),
        help='Date to scrape (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent / 'articles',
        help='Directory to save articles JSON'
    )
    parser.add_argument(
        '--sources',
        nargs='+',
        default=[
            'http://cnn.com',
            'http://cnbc.com',
            'http://washingtonpost.com',
            'http://npr.org',
        ],
        help='List of news source URLs'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    date = args.date
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logging.info(f"Starting scrape for {date.isoformat()}")

    start_time = time.time()
    all_articles: List[NewsArticle] = []
    for source_url in args.sources:
        logging.info(f"Scraping {source_url}")
        articles = scrape_source(source_url, date)
        all_articles.extend(articles)

    serialized = [asdict(art) for art in all_articles]
    filename = out_dir / f'articles_{date.isoformat()}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)

    duration = time.time() - start_time
    logging.info(f"Scraped {len(serialized)} articles in {duration:.2f}s")
    logging.info(f"Results saved to {filename}")


if __name__ == '__main__':
    main()