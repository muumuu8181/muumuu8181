#!/usr/bin/env python3
import feedparser
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup, Tag, NavigableString
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
import uuid
from urllib.parse import urlparse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsFetcher:
    def __init__(self, urls: List[dict], request_delay: int = 2):
        self.urls = urls
        self.request_delay = request_delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_rss_feed(self, url_config: dict) -> List[dict]:
        """Fetch articles from RSS feed"""
        articles = []
        try:
            logger.info(f"Fetching RSS feed: {url_config['name']}")
            feed = feedparser.parse(url_config['url'])
            
            for entry in feed.entries:
                article = {
                    'id': str(uuid.uuid4()),
                    'title': entry.title,
                    'content': entry.get('description', ''),
                    'url_id': url_config['id'],
                    'published_date': entry.get('published', datetime.now().isoformat()),
                    'fetched_date': datetime.now().isoformat(),
                    'tags': url_config['tags'],
                    'source': url_config['name']
                }
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error fetching RSS feed {url_config['name']}: {str(e)}")
            
        return articles

    def fetch_sitemap(self, url_config: dict) -> List[dict]:
        """Fetch articles from sitemap"""
        articles = []
        try:
            logger.info(f"Fetching sitemap: {url_config['name']}")
            response = requests.get(url_config['url'], headers=self.headers)
            root = ET.fromstring(response.content)
            
            for url in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url')[:10]:  # Limit to 10 articles
                loc_elem = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                lastmod_elem = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')
                
                if loc_elem is not None and loc_elem.text:
                    # Fetch article content
                    time.sleep(self.request_delay)
                    content = self.fetch_article_content(loc_elem.text)
                    
                    article = {
                        'id': str(uuid.uuid4()),
                        'title': loc_elem.text.split('/')[-1].replace('-', ' ').title(),
                        'content': content or '',
                        'url_id': url_config['id'],
                        'published_date': lastmod_elem.text if lastmod_elem is not None else datetime.now().isoformat(),
                        'fetched_date': datetime.now().isoformat(),
                        'tags': url_config['tags'],
                        'source': url_config['name']
                    }
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Error fetching sitemap {url_config['name']}: {str(e)}")
            
        return articles

    def fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch full article content"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find main article content
            article_body = []
            
            # Try different article layouts
            article = soup.find('article')
            if article and isinstance(article, Tag):
                for elem in article.find_all(['p', 'h2']):
                    if not isinstance(elem, NavigableString) and \
                       not any(cls in elem.get('class', []) for cls in ['visually-hidden']):
                        text = elem.get_text().strip()
                        if text:
                            article_body.append(text)
            
            # Try text blocks
            if not article_body:
                for block in soup.find_all(['p', 'div'], class_=['text-block', 'article-body']):
                    if isinstance(block, Tag):
                        text = block.get_text().strip()
                        if text:
                            article_body.append(text)
            
            return '\n\n'.join(filter(None, article_body))
            
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {str(e)}")
            return None

    def fetch_all_news(self) -> List[dict]:
        """Fetch news from all active sources"""
        all_articles = []
        
        for url_config in self.urls:
            if not url_config.get('active', True):
                continue
                
            # Add delay between sources
            time.sleep(self.request_delay)
            
            source_type = url_config.get('type', '').lower()
            try:
                if source_type == 'rss':
                    logger.info(f"Fetching RSS feed: {url_config['name']}")
                    articles = self.fetch_rss_feed(url_config)
                elif source_type == 'sitemap':
                    logger.info(f"Fetching sitemap: {url_config['name']}")
                    articles = self.fetch_sitemap(url_config)
                else:
                    logger.warning(f"Unknown source type '{source_type}' for {url_config['name']}")
                    continue
                    
                all_articles.extend(articles)
                logger.info(f"Fetched {len(articles)} articles from {url_config['name']}")
                
            except Exception as e:
                logger.error(f"Error fetching from {url_config['name']}: {str(e)}")
                continue
            
        return all_articles

def fetch_news(urls: List[dict]) -> List[dict]:
    """Helper function to fetch news articles"""
    fetcher = NewsFetcher(urls)
    return fetcher.fetch_all_news()
