#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime
import feedparser
import requests
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('url_indexer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class URLIndexer:
    def __init__(self):
        self.index_dir = "url_indexes"
        os.makedirs(self.index_dir, exist_ok=True)
        
        # BBC News RSS feeds
        self.bbc_feeds = {
            'top': 'http://feeds.bbci.co.uk/news/rss.xml',
            'world': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'business': 'http://feeds.bbci.co.uk/news/business/rss.xml',
            'technology': 'http://feeds.bbci.co.uk/news/technology/rss.xml',
            'science': 'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml'
        }
        
        # Reuters sitemap URL
        self.reuters_sitemap = "https://www.reuters.com/arc/outboundfeeds/news-sitemap-index/?outputType=xml"

    def collect_bbc_urls(self):
        """Collect URLs from BBC RSS feeds"""
        articles = []
        for category, feed_url in self.bbc_feeds.items():
            try:
                logger.info(f"Collecting BBC {category} URLs...")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    article = {
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.published,
                        'source': f'BBC News - {category.title()}',
                        'category': category,
                        'indexed_at': datetime.now().isoformat()
                    }
                    articles.append(article)
                    
            except Exception as e:
                logger.error(f"Error collecting BBC {category} URLs: {str(e)}")
                
        return articles

    def collect_reuters_urls(self):
        """Collect URLs from Reuters sitemap"""
        articles = []
        try:
            logger.info("Collecting Reuters URLs from sitemap...")
            response = requests.get(self.reuters_sitemap)
            root = ET.fromstring(response.content)
            
            # Process first sitemap (most recent articles)
            for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap')[:1]:
                loc_elem = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None and loc_elem.text is not None:
                    # Fetch articles from sitemap
                    try:
                        news_response = requests.get(loc_elem.text)
                        news_root = ET.fromstring(news_response.content)
                        
                        for url in news_root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                            loc_elem = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                            lastmod_elem = url.find('{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod')
                            
                            if loc_elem is not None and loc_elem.text is not None:
                                title = loc_elem.text.split('/')[-1].replace('-', ' ').title()
                                article = {
                                    'title': title,
                                    'link': loc_elem.text,
                                    'published': lastmod_elem.text if lastmod_elem is not None and lastmod_elem.text is not None else None,
                                    'source': 'Reuters',
                                    'category': 'news',  # Default category for Reuters
                                    'indexed_at': datetime.now().isoformat()
                                }
                                articles.append(article)
                    except requests.RequestException as e:
                        logger.error(f"Error fetching Reuters sitemap content: {str(e)}")
                    except ET.ParseError as e:
                        logger.error(f"Error parsing Reuters sitemap XML: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error collecting Reuters URLs: {str(e)}")
            
        return articles

    def save_url_index(self, articles):
        """Save collected URLs to JSON index file"""
        if not articles:
            logger.warning("No URLs to index")
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.index_dir, f'url_index_{timestamp}.json')
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_urls': len(articles),
                    'articles': articles
                }, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(articles)} URLs to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving URL index: {str(e)}")
            return None

    def collect_all_urls(self):
        """Collect URLs from all sources"""
        all_articles = []
        
        # Collect from BBC
        bbc_articles = self.collect_bbc_urls()
        all_articles.extend(bbc_articles)
        logger.info(f"Collected {len(bbc_articles)} BBC URLs")
        
        # Collect from Reuters
        reuters_articles = self.collect_reuters_urls()
        all_articles.extend(reuters_articles)
        logger.info(f"Collected {len(reuters_articles)} Reuters URLs")
        
        # Save URL index
        index_file = self.save_url_index(all_articles)
        
        return len(all_articles), index_file

def main():
    indexer = URLIndexer()
    total_urls, index_file = indexer.collect_all_urls()
    logger.info(f"Completed URL collection. Total URLs: {total_urls}")
    if index_file:
        logger.info(f"URL index saved to: {index_file}")

if __name__ == "__main__":
    main()
