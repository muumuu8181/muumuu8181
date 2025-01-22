#!/usr/bin/env python3
import os
import logging
from datetime import datetime
import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsFetcher:
    def __init__(self):
        self.output_dir = "news_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
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

    def fetch_bbc_news(self):
        """Fetch news from BBC RSS feeds"""
        articles = []
        for category, feed_url in self.bbc_feeds.items():
            try:
                logger.info(f"Fetching BBC {category} news...")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    article = {
                        'title': entry.title,
                        'link': entry.link,
                        'description': entry.description,
                        'published': entry.published,
                        'source': f'BBC News - {category.title()}'
                    }
                    articles.append(article)
                    
            except Exception as e:
                logger.error(f"Error fetching BBC {category} news: {str(e)}")
                
        return articles

    def fetch_reuters_news(self):
        """Fetch news from Reuters sitemap"""
        articles = []
        try:
            logger.info("Fetching Reuters sitemap...")
            response = requests.get(self.reuters_sitemap)
            root = ET.fromstring(response.content)
            
            # Process first 100 articles (adjust as needed)
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
                                    'source': 'Reuters'
                                }
                                articles.append(article)
                    except requests.RequestException as e:
                        logger.error(f"Error fetching Reuters sitemap content: {str(e)}")
                    except ET.ParseError as e:
                        logger.error(f"Error parsing Reuters sitemap XML: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error fetching Reuters news: {str(e)}")
            
        return articles

    def save_articles(self, articles):
        """Save fetched articles to text file"""
        if not articles:
            logger.warning("No articles to save")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f'news_{timestamp}.txt')
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"News Articles - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                for article in articles:
                    f.write(f"Title: {article['title']}\n")
                    f.write(f"Source: {article['source']}\n")
                    f.write(f"Published: {article['published']}\n")
                    f.write(f"Link: {article['link']}\n")
                    if 'description' in article:
                        f.write(f"Description: {article['description']}\n")
                    f.write("-" * 80 + "\n\n")
                    
            logger.info(f"Saved {len(articles)} articles to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving articles: {str(e)}")

    def fetch_all_news(self):
        """Fetch news from all sources"""
        all_articles = []
        
        # Fetch from BBC
        bbc_articles = self.fetch_bbc_news()
        all_articles.extend(bbc_articles)
        logger.info(f"Fetched {len(bbc_articles)} BBC articles")
        
        # Fetch from Reuters
        reuters_articles = self.fetch_reuters_news()
        all_articles.extend(reuters_articles)
        logger.info(f"Fetched {len(reuters_articles)} Reuters articles")
        
        # Save all articles
        self.save_articles(all_articles)
        
        return len(all_articles)

def main():
    fetcher = NewsFetcher()
    total_articles = fetcher.fetch_all_news()
    logger.info(f"Completed news fetching. Total articles: {total_articles}")

if __name__ == "__main__":
    main()
