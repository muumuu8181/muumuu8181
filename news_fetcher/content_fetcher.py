#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup, Tag, NavigableString
import time
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('content_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContentFetcher:
    def __init__(self):
        self.output_dir = "full_text_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Configure delay between requests (in seconds)
        self.request_delay = 2

    def fetch_bbc_content(self, url):
        """Extract content from BBC news article"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find main article content
            article_body = []
            
            # Try different BBC article layouts
            # Try multiple BBC article layouts
            # Layout 1: Article tag
            article = soup.find('article')
            if article and isinstance(article, Tag):
                for elem in article.find_all(['p', 'h2']):
                    if not isinstance(elem, NavigableString) and \
                       not any(cls in elem.get('class', []) for cls in ['visually-hidden']):
                        text = elem.get_text().strip()
                        if text:
                            article_body.append(text)
            
            # Layout 2: Text blocks
            if not article_body:
                for block in soup.find_all(attrs={'data-component': 'text-block'}):
                    if isinstance(block, Tag):
                        text = block.get_text().strip()
                        if text:
                            article_body.append(text)
            
            # Layout 3: Story body
            if not article_body:
                story = soup.find('div', class_='story-body')
                if story and isinstance(story, Tag):
                    for p in story.find_all('p'):
                        if isinstance(p, Tag):
                            text = p.get_text().strip()
                            if text:
                                article_body.append(text)
            
            return '\n\n'.join(filter(None, article_body))
            
        except Exception as e:
            logger.error(f"Error fetching BBC content from {url}: {str(e)}")
            return None

    def fetch_reuters_content(self, url):
        """Extract content from Reuters news article"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find main article content
            article_body = []
            
            # Try different Reuters article layouts
            # Try multiple Reuters article layouts
            # Layout 1: Modern article body
            article = soup.find('div', class_='article-body__content__17Yit')
            if article and isinstance(article, Tag):
                for elem in article.find_all(['p', 'h2']):
                    if isinstance(elem, Tag):
                        text = elem.get_text().strip()
                        if text:
                            article_body.append(text)
            
            # Layout 2: Text blocks
            if not article_body:
                for block in soup.find_all('p', class_='text__text__1FZLe'):
                    if isinstance(block, Tag):
                        text = block.get_text().strip()
                        if text:
                            article_body.append(text)
            
            # Layout 3: Standard article body
            if not article_body:
                body = soup.find('div', class_='ArticleBody__content___2gQno2')
                if body and isinstance(body, Tag):
                    for p in body.find_all('p'):
                        if isinstance(p, Tag):
                            text = p.get_text().strip()
                            if text:
                                article_body.append(text)
            
            return '\n\n'.join(filter(None, article_body))
            
        except Exception as e:
            logger.error(f"Error fetching Reuters content from {url}: {str(e)}")
            return None

    def fetch_article_content(self, article):
        """Fetch article content based on source"""
        url = article['link']
        domain = urlparse(url).netloc
        
        # Add delay between requests
        time.sleep(self.request_delay)
        
        if 'bbc.co.uk' in domain:
            return self.fetch_bbc_content(url)
        elif 'reuters.com' in domain:
            return self.fetch_reuters_content(url)
        else:
            logger.warning(f"Unsupported domain: {domain}")
            return None

    def save_article_content(self, article, content):
        """Save article content to file"""
        if not content:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_title = ''.join(c for c in article['title'] if c.isalnum() or c in (' ', '-', '_'))[:50]
        filename = os.path.join(self.output_dir, f'article_{timestamp}_{safe_title}.txt')
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {article['title']}\n")
                f.write(f"Source: {article['source']}\n")
                f.write(f"Category: {article.get('category', 'N/A')}\n")
                f.write(f"Published: {article['published']}\n")
                f.write(f"URL: {article['link']}\n")
                f.write(f"Fetched: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(content)
                
            logger.info(f"Saved article content to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving article content: {str(e)}")
            return None

    def process_url_index(self, index_file):
        """Process URL index file and fetch content for each article"""
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            articles = data['articles']
            total_articles = len(articles)
            successful = 0
            failed = 0
            
            logger.info(f"Processing {total_articles} articles from {index_file}")
            
            for i, article in enumerate(articles, 1):
                logger.info(f"Processing article {i}/{total_articles}: {article['title']}")
                
                content = self.fetch_article_content(article)
                if content:
                    if self.save_article_content(article, content):
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1
                    
            logger.info(f"Completed processing {total_articles} articles.")
            logger.info(f"Successful: {successful}, Failed: {failed}")
            
            return successful, failed
            
        except Exception as e:
            logger.error(f"Error processing index file {index_file}: {str(e)}")
            return 0, 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fetch full content of articles from URL index')
    parser.add_argument('index_file', help='Path to URL index JSON file')
    args = parser.parse_args()
    
    fetcher = ContentFetcher()
    successful, failed = fetcher.process_url_index(args.index_file)
    logger.info(f"Content fetching completed. Successful: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main()
