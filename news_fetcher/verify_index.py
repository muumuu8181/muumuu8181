#!/usr/bin/env python3
import json
import logging
from url_indexer import URLIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('verify_index.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_url_index():
    """Verify URL index creation and content"""
    try:
        # Create index
        logger.info("Creating new URL index for verification...")
        indexer = URLIndexer()
        total_urls, index_file = indexer.collect_all_urls()
        
        if not index_file:
            logger.error("Failed to create URL index")
            return False
            
        # Verify index content
        logger.info(f"Verifying index file: {index_file}")
        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Check required fields
        required_fields = ['timestamp', 'total_urls', 'articles']
        if not all(field in data for field in required_fields):
            logger.error("Missing required fields in index file")
            return False
            
        # Check articles
        articles = data['articles']
        if not articles:
            logger.error("No articles found in index")
            return False
            
        # Verify article metadata
        required_article_fields = ['title', 'link', 'published', 'source']
        valid_articles = 0
        invalid_articles = 0
        
        for article in articles:
            if all(field in article for field in required_article_fields):
                valid_articles += 1
            else:
                invalid_articles += 1
                
        # Print verification results
        logger.info(f"Index file timestamp: {data['timestamp']}")
        logger.info(f"Total URLs in index: {data['total_urls']}")
        logger.info(f"Valid articles: {valid_articles}")
        logger.info(f"Invalid articles: {invalid_articles}")
        logger.info(f"Sources found: {set(article['source'] for article in articles)}")
        
        return valid_articles > 0 and invalid_articles == 0
        
    except Exception as e:
        logger.error(f"Error verifying URL index: {str(e)}")
        return False

if __name__ == "__main__":
    success = verify_url_index()
    if success:
        logger.info("URL index verification successful")
    else:
        logger.error("URL index verification failed")
