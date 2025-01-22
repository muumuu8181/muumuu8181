#!/usr/bin/env python3
import schedule
import time
import logging
from news_fetcher import NewsFetcher
from url_indexer import URLIndexer
from content_fetcher import ContentFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def index_urls_job():
    """Job to collect and index URLs from all sources"""
    try:
        logger.info("Starting scheduled URL indexing")
        indexer = URLIndexer()
        total_urls, index_file = indexer.collect_all_urls()
        logger.info(f"Completed URL indexing. Total URLs: {total_urls}")
        return index_file
    except Exception as e:
        logger.error(f"Error in scheduled URL indexing: {str(e)}")
        return None

def fetch_content_job(index_file):
    """Job to fetch full content from indexed URLs"""
    if not index_file:
        logger.error("No index file provided for content fetching")
        return
        
    try:
        logger.info(f"Starting content fetching from index: {index_file}")
        fetcher = ContentFetcher()
        successful, failed = fetcher.process_url_index(index_file)
        logger.info(f"Completed content fetching. Successful: {successful}, Failed: {failed}")
    except Exception as e:
        logger.error(f"Error in scheduled content fetching: {str(e)}")

def combined_news_job():
    """Combined job to run URL indexing followed by content fetching"""
    try:
        # First, collect and index URLs
        index_file = index_urls_job()
        
        # Then, if successful, fetch full content
        if index_file:
            fetch_content_job(index_file)
        else:
            logger.error("URL indexing failed, skipping content fetching")
            
    except Exception as e:
        logger.error(f"Error in combined news job: {str(e)}")

def legacy_fetch_news_job():
    """Legacy job to fetch news using the old method (kept for compatibility)"""
    try:
        logger.info("Starting scheduled legacy news fetch")
        fetcher = NewsFetcher()
        total_articles = fetcher.fetch_all_news()
        logger.info(f"Completed legacy fetch. Total articles: {total_articles}")
    except Exception as e:
        logger.error(f"Error in scheduled legacy news fetch: {str(e)}")

def main():
    # Schedule combined jobs to run at 6:00, 14:00, and 22:00 every day
    schedule.every().day.at("06:00").do(combined_news_job)
    schedule.every().day.at("14:00").do(combined_news_job)
    schedule.every().day.at("22:00").do(combined_news_job)
    
    # Also keep the legacy job running at the same times
    schedule.every().day.at("06:00").do(legacy_fetch_news_job)
    schedule.every().day.at("14:00").do(legacy_fetch_news_job)
    schedule.every().day.at("22:00").do(legacy_fetch_news_job)
    
    logger.info("Enhanced news fetcher scheduler started")
    logger.info("Scheduled times: 06:00, 14:00, 22:00")
    logger.info("Running both new (URL index + content) and legacy jobs")
    
    # Run jobs continuously
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in scheduler: {str(e)}")
            # Wait before retrying
            time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    main()
