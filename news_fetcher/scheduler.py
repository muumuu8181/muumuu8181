#!/usr/bin/env python3
import schedule
import time
import logging
from news_fetcher import NewsFetcher

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

def fetch_news_job():
    """Job to fetch news from all sources"""
    try:
        logger.info("Starting scheduled news fetch")
        fetcher = NewsFetcher()
        total_articles = fetcher.fetch_all_news()
        logger.info(f"Completed scheduled fetch. Total articles: {total_articles}")
    except Exception as e:
        logger.error(f"Error in scheduled news fetch: {str(e)}")

def main():
    # Schedule jobs to run at 6:00, 14:00, and 22:00 every day
    schedule.every().day.at("06:00").do(fetch_news_job)
    schedule.every().day.at("14:00").do(fetch_news_job)
    schedule.every().day.at("22:00").do(fetch_news_job)
    
    logger.info("News fetcher scheduler started")
    logger.info("Scheduled times: 06:00, 14:00, 22:00")
    
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
