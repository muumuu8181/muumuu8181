#!/usr/bin/env python3
import os
import logging
from content_fetcher import ContentFetcher
from url_indexer import URLIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('verify_content.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def verify_content_extraction():
    """Verify content extraction and storage"""
    try:
        # First, create a new URL index
        logger.info("Creating new URL index for content verification...")
        indexer = URLIndexer()
        total_urls, index_file = indexer.collect_all_urls()
        
        if not index_file:
            logger.error("Failed to create URL index")
            return False
            
        # Then fetch content using the index
        logger.info("Fetching content from URLs...")
        fetcher = ContentFetcher()
        successful, failed = fetcher.process_url_index(index_file)
        
        # Verify content files
        output_dir = fetcher.output_dir
        content_files = [f for f in os.listdir(output_dir) if f.startswith('article_')]
        
        if not content_files:
            logger.error("No content files found in output directory")
            return False
            
        # Sample and verify content files
        sample_size = min(5, len(content_files))
        logger.info(f"Verifying {sample_size} sample content files...")
        
        valid_files = 0
        invalid_files = 0
        
        for filename in content_files[:sample_size]:
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check required fields
                required_fields = ['Title:', 'Source:', 'Published:', 'URL:']
                if all(field in content for field in required_fields):
                    # Check for actual article content
                    separator = "=" * 80
                    if separator in content:
                        article_text = content.split(separator)[1].strip()
                        if len(article_text) > 100:  # Minimum content length
                            valid_files += 1
                            continue
                            
                invalid_files += 1
                logger.warning(f"Invalid content file: {filename}")
                
            except Exception as e:
                logger.error(f"Error reading content file {filename}: {str(e)}")
                invalid_files += 1
                
        # Print verification results
        logger.info(f"Total content files: {len(content_files)}")
        logger.info(f"Sample size: {sample_size}")
        logger.info(f"Valid files in sample: {valid_files}")
        logger.info(f"Invalid files in sample: {invalid_files}")
        logger.info(f"Content fetch success rate: {successful}/{successful + failed}")
        
        return valid_files > 0 and invalid_files == 0
        
    except Exception as e:
        logger.error(f"Error in content verification: {str(e)}")
        return False

if __name__ == "__main__":
    success = verify_content_extraction()
    if success:
        logger.info("Content verification successful")
    else:
        logger.error("Content verification failed")
