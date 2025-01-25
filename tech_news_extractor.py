#!/usr/bin/env python3
import feedparser
import json
from datetime import datetime
import time

# Tech/AI news sources
TECH_NEWS_SOURCES = [
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "category": "technology"
    },
    {
        "name": "MIT Technology Review",
        "url": "https://www.technologyreview.com/feed/",
        "category": "technology"
    },
    {
        "name": "Wired",
        "url": "https://www.wired.com/feed/rss",
        "category": "technology"
    }
]

def fetch_articles():
    """Fetch articles from tech news sources"""
    all_articles = []
    
    for source in TECH_NEWS_SOURCES:
        try:
            print(f"Fetching articles from {source['name']}...")
            feed = feedparser.parse(source['url'])
            
            for entry in feed.entries[:10]:  # Limit to 10 most recent articles per source
                article = {
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.get('published', datetime.now().isoformat()),
                    'source': source['name'],
                    'category': source['category'],
                    'summary': entry.get('summary', '')
                }
                all_articles.append(article)
                
            # Add delay between requests
            time.sleep(2)
            
        except Exception as e:
            print(f"Error fetching from {source['name']}: {str(e)}")
            continue
    
    return all_articles

def save_articles(articles):
    """Save articles to JSON file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'tech_articles_{timestamp}.json'
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_articles': len(articles),
                'articles': articles
            }, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(articles)} articles to {filename}")
        return filename
        
    except Exception as e:
        print(f"Error saving articles: {str(e)}")
        return None

def main():
    print("Starting tech news extraction...")
    articles = fetch_articles()
    if articles:
        save_articles(articles)
    print("Extraction complete.")

if __name__ == "__main__":
    main()
