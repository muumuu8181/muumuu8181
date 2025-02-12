import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from typing import List
from .models import News

logger = logging.getLogger(__name__)

class NewsScraperError(Exception):
    pass

def fetch_google_news() -> List[News]:
    try:
        response = requests.get("https://news.google.com")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_items = []
        articles = soup.find_all('article')
        
        for article in articles:
            try:
                title_elem = article.find('a', {'tabindex': '0'})
                if not title_elem:
                    continue
                    
                title = title_elem.text.strip()
                url = f"https://news.google.com{title_elem.get('href', '')[1:]}"
                
                # Get source and time
                source = article.find('img', {'class': 'tvs3Id'})
                source = source.get('alt', 'Unknown') if source else 'Unknown'
                
                time_elem = article.find('time')
                published_text = time_elem.text if time_elem else ''
                # Simple parsing of relative time
                published_at = datetime.utcnow()  # Default to now, improve parsing later
                
                # For now, we'll use a default category
                category = "General"
                
                news_items.append(News(
                    title=title,
                    content="",  # We'll fetch full content later if needed
                    url=url,
                    source=source,
                    category=category,
                    published_at=published_at
                ))
            except Exception as e:
                logger.error(f"Error parsing article: {str(e)}")
                continue
                
        return news_items
    except Exception as e:
        raise NewsScraperError(f"Failed to fetch news: {str(e)}")
