import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from typing import List
from .models import News

logger = logging.getLogger(__name__)

class NewsScraperError(Exception):
    pass

def fetch_google_news() -> List[News]:
    try:
        url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        news_items = []
        items = root.findall(".//item")
        
        for item in items:
            try:
                title = item.find("title").text.strip()
                link = item.find("link").text.strip()
                description = item.find("description").text
                source_elem = item.find("source")
                source = source_elem.text if source_elem is not None else "Unknown"
                pub_date = item.find("pubDate").text
                
                # Parse description HTML to extract content and related articles
                soup = BeautifulSoup(description, 'html.parser')
                # Get all articles from description
                articles = []
                for article_elem in soup.find_all('a'):
                    article_title = article_elem.text.strip()
                    article_source = article_elem.find_next('font', {'color': '#6f6f6f'})
                    source_text = article_source.text if article_source else source
                    articles.append(f"{article_title} ({source_text})")

                # Combine articles into content
                content = "\n".join([
                    "Main Article:",
                    articles[0],
                    "\nRelated Articles:",
                    *articles[1:]
                ]) if articles else ""
                
                # Get source from first article
                first_source = soup.find('font', {'color': '#6f6f6f'})
                if first_source:
                    source = first_source.text
                
                # Parse publication date
                try:
                    published_at = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S GMT")
                except ValueError:
                    # Fallback to current time if parsing fails
                    published_at = datetime.utcnow()
                    logger.warning(f"Failed to parse date '{pub_date}', using current time")
                
                # For now, we'll use a default category
                # TODO: Implement category detection based on content/keywords
                category = "General"
                
                news_items.append(News(
                    title=title,
                    content=content,
                    url=link,
                    source=source,
                    category=category,
                    published_at=published_at
                ))
            except Exception as e:
                logger.error(f"Error parsing RSS item: {str(e)}")
                continue
                
        return news_items
    except Exception as e:
        raise NewsScraperError(f"Failed to fetch news: {str(e)}")
