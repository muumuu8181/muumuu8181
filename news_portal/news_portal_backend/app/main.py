from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import uuid
import json
import os
import schedule
import time
import threading
import logging
from .fetcher import fetch_news

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# In-memory storage
urls: List[dict] = []
articles: List[dict] = []
tags: List[dict] = []

# Load initial configuration
def load_url_config():
    config_path = os.path.join(os.path.dirname(__file__), 'url_config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Load tags
        for tag in config.get('tags', []):
            tag_dict = {
                'id': str(uuid.uuid4()),
                'name': tag['name'],
                'category': tag['category']
            }
            tags.append(tag_dict)
            
        # Load URLs
        for source in config.get('sources', []):
            url_dict = {
                'id': str(uuid.uuid4()),
                'name': source['name'],
                'url': source['url'],
                'type': source['type'],
                'tags': source['tags'],
                'active': source.get('active', True),
                'created_at': datetime.now().isoformat()
            }
            urls.append(url_dict)
            
    except Exception as e:
        print(f"Error loading URL configuration: {str(e)}")

# Schedule configuration
def run_scheduled_jobs():
    while True:
        schedule.run_pending()
        time.sleep(60)

def fetch_news_job():
    """Scheduled job to fetch news"""
    try:
        logger.info("Starting scheduled news fetch")
        active_urls = [url for url in urls if url.get('active', True)]
        new_articles = fetch_news(active_urls)
        
        # Add new articles to the in-memory store
        articles.extend(new_articles)
        
        # Keep only last 7 days of articles by default
        cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
        articles[:] = [
            article for article in articles
            if article['published_date'] >= cutoff_date
        ]
        
        logger.info(f"Completed scheduled fetch. New articles: {len(new_articles)}")
    except Exception as e:
        logger.error(f"Error in scheduled news fetch: {str(e)}")

# Schedule news fetching three times daily
schedule.every().day.at("06:00").do(fetch_news_job)
schedule.every().day.at("14:00").do(fetch_news_job)
schedule.every().day.at("22:00").do(fetch_news_job)

# Start scheduler in background thread
scheduler_thread = threading.Thread(target=run_scheduled_jobs, daemon=True)
scheduler_thread.start()

# Load configuration on startup
load_url_config()

# Run initial fetch
fetch_news_job()

# Endpoint to reload configuration
@app.post("/api/config/reload")
async def reload_config():
    urls.clear()
    tags.clear()
    load_url_config()
    return {"message": "Configuration reloaded", "urls": len(urls), "tags": len(tags)}

# Models
class URLBase(BaseModel):
    name: str
    url: str
    tags: List[str] = []
    active: bool = True

class URLCreate(URLBase):
    pass

class URL(URLBase):
    id: str
    created_at: str

class TagBase(BaseModel):
    name: str
    category: str

class TagCreate(TagBase):
    pass

class Tag(TagBase):
    id: str

class ArticleBase(BaseModel):
    title: str
    content: str
    url_id: str
    published_date: Optional[str] = None
    fetched_date: Optional[str] = None
    tags: List[str] = []
    source: str

class Article(ArticleBase):
    id: str

# URL endpoints
@app.get("/api/urls", response_model=List[URL])
async def list_urls():
    return urls

@app.post("/api/urls", response_model=URL)
async def create_url(url: URLCreate):
    url_dict = url.dict()
    url_dict.update({
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat()
    })
    urls.append(url_dict)
    return url_dict

@app.get("/api/urls/{url_id}", response_model=URL)
async def get_url(url_id: str):
    for url in urls:
        if url["id"] == url_id:
            return url
    raise HTTPException(status_code=404, detail="URL not found")

@app.put("/api/urls/{url_id}", response_model=URL)
async def update_url(url_id: str, url_update: URLBase):
    for i, url in enumerate(urls):
        if url["id"] == url_id:
            url_dict = url_update.dict()
            url_dict["id"] = url_id
            url_dict["created_at"] = url["created_at"]
            urls[i] = url_dict
            return url_dict
    raise HTTPException(status_code=404, detail="URL not found")

@app.delete("/api/urls/{url_id}")
async def delete_url(url_id: str):
    for i, url in enumerate(urls):
        if url["id"] == url_id:
            del urls[i]
            return {"message": "URL deleted"}
    raise HTTPException(status_code=404, detail="URL not found")

# Tag endpoints
@app.get("/api/tags", response_model=List[Tag])
async def list_tags():
    return tags

@app.post("/api/tags", response_model=Tag)
async def create_tag(tag: TagCreate):
    tag_dict = tag.dict()
    tag_dict["id"] = str(uuid.uuid4())
    tags.append(tag_dict)
    return tag_dict

@app.get("/api/urls/tags", response_model=List[dict])
async def get_url_tags():
    return [{"url_id": url["id"], "tags": url["tags"]} for url in urls]

@app.post("/api/urls/{url_id}/tags")
async def add_url_tags(url_id: str, new_tags: List[str]):
    for url in urls:
        if url["id"] == url_id:
            url["tags"] = list(set(url["tags"] + new_tags))
            return {"message": "Tags added", "tags": url["tags"]}
    raise HTTPException(status_code=404, detail="URL not found")

# Article endpoints
@app.get("/api/articles", response_model=List[Article])
async def list_articles(
    tags: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: Optional[str] = None
):
    filtered_articles = articles.copy()
    
    if tags:
        filtered_articles = [
            article for article in filtered_articles
            if any(tag in article["tags"] for tag in tags)
        ]
    
    if start_date:
        filtered_articles = [
            article for article in filtered_articles
            if article["published_date"] >= start_date
        ]
    
    if end_date:
        filtered_articles = [
            article for article in filtered_articles
            if article["published_date"] <= end_date
        ]
    
    if source:
        filtered_articles = [
            article for article in filtered_articles
            if article["source"] == source
        ]
    
    return filtered_articles

@app.get("/api/articles/{article_id}", response_model=Article)
async def get_article(article_id: str):
    for article in articles:
        if article["id"] == article_id:
            return article
    raise HTTPException(status_code=404, detail="Article not found")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
