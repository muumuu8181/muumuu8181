from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Optional
from apscheduler.schedulers.background import BackgroundScheduler
import logging

from .models import News
from .scraper import fetch_google_news, NewsScraperError

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
news_storage: List[News] = []
categories = ["General", "World", "Business", "Technology", "Entertainment", "Sports", "Science", "Health"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_news():
    """Background task to fetch and update news"""
    try:
        global news_storage
        new_items = fetch_google_news()
        if new_items:
            # Keep only last 1000 items to manage memory
            news_storage = (new_items + news_storage)[:1000]
            logger.info(f"Successfully fetched {len(new_items)} news items")
    except NewsScraperError as e:
        logger.error(f"Failed to update news: {str(e)}")

# Setup scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(update_news, 'interval', minutes=10)
scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/api/news/fetch")
async def fetch_news(background_tasks: BackgroundTasks):
    """Manually trigger news fetch"""
    background_tasks.add_task(update_news)
    return {"message": "News fetch triggered"}

@app.get("/api/news")
async def get_news(
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
) -> dict:
    """Get news with filtering and pagination"""
    filtered_news = news_storage.copy()
    
    if from_date:
        filtered_news = [n for n in filtered_news if n.published_at >= from_date]
    if to_date:
        filtered_news = [n for n in filtered_news if n.published_at <= to_date]
    if category:
        filtered_news = [n for n in filtered_news if n.category.lower() == category.lower()]
    if source:
        filtered_news = [n for n in filtered_news if n.source.lower() == source.lower()]
    
    # Calculate pagination
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_news = filtered_news[start_idx:end_idx]
    
    return {
        "total": len(filtered_news),
        "page": page,
        "limit": limit,
        "items": paginated_news
    }

@app.get("/api/categories")
async def get_categories():
    """Get available news categories"""
    return {"categories": categories}
