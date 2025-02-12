from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Optional
from apscheduler.schedulers.background import BackgroundScheduler
import logging

from app.models import News, NewsStorage
from app.scraper import fetch_google_news, NewsScraperError

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://news-scraping-app-63a2iyf9.devinapps.com"],  # Allow only our frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize storage
news_store = NewsStorage()
categories = ["General", "World", "Business", "Technology", "Entertainment", "Sports", "Science", "Health"]

def update_news():
    """Background task to fetch and update news"""
    try:
        new_items = fetch_google_news()
        if new_items:
            new_count = news_store.save_news(new_items)
            logger.info(f"Successfully fetched and saved {new_count} new news items")
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
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
) -> dict:
    """Get news with filtering"""
    filtered_news = news_store.get_news(from_date, to_date, category)
    
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

@app.get("/api/news/text")
async def get_news_text(
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    category: Optional[str] = None,
) -> str:
    """Get news as plain text for easy copying"""
    news_items = news_store.get_news(from_date, to_date, category)
    text_output = []
    for news in news_items:
        text_output.append(f"【{news.category}】{news.title}\n{news.url}\n")
    return "\n".join(text_output)
