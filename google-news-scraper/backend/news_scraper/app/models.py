from datetime import datetime
from typing import Optional, Set
from uuid import UUID, uuid4
from pydantic import BaseModel
import json
import os

class News(BaseModel):
    id: UUID = uuid4()
    title: str
    content: str
    url: str
    source: str
    category: str
    published_at: datetime
    fetched_at: datetime = datetime.utcnow()

class NewsStorage:
    def __init__(self):
        self.file_path = "news_archive.txt"
        self.url_cache: Set[str] = set()
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        news = json.loads(line)
                        self.url_cache.add(news["url"])
                    except:
                        continue

    def save_news(self, news_items: list[News]) -> int:
        new_count = 0
        with open(self.file_path, "a", encoding="utf-8") as f:
            for news in news_items:
                if news.url not in self.url_cache:
                    f.write(news.model_dump_json() + "\n")
                    self.url_cache.add(news.url)
                    new_count += 1
        return new_count

    def get_news(self, from_date: Optional[datetime] = None,
                to_date: Optional[datetime] = None,
                category: Optional[str] = None) -> list[News]:
        news_items = []
        if not os.path.exists(self.file_path):
            return news_items

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    news_dict = json.loads(line)
                    news = News(**news_dict)
                    
                    if from_date and news.published_at < from_date:
                        continue
                    if to_date and news.published_at > to_date:
                        continue
                    if category and news.category.lower() != category.lower():
                        continue
                        
                    news_items.append(news)
                except:
                    continue

        return sorted(news_items, key=lambda x: x.published_at, reverse=True)
