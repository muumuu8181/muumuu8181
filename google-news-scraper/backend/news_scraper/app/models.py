from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel

class News(BaseModel):
    id: UUID = uuid4()
    title: str
    content: str
    url: str
    source: str
    category: str
    published_at: datetime
    fetched_at: datetime = datetime.utcnow()
