# News Portal

A full-stack web application for managing and viewing news articles with mobile support. This application fetches news from multiple international sources three times daily, categorizes them with tags, and presents them through a mobile-responsive interface.

## Features

- **Automated News Fetching**
  - Scheduled fetching from multiple sources (RSS feeds and Sitemaps)
  - Runs at 06:00, 14:00, and 22:00 daily
  - Supports BBC News, Reuters, and configurable additional sources

- **Mobile-First Interface**
  - Responsive design for all screen sizes
  - Tag-based article filtering
  - Date range selection
  - Full article view with rich content

- **Flexible Configuration**
  - External URL configuration via JSON
  - Tag management system
  - Source activation/deactivation
  - Category-based organization

## Project Structure

```
news_portal/
├── news_portal_backend/    # FastAPI backend application
│   ├── app/               # Application code
│   │   ├── main.py       # FastAPI application
│   │   ├── fetcher.py    # News fetching logic
│   │   └── url_config.json # Source configuration
│   └── README.md         # Backend documentation
└── news_portal_frontend/  # React frontend application
    ├── src/              # Source code
    │   ├── pages/        # React components
    │   └── utils/        # Utilities and API client
    └── README.md         # Frontend documentation
```

## Installation

### Backend Setup

1. Install Python dependencies:
```bash
cd news_portal_backend
poetry install  # or pip install -r requirements.txt
```

2. Configure news sources in `app/url_config.json`:
```json
{
  "sources": [
    {
      "url": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
      "name": "BBC Science",
      "type": "rss",
      "tags": ["science"],
      "active": true
    }
  ]
}
```

3. Start the backend server:
```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
# or with pip: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd news_portal_frontend
npm install  # or pnpm install
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env to set VITE_BACKEND_URL to your backend URL
```

3. Start the development server:
```bash
npm run dev  # or pnpm dev
```

## Deployment

### Backend Deployment

The backend can be deployed using Docker:

1. Build the Docker image:
```bash
cd news_portal_backend
docker build -t news-portal-backend .
```

2. Run the container:
```bash
docker run -p 8000:8000 news-portal-backend
```

For production deployment, we recommend using a platform like Fly.io:
```bash
flyctl deploy
```

### Frontend Deployment

1. Build the production bundle:
```bash
cd news_portal_frontend
npm run build  # or pnpm build
```

2. Deploy the `dist` directory to your preferred static hosting service.

## Usage

### Managing News Sources

1. Add or modify sources in `news_portal_backend/app/url_config.json`
2. Use the URLs page in the frontend to activate/deactivate sources
3. Sources support both RSS feeds and Sitemaps

### Filtering Articles

1. Use the tag checkboxes to filter by category
2. Select date ranges from the dropdown:
   - Today
   - Last 7 days
   - Last 30 days
   - All time

### Managing Tags

1. Create new tags from the Tags page
2. Assign tags to news sources
3. Use tags for filtering articles

## API Documentation

The backend API documentation is available at:
- Swagger UI: `http://your-backend/docs`
- ReDoc: `http://your-backend/redoc`

Key endpoints:
- `GET /api/articles` - List articles with filtering
- `GET /api/articles/{id}` - Get article details
- `GET /api/tags` - List available tags
- `GET /api/urls` - List configured news sources

## Related Projects

- [news_fetcher](../news_fetcher/): Core news fetching functionality

## Contributing

Please refer to the repository's contributing guidelines for information on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
