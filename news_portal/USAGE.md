# News Portal - Advanced Usage Guide

This guide provides detailed examples for using the News Portal's advanced features.

## API Usage Examples

### Article Filtering

#### Tag-based Filtering

To filter articles by tags, use the `tags` query parameter. You can specify multiple tags to find articles that match any of the provided tags.

```bash
# Get articles tagged with "health"
curl "http://your-backend/api/articles?tags=health"

# Get articles tagged with either "health" or "business"
curl "http://your-backend/api/articles?tags=health&tags=business"
```

Example response:
```json
[
  {
    "id": "c352eafe-02f9-4d44-aaf8-df051f3f74cd",
    "title": "AI in Healthcare: Latest Developments",
    "content": "Recent developments in AI are transforming healthcare...",
    "source": "BBC Health",
    "published_date": "2025-01-22T10:30:00",
    "tags": ["health", "technology"]
  }
]
```

#### Date-based Filtering

Filter articles by date range using `start_date` and `end_date` parameters:

```bash
# Get articles from the last 7 days
curl "http://your-backend/api/articles?start_date=$(date -d '7 days ago' +%Y-%m-%d)"

# Get articles between specific dates
curl "http://your-backend/api/articles?start_date=2025-01-15&end_date=2025-01-22"
```

#### Combined Filtering

Combine multiple filters to narrow down results:

```bash
# Get health articles from the last 7 days
curl "http://your-backend/api/articles?tags=health&start_date=$(date -d '7 days ago' +%Y-%m-%d)"
```

### Tag Management

#### Creating New Tags

1. Navigate to the Tags page in the frontend
2. Click "Create New Tag"
3. Enter:
   - Tag Name (e.g., "artificial-intelligence")
   - Category (e.g., "technology")
4. Click "Create Tag"

#### Associating Tags with Sources

1. Navigate to the URLs page
2. Find the source you want to tag
3. Click "Edit"
4. In the tags field, enter comma-separated tags
5. Click "Save"

Example URL configuration with tags:
```json
{
  "sources": [
    {
      "url": "http://feeds.bbci.co.uk/news/technology/rss.xml",
      "name": "BBC Technology",
      "type": "rss",
      "tags": ["technology", "artificial-intelligence"],
      "active": true
    }
  ]
}
```

### Source Management

#### Adding New Sources

1. Navigate to the URLs page
2. Click "Add New URL"
3. Enter:
   - Name: Display name for the source
   - URL: RSS feed or sitemap URL
   - Tags: Comma-separated list of tags
4. Click "Add URL"

Example source configurations:

```json
// RSS Feed Example
{
  "url": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
  "name": "BBC Science",
  "type": "rss",
  "tags": ["science", "environment"],
  "active": true
}

// Sitemap Example
{
  "url": "https://www.reuters.com/arc/outboundfeeds/science-news/?outputType=xml",
  "name": "Reuters Science",
  "type": "sitemap",
  "tags": ["science"],
  "active": true
}
```

## Frontend Features

### Advanced Article Filtering

1. **Multiple Tag Selection**
   - Use checkboxes in the tag filter section
   - Articles matching ANY selected tag will be shown
   - Clear all filters using the reset button

2. **Date Range Selection**
   - Click the date filter dropdown
   - Choose from preset ranges:
     - Today
     - Last 7 days
     - Last 30 days
     - All time

3. **Search Functionality**
   - Use the search bar to filter articles by title
   - Search is case-insensitive
   - Matches partial words

### Article Reading

1. **Article List View**
   - Shows article preview with title, source, and tags
   - Click "Read more" to view full article

2. **Article Detail View**
   - Full article content
   - Source and publication date
   - Associated tags
   - Back button to return to list

### Mobile Usage

The interface is optimized for mobile devices:

1. **Navigation**
   - Menu collapses to hamburger on small screens
   - Touch-friendly buttons and controls
   - Responsive grid layout

2. **Article Reading**
   - Text size adjusts for readability
   - Images scale appropriately
   - Swipe gestures supported

## Troubleshooting

### Common Issues

1. **Articles Not Loading**
   - Check backend URL configuration
   - Verify network connectivity
   - Check browser console for errors

2. **Filtering Not Working**
   - Clear browser cache
   - Verify tag names match exactly
   - Check date format (YYYY-MM-DD)

3. **Source Updates**
   - Sources are fetched three times daily
   - Manual refresh available via API
   - Check source status in URLs page
