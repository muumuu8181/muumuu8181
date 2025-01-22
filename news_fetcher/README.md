# News Fetcher

A Python application that automatically fetches and aggregates news from multiple international sources three times daily.

## Features

- Fetches news from multiple international sources (BBC News, Reuters)
- Runs automatically three times per day (06:00, 14:00, 22:00)
- Saves news articles in text format with timestamps
- Includes error handling and logging
- Supports multiple news categories (World, Business, Technology, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/muumuu8181/muumuu8181.git
cd muumuu8181/news_fetcher
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the News Fetcher

To run the news fetcher once:
```bash
python news_fetcher.py
```

### Running the Scheduler

To start the automated scheduler (runs three times daily):
```bash
python scheduler.py
```

The scheduler will run the news fetcher at the following times:
- 06:00 (Morning)
- 14:00 (Afternoon)
- 22:00 (Evening)

### Output

News articles are saved in the `news_output` directory with timestamps in the filename format: `news_YYYYMMDD_HHMMSS.txt`

### Logs

- News fetcher logs: `news_fetcher.log`
- Scheduler logs: `scheduler.log`

## Configuration

The news sources and categories can be modified in `news_fetcher.py`:
- BBC News feeds are configured in the `bbc_feeds` dictionary
- Reuters news is fetched from their sitemap

## Error Handling

The application includes comprehensive error handling:
- Network errors are logged and retried
- XML parsing errors are caught and logged
- Scheduler errors are handled with a 5-minute retry delay

## Requirements

- Python 3.6+
- Required packages listed in `requirements.txt`
