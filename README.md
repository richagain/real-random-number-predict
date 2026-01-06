# Singapore TOTO Predictor

A Python web application that analyzes historical Singapore TOTO lottery data and generates number predictions using statistical analysis and machine learning.

## Features

- **Data Scraping**: Automatically fetches historical TOTO results
- **Statistical Analysis**: Frequency analysis, hot/cold numbers, pattern detection
- **ML Predictions**: LSTM neural network for sequence prediction
- **5 Prediction Strategies**:
  - Hot Numbers (frequently drawn)
  - Cold Numbers (overdue)
  - Balanced (optimal odd/even, high/low distribution)
  - Machine Learning (LSTM-based)
  - Ensemble (combines all strategies)
- **Web Dashboard**: Interactive UI with charts and visualizations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Populate Database

Scrape historical data from the web:

```bash
python run.py populate --max-pages 10
```

### 3. Run Web Server

```bash
python run.py server
```

Open http://localhost:8000 in your browser.

## Usage

### Command Line

```bash
# Run web server
python run.py server --host 0.0.0.0 --port 8000

# Populate database (scrape data)
python run.py populate --max-pages 20

# Train ML model
python run.py train --epochs 50
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/draws` | GET | List historical draws |
| `/api/statistics` | GET | Number frequency statistics |
| `/api/patterns` | GET | Pattern analysis |
| `/api/predict` | POST | Generate predictions |
| `/api/predict/all` | POST | Predictions from all strategies |
| `/api/scrape` | POST | Trigger data scrape |
| `/api/train` | POST | Train ML model |

### Generate Predictions

```python
from app.database.db import get_sync_session, init_db_sync
from app.analysis.predictor import TOTOPredictor

init_db_sync()
session = get_sync_session()
predictor = TOTOPredictor(session)

# Get predictions
predictions = predictor.predict(strategy="ensemble", count=5)
for p in predictions:
    print(f"Numbers: {p['numbers']} ({p['strategy']})")
```

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── api/routes.py        # API endpoints
│   ├── database/            # SQLAlchemy models
│   ├── scraper/             # Web scraping
│   ├── analysis/            # Statistics & ML
│   └── templates/           # HTML templates
├── data/                    # SQLite database
├── requirements.txt
└── run.py                   # CLI runner
```

## Singapore TOTO Rules

- Pick 6 numbers from 1-49
- Draws every Monday and Thursday at 6:30pm
- 6 winning numbers + 1 additional number
- Jackpot odds: 1 in 13,983,816

## Disclaimer

This application is for **entertainment purposes only**. Lottery numbers are randomly drawn, and no prediction system can guarantee winning numbers. Please gamble responsibly.

## License

MIT
