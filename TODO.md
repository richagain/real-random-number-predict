# Singapore TOTO Predictor - Project Status

## Current Progress (2026-01-07)

### Completed Features

- [x] **Project Structure** - Full Python project with FastAPI
- [x] **Database** - SQLite with SQLAlchemy models
  - `Draw` model for historical results
  - `Prediction` model for tracking predictions
  - `NumberStatistics` model for caching stats
- [x] **Web Scraper** - Scrapes from lottolyzer.com
  - 1805 historical draws loaded (full history)
  - Supports pagination and bulk loading
- [x] **Statistical Analysis**
  - Number frequency analysis
  - Hot/cold number detection
  - Odd/even distribution
  - High/low distribution
  - Sum statistics
- [x] **Pattern Detection**
  - Consecutive number patterns
  - Frequent number pairs/triples
  - Gap analysis (draws since last appearance)
  - Day-of-week patterns (Monday vs Thursday)
- [x] **Prediction Engine** - 5 strategies implemented:
  - Hot Numbers (frequency-based)
  - Cold Numbers (overdue-based)
  - Balanced (statistical distribution)
  - ML/LSTM (neural network) - trained with TensorFlow
  - Ensemble (combines all strategies)
- [x] **Web Dashboard** - 4 pages with Tailwind CSS
  - Dashboard (/) - overview, quick predict
  - History (/history) - paginated draw list
  - Statistics (/statistics) - charts with Plotly
  - Predict (/predict) - strategy selection UI
- [x] **REST API** - Full CRUD endpoints
  - GET /api/health
  - GET /api/draws
  - GET /api/statistics
  - GET /api/patterns
  - GET /api/scheduler
  - POST /api/predict
  - POST /api/scrape
  - POST /api/train
- [x] **Scheduled Scraping** - Automatic updates via APScheduler
  - Monday 7:00pm SGT (after draw)
  - Thursday 7:00pm SGT (after draw)
  - Daily 8:00am SGT (catch-up check)

### Database Status
- **1805 draws** loaded (full history)
- Date range: 1997 to 2026-01-05
- Latest draw: #4145 on 2026-01-05

---

## Next Steps (TODO)

### High Priority

- [ ] **Prediction Tracking** - Save predictions and compare with actual results
  - Store predictions in database before each draw
  - After draw, compare predicted vs actual numbers
  - Track hit rate per strategy over time

- [ ] **Backtest System** - Test strategies against historical data
  - Simulate predictions on past draws
  - Calculate accuracy metrics per strategy
  - Generate performance reports

### Medium Priority

- [ ] **Add More ML Models**
  - Random Forest classifier
  - XGBoost
  - Transformer-based model

### Low Priority / Enhancements

- [ ] **User Accounts** - Save favorite numbers, prediction history
- [ ] **Email Notifications** - Send predictions before each draw
- [ ] **Mobile-Responsive** - Improve mobile UI
- [ ] **Export Features** - Download predictions as PDF/CSV
- [ ] **Number Wheel** - Visual number picker
- [ ] **Comparison Tool** - Compare user picks vs predictions

---

## Completed (2026-01-07)

- [x] **Load Full History** - 1805 draws loaded
- [x] **Install TensorFlow** - LSTM model trained (50 epochs)
- [x] **Add Scheduled Scraping** - Mon/Thu 7pm & daily 8am SGT
- [x] **Add .gitignore** - Proper Python/FastAPI gitignore

---

## Quick Reference

### Run Commands
```bash
# Start server
./venv/bin/python run.py server

# Populate more data
./venv/bin/python run.py populate --max-pages 20

# Train ML model
./venv/bin/python run.py train --epochs 50
```

### Key Files
| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI entry point |
| `app/scheduler.py` | APScheduler for auto-updates |
| `app/scraper/singapore_pools.py` | Web scraper |
| `app/analysis/predictor.py` | Prediction strategies |
| `app/analysis/statistics.py` | Frequency analysis |
| `app/api/routes.py` | API endpoints |
| `data/toto.db` | SQLite database (gitignored) |

### API Examples
```bash
# Get statistics
curl http://localhost:8000/api/statistics?lookback=100

# Generate prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"strategy": "ensemble", "count": 5}'

# Scrape new data
curl -X POST http://localhost:8000/api/scrape?max_pages=5
```

---

## Notes

- Lottery prediction is for entertainment only - numbers are random
- TOTO draws: Monday & Thursday at 6:30pm Singapore time
- Jackpot odds: 1 in 13,983,816
- Data source: https://en.lottolyzer.com/history/singapore/toto/
