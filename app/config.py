"""Application configuration."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{DATA_DIR}/toto.db")
DATABASE_URL_SYNC = os.getenv("DATABASE_URL_SYNC", f"sqlite:///{DATA_DIR}/toto.db")

# Singapore Pools URLs
SINGAPORE_POOLS_BASE_URL = "https://www.singaporepools.com.sg"
TOTO_RESULTS_URL = f"{SINGAPORE_POOLS_BASE_URL}/en/product/pages/toto_results.aspx"

# TOTO Game Configuration
TOTO_MIN_NUMBER = 1
TOTO_MAX_NUMBER = 49
TOTO_NUMBERS_DRAWN = 6
TOTO_DRAW_DAYS = ["Monday", "Thursday"]

# Analysis Configuration
HOT_NUMBERS_COUNT = 10  # Top N frequent numbers
COLD_NUMBERS_COUNT = 10  # Bottom N frequent numbers
DEFAULT_LOOKBACK_DRAWS = 100  # Default number of draws for analysis

# ML Model Configuration
LSTM_SEQUENCE_LENGTH = 20  # Number of past draws to use for prediction
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
MODEL_PATH = DATA_DIR / "models"
MODEL_PATH.mkdir(exist_ok=True)

# Web Application
APP_TITLE = "Singapore TOTO Predictor"
APP_DESCRIPTION = "Predict Singapore TOTO numbers using statistical analysis and machine learning"
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
