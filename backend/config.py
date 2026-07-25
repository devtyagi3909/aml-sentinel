"""
Configuration module for AML Sentinel.
Loads environment variables and provides application-wide settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Data paths
DATA_PATH = DATA_DIR / os.getenv("DATA_PATH", "sample_transactions.csv")
CUSTOMER_DATA_PATH = DATA_DIR / os.getenv("CUSTOMER_DATA_PATH", "sample_customers.csv")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# AML Thresholds
CTR_THRESHOLD = 10_000  # Currency Transaction Report threshold (USD)
STRUCTURING_LOWER = 8_000  # Lower bound for structuring detection
STRUCTURING_UPPER = 9_999  # Upper bound for structuring detection
STRUCTURING_COUNT_THRESHOLD = 3  # Min transactions to flag structuring
STRUCTURING_WINDOW_DAYS = 7  # Window for structuring detection

VELOCITY_SPIKE_MULTIPLIER = 5  # X times normal frequency = spike
RAPID_MOVEMENT_HOURS = 24  # Hours for rapid in/out detection
DORMANCY_DAYS = 90  # Days of inactivity before "dormant"

# Risk thresholds
RISK_LOW_UPPER = 0.3
RISK_MEDIUM_UPPER = 0.6
RISK_HIGH_UPPER = 0.8
# Above RISK_HIGH_UPPER = Critical

# High-risk jurisdictions (FATF grey/black list examples)
HIGH_RISK_COUNTRIES = [
    "KY", "PA", "VG", "BS", "BZ",  # Caribbean/Central America
    "MM", "KP", "IR", "SY",  # Sanctioned/high-risk
    "AF", "YE", "SO",  # Conflict zones
    "MT", "CY",  # EU grey-list
]

# Charts output directory
CHARTS_DIR = BASE_DIR / "output" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
