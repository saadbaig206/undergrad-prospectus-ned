import os
from dotenv import load_dotenv
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SEAT_DIST_FILE_LINK = os.getenv("SEAT_DIST_FILE_LINK", f"{API_BASE_URL}/seat_distribution.pdf")
MAX_CONTEXT_MESSAGES = 10
REWRITE_LOOKBACK = 4
MAX_CONTEXT_CHARS = 12000
MAX_PER_SECTION = 6