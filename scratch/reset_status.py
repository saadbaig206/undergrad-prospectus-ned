import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import update_ingestion_status

print("Resetting database ingestion status to idle...")
try:
    update_ingestion_status("undergraduate", "idle", None)
    print("Undergraduate status reset to idle.")
except Exception as e:
    print(f"Error resetting undergraduate: {e}")

try:
    update_ingestion_status("postgraduate", "idle", None)
    print("Postgraduate status reset to idle.")
except Exception as e:
    print(f"Error resetting postgraduate: {e}")

print("Done.")
