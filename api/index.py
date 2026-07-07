import os
import sys

# Add the workspace root directory to sys.path so Vercel can resolve backend and core modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.api import app
