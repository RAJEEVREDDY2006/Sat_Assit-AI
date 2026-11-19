"""
SatAssist Backend Runner
Author: Sat_Assist Team
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("=" * 65)
    print("🛰️  SatAssist AI - Remote Sensing QA & Grounding Server")
    print("📍 Local URL: http://127.0.0.1:8001")
    print("📖 API Documentation: http://127.0.0.1:8001/docs")
    print("=" * 65)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8001, reload=False)
