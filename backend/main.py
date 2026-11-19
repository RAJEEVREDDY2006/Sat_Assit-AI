"""
SatAssist FastAPI Application Server
Author: Sat_Assist Team

AI-Powered Remote Sensing & Satellite Image Question-Answering Chatbot Backend
"""

import os
import torch
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import upload, chat, samples, annotate, bitemporal
from backend.storage.session_store import session_store
from backend.model.vlm_engine import vlm_engine

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
DATASET_IMAGES_DIR = BASE_DIR / "data" / "vrsbench_5k" / "images"
ANNOTATED_DIR = BASE_DIR / "data" / "annotated"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="🛰️ SatAssist AI - Satellite Image QA & Grounding API",
    description="Vision-Language AI system for Remote Sensing Imagery understanding, VQA, and Visual Grounding using VRSBench.",
    version="1.0.0"
)

# Enable CORS for Frontend (Vite / React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files for direct image rendering
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/dataset_images", StaticFiles(directory=str(DATASET_IMAGES_DIR)), name="dataset_images")
app.mount("/annotated", StaticFiles(directory=str(ANNOTATED_DIR)), name="annotated")

# Register API Routers
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(samples.router)
app.include_router(annotate.router)
app.include_router(bitemporal.router)

@app.get("/")
def root():
    return {
        "service": "SatAssist Remote Sensing AI Chatbot API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/api/health")
def health_check():
    cuda_avail = torch.cuda.is_available()
    gpu_info = "N/A"
    vram_usage = "N/A"

    if cuda_avail:
        try:
            gpu_info = torch.cuda.get_device_name(0)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            vram_usage = f"Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB"
        except Exception:
            pass

    return {
        "status": "healthy",
        "cuda_available": cuda_avail,
        "device": gpu_info if cuda_avail else "CPU",
        "vram_usage": vram_usage,
        "model_loaded": vlm_engine.is_loaded,
        "active_sessions": len(session_store.sessions)
    }

@app.on_event("startup")
def startup_preload_model():
    """Warms up the VLM model in a background thread on server startup."""
    import threading
    def _warmup():
        try:
            print("[SatAssist] Warming up VLM model on GPU...")
            vlm_engine.load_model()
            print("🚀 [SatAssist] VLM model warmed up and ready for instant inference!")
        except Exception as e:
            print(f"[SatAssist] Model warmup warning: {e}")
    threading.Thread(target=_warmup, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    print("🛰️ Starting SatAssist FastAPI Server on http://127.0.0.1:8001 ...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)
