# 🛰️ SatAssist AI — Satellite Image Question-Answering & Visual Grounding Chatbot

An AI-powered Vision-Language chatbot designed specifically for **remote sensing and satellite imagery analysis**, built on the **VRSBench** benchmark dataset.

---

## 🌟 Key Features

1. **Satellite Image Upload & Normalization**:
   - Supports **JPG, PNG, WebP, and multi-band 16-bit TIFF / GeoTIFF** remote sensing imagery.
   - Dynamic 2%–98% percentile contrast stretching for aerial sensors.
2. **Visual Question Answering (VQA)**:
   - Multi-turn conversational questions about terrain, airfields, harbors, storage tanks, road networks, and buildings.
3. **Satellite Image Captioning**:
   - Generates comprehensive remote sensing scene summaries.
4. **Visual Grounding & Localization**:
   - Detects requested objects (aircraft, ships, buildings) and highlights them with **interactive glowing bounding box overlays** on the satellite canvas.
5. **VRSBench 5,000 Curated Dataset**:
   - Built with **5,000 unique satellite images** sampled with `seed=42` across airports, harbors, urban centers, industrial zones, and agricultural areas.
   - 30,107 VQA question-answer pairs and grounding annotations.
6. **1-Click Preset Gallery**:
   - Explore and test preset satellite scenes directly from your downloaded dataset in the web UI.

---

## 🏗️ Architecture

```
User (Browser)
      │
      ▼
React + Vite Frontend (http://localhost:5173)
  ├── Satellite Viewport (Interactive Zoom, Pan, SVG Bounding Boxes)
  ├── AI Chat Panel (Multi-Turn VQA, Task Modes, Suggested Prompts)
  └── VRSBench Preset Gallery (1-Click Sample Testing)
      │
      ▼
FastAPI Backend (http://127.0.0.1:8000)
  ├── POST /api/upload (Multi-Format Image Normalizer)
  ├── POST /api/chat (VLM Conversation & Grounding Parser)
  ├── GET /api/samples (Local Dataset Sample Loader)
  └── GET /api/health (GPU VRAM Telemetry)
      │
      ▼
Vision-Language Model (VLM) Engine (NVIDIA RTX 3050 - 6 GB VRAM)
```

---

## 🚀 Quick Start Guide

### 1. Start Both Backend & Frontend (1-Click)

Double-click or run from PowerShell / Command Prompt:

```powershell
.\run_app.bat
```

### 2. Manual Startup

**Terminal 1 (Backend):**
```powershell
.venv\Scripts\python.exe run_backend.py
```
*Backend runs on `http://127.0.0.1:8000` (API documentation at `http://127.0.0.1:8000/docs`).*

**Terminal 2 (Frontend):**
```powershell
cd frontend
npm run dev
```
*Frontend runs on `http://localhost:5173`.*

---

## 📁 Directory Structure

```
c:\Sat_Assist\
├── backend/
│   ├── model/
│   │   ├── bounding_box_parser.py   # VRSBench & Qwen bounding box parser
│   │   ├── image_processor.py       # TIFF/GeoTIFF 16-bit normalizer
│   │   └── vlm_engine.py            # Remote Sensing VLM inference engine
│   ├── routers/
│   │   ├── upload.py                # Image upload handler
│   │   ├── chat.py                  # Multi-turn VQA & chat handler
│   │   └── samples.py               # VRSBench dataset sample loader
│   ├── storage/
│   │   └── session_store.py         # In-memory session manager
│   └── main.py                      # FastAPI server
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx           # Hardware telemetry & status
│   │   │   ├── SatelliteViewer.jsx  # Pan/Zoom viewer & SVG grounding layer
│   │   │   ├── ChatPanel.jsx        # Multi-turn chat & prompt suggestions
│   │   │   └── SampleGallery.jsx    # 1-click dataset sample drawer
│   │   ├── App.jsx                  # Main dashboard state orchestrator
│   │   └── index.css                # Geospatial dark theme
├── data/
│   └── vrsbench_5k/
│       ├── images/                  # 5,000 satellite images (512x512)
│       ├── vrsbench_5k_train.json   # 4,000 training samples
│       ├── vrsbench_5k_val.json     # 1,000 validation samples
│       └── dataset_stats.json       # Dataset metrics
├── scripts/
│   ├── download_vrsbench_5k.py      # Seed=42 dataset curator
│   ├── test_vlm_engine.py           # Engine verification script
│   └── visualize_samples.py         # Terminal visualizer
├── requirements.txt
├── run_backend.py
└── run_app.bat
```
