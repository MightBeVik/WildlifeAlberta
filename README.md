# WildSight AI — Wildlife Detection & Classification

Camera trap image analysis system for the eDNA Research Lab at SAIT. Automated detection and classification of wildlife species across Alberta using computer vision.

## Tech Stack

- **Backend**: Python, FastAPI, YOLOv5 (MegaDetector), EfficientNet-B3
- **Frontend**: React, Vite
- **Pipeline**: Two-stage — detection (bounding boxes) then species classification

## Quick Start

### Backend
```bash
cd backend
..\venv\Scripts\activate   # Windows
python run.py
# API at http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# UI at http://localhost:5173
```

### Enable ML Models (optional)
```bash
pip install torch torchvision ultralytics
```

## Project Structure

```
├── backend/          # FastAPI server
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/  # upload, detect, classify
│   │   ├── services/ # detector, classifier, preprocessor
│   │   └── models/   # Pydantic schemas
│   └── requirements.txt
├── frontend/         # React + Vite
│   └── src/
│       ├── components/
│       └── services/
├── data/             # Dataset storage
└── outputs/          # Predictions and reports
```

## Team

Group 6 — Barile, Vik, Nate, Osele, Max
