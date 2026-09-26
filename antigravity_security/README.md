# Project Antigravity: Real-Time Mobile Face Recognition & Intruder Alerting System

A production-grade, end-to-end mobile security platform that detects and discriminates between authorized personnel and unauthorized intruders in real-time video streams.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Mobile ["Flutter Cross-Platform Mobile Client"]
        A[Login / Register] --> B[OS Permissions: Camera & Push Notifications]
        B --> C[Add Authorized Personnel: Capture Face]
        B --> D[Live Camera Feed Connection: Native / IP Camera]
        D -->|WebSocket Base64 Frames 10-15 FPS| E[Real-Time Video Streamer]
        G[HUD Overlay Engine: Green/Red Bounding Box] <--|Detections & Alert Payloads| E
        H[Siren Audio Alarm & Push Notification] <--|Intruder Flagged| G
    end

    subgraph Backend ["FastAPI Async Server & Face Engine"]
        E <-->|WS /ws/live-stream| I[FastAPI WebSocket Handler]
        C -->|POST /api/v1/authorized/upload| J[Authorized Face Enrollment Endpoint]
        J --> K[Face Localization & 160x160 Alignment]
        I --> K
        K --> L[FaceNet PyTorch 512-D Embedding Extraction]
        L --> M[L2-Normalized Hypersphere Mapping ||v||=1]
        M --> N[(In-Memory Vector Cache: Zero-Latency Lookup)]
        N <-->|Cosine Similarity Sim >= 0.68| O{Decision Boundary}
        O -->|Match Found| P[Authorized: Return Name & Confidence]
        O -->|No Match / Sim < 0.68| Q[INTRUDER ALERT: Save Snapshot & Fire Notification]
    end

    subgraph Database ["Persistence Layer"]
        J --> R[(SQLite / PostgreSQL + pgvector)]
        Q --> S[(Intruder Alert Logs & Image Storage)]
    end

    subgraph ML_Pipeline ["Authentic Kaggle Training Pipeline"]
        T[Kaggle API: jessicali9530/lfw-dataset] -->|Authentic Real Human Faces Only| U[LFW Triplet Loader A-P-N]
        U --> V[PyTorch Triplet Margin Loss Training]
        V --> W[Model Checkpoint facenet_lfw_best.pth & ONNX Export]
        W --> L
    end
```

---

## 📂 Repository Structure

```
antigravity_security/
├── ml_pipeline/
│   ├── fetch_kaggle_lfw.py        # Kaggle API downloader for authentic LFW human faces
│   ├── dataset.py                 # PyTorch Triplet & Verification Pair dataset loaders
│   ├── model.py                   # 512-D L2-normalized FaceNet backbone & Cosine Triplet Loss
│   ├── train.py                   # PyTorch training loop with mixed precision & ONNX export
│   ├── evaluate.py                # ROC-AUC, Equal Error Rate (EER), & optimal threshold optimization
│   └── requirements_ml.txt        # ML dependencies
├── backend/
│   ├── app/
│   │   ├── config.py              # App settings, secrets, and operational similarity threshold
│   │   ├── database.py            # SQLAlchemy engine, session maker, SQLite/PostgreSQL
│   │   ├── models.py              # User, AuthorizedPersonnel, and IntruderAlert ORM entities
│   │   ├── schemas.py             # Pydantic v2 schemas for requests & responses
│   │   ├── auth.py                # JWT auth, password hashing, OAuth2 scheme
│   │   ├── face_engine.py         # OpenCV detector + FaceNet embedding + in-memory vector cache
│   │   └── main.py                # FastAPI endpoints, real-time WebSocket, alert logs
│   └── requirements.txt           # Backend dependencies
└── mobile/
    ├── pubspec.yaml               # Flutter packages (camera, websocket, notifications, audio)
    └── lib/
        ├── main.dart              # App entrypoint & dark theme configuration
        ├── models/
        │   └── detection_result.dart # Detection event & bounding box models
        ├── services/
        │   ├── api_service.dart   # REST client for Auth, Upload, and Lists
        │   └── alert_service.dart # Sound siren audio & local push notification trigger
        └── screens/
            ├── auth_screen.dart   # Step 1: Login / Sign Up
            ├── permission_screen.dart # Step 2: Camera & Notification OS permissions
            ├── add_authorized_screen.dart # Step 3: Face capture & instant 512-D embedding enrollment
            └── live_camera_screen.dart    # Step 4 & 5: Live camera streaming & real-time intruder alerting
```

---

## 🚀 Step-by-Step Execution Guide

### 1. ML Pipeline: Kaggle Ingestion & Training

```bash
cd antigravity_security/ml_pipeline
pip install -r requirements_ml.txt

# Set your Kaggle API credentials
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"

# 1. Download authentic LFW dataset (Zero AI/synthetic images)
python fetch_kaggle_lfw.py

# 2. Train FaceNet embedding model using Triplet Loss
python train.py --epochs 15 --batch_size 32 --lr 1e-4

# 3. Evaluate ROC-AUC and compute optimal cosine decision threshold
python evaluate.py
```

### 2. Backend Server Launch

```bash
cd antigravity_security/backend
pip install -r requirements.txt

# Run FastAPI server on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at `http://localhost:8000/docs`.

### 3. Flutter Mobile App Launch

```bash
cd antigravity_security/mobile
flutter pub get

# Launch on connected Android/iOS device or emulator
flutter run
```
