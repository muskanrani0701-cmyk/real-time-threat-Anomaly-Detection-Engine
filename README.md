# Real-Time Threat & Anomaly Detection Engine (Multimodal Fusion Engine)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/Backend-Flask%20%7C%20Flask--SocketIO-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Broker](https://img.shields.io/badge/Broker-Redis-red.svg)](https://redis.io/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Deep Learning](https://img.shields.io/badge/DL-TensorFlow%20%2F%20Keras-yellow.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, real-time cyber-physical anomaly detection and threat fusion engine. It ingests parallel streams of physical security video and network telemetry, evaluates them through isolated machine learning pipelines, and correlates findings via a neural network fusion layer to flag multi-vector threats with sub-second latency.

---

## Table of Contents
1. [Executive Summary & Problem Statement](#executive-summary--problem-statement)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Generating Baseline Model Artifacts](#generating-baseline-model-artifacts)
7. [Running the Application](#running-the-application)
   - [Starting the Background Workers](#1-start-the-inference-workers)
   - [Starting the API Server](#2-start-the-flask-api-server)
   - [Simulating Live Traffic](#3-simulate-live-telemetry--traffic)
8. [API Reference](#api-reference)
9. [Multimodal Fusion & Temporal Windowing](#multimodal-fusion--temporal-windowing)
10. [Performance Targets & SLAs](#performance-targets--slas)
11. [Running Tests](#running-tests)
12. [License](#license)

---

## Executive Summary & Problem Statement

* **Target Audience**: Security Operations Center (SOC) analysts and IT infrastructure managers.
* **Core Problem**: Traditional security systems operate in silos. A physical breach (e.g., unauthorized server room entry) and an unauthorized SSH connection or port scan are handled by separate tools, delaying triage and failing to recognize coordinated, multi-vector attacks.
* **Solution**: A decoupled, event-driven engine that ingests video frames and network telemetry asynchronously, extracts spatial and behavioral feature vectors, and fuses them within a temporal sliding window ($2.0$s) to compute a unified threat risk score ($\ge 0.7 \implies$ Anomaly Alert).

---

## System Architecture

The engine strictly adheres to a decoupled architecture separating ingestion, queueing, model inference, and persistence:

```
[ RTSP / Camera Feed ]                 [ Network Telemetry / Syslog ]
          │                                           │
          ▼ Binary Frames                             ▼ JSON Telemetry
┌────────────────────────────────────────────────────────────────────────┐
│                        API INGESTION LAYER                             │
│   • WebSocket: /ws/stream/video (Flask-SocketIO)                       │
│   • REST:      /api/v1/network/log (Flask POST)                        │
│   • Instant payload validation (Pydantic) & Enqueue (≤ 20ms p95)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ LPUSH / Streams
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     MESSAGE BROKER (REDIS)                             │
│   • Queues: video_stream  |  network_stream                            │
│   • Temporal State Cache (Sliding Window Δt ≤ 2.0s)                    │
│   • Pub/Sub Channel: threat_alerts                                     │
└─────────────────┬────────────────────────────────────┬─────────────────┘
                  │                                    │
                  ▼ BLPOP                              ▼ BLPOP
┌───────────────────────────────────┐ ┌──────────────────────────────────┐
│        CNN INFERENCE WORKER       │ │        KNN INFERENCE WORKER      │
│ • Model: spatial_cnn.h5           │ │ • Model: network_knn.pkl         │
│ • Startup Preload (Zero disk I/O) │ │ • Startup Preload                │
│ • Input: 224x224 RGB Float32      │ │ • StandardScaler Pipeline        │
│ • Emits: 16-dim spatial embedding │ │ • Computes distance to baseline  │
└─────────────────┬─────────────────┘ └────────────────┬─────────────────┘
                  │                                    │
                  └─────────────────┬──────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   ANN FUSION & SCORING WORKER                          │
│   • Temporal Window Correlation: Correlates events within 2.0s         │
│   • Model: fusion_ann.h5 (Keras)                                       │
│   • Input: Concatenated CNN embedding (16) + KNN distance (1) = 17     │
│   • Output: Unified Risk Score ∈ [0.0, 1.0] (Sigmoid activation)       │
│   • SLA: End-to-end inference latency ≤ 500ms                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Risk Score ≥ 0.7
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     PERSISTENCE & ALERTING                             │
│   • Database (SQLite / events.db): events table                        │
│   • Broadcast alerts via Redis Pub/Sub (threat_alerts)                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Description |
| :--- | :--- | :--- |
| **Backend & API** | **Python 3.10+ / Flask** | RESTful ingestion API with non-blocking enqueuing |
| **Streaming & Sockets**| **Flask-SocketIO / simple-websocket** | Low-latency binary frame ingestion over WebSocket |
| **Validation** | **Pydantic v2** | High-performance schema validation for telemetry |
| **Message Broker** | **Redis (redis-py + fakeredis fallback)** | Decoupled queues (`video_stream`, `network_stream`) and Pub/Sub |
| **Machine Learning** | **scikit-learn / joblib** | K-Nearest Neighbors (`NearestNeighbors`, `StandardScaler`) for network clustering |
| **Deep Learning** | **TensorFlow / Keras 3** | Spatial CNN (`spatial_cnn.h5`) & Multimodal ANN Fusion (`fusion_ann.h5`) |
| **Vision & Image** | **OpenCV (`opencv-python-headless`) / Pillow** | Frame decoding, resizing ($224 \times 224$), RGB normalization |
| **Persistence** | **SQLite / SQLAlchemy** | Stores flagged anomalies in the `events` table |
| **Testing** | **pytest / pytest-asyncio / requests** | End-to-end latency benchmarks and unit verification |

---

## Project Structure

```
project_1/
├── app.py                     # Flask & SocketIO application (REST + WebSocket endpoints)
├── broker.py                  # Redis client with queues, Pub/Sub, and sliding temporal cache
├── config.py                  # Central configuration (timeouts, thresholds, queue names)
├── db.py                      # SQLite database schema and insertion helpers
├── generate_models.py         # Trains and serializes baseline CNN, KNN, and ANN models
├── requirements.txt           # Pinned production and development dependencies
├── schemas.py                 # Pydantic models for request validation
├── simulate_traffic.py        # Client traffic simulator for dual ingestion
├── workers.py                 # Standalone/multithreaded inference workers (CNN, KNN, ANN)
├── LICENSE                    # Open-source MIT License
├── README.md                  # Complete project documentation
├── .gitignore                 # Standard Python/Flask/ML gitignore
├── ml_engine/
│   └── artifacts/             # Serialized model weights matching SRS 3.1
│       ├── spatial_cnn.h5     # Keras CNN (224x224 RGB -> 16-dim embedding)
│       ├── network_knn.pkl    # scikit-learn Pipeline (Scaler + KNN)
│       └── fusion_ann.h5      # Keras ANN (17-dim -> Sigmoid risk score)
└── tests/
    └── test_pipeline.py       # Full integration and latency test suite
```

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10, 3.11, 3.12, 3.13, or 3.14.
* Optional: A running Redis server on `localhost:6379`. *(Note: If Redis is not detected, `broker.py` automatically falls back to an embedded in-memory Redis instance with an identical API).*

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone <repository_url>
cd project_1

# Create and activate virtual environment (optional but recommended)
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Generating Baseline Model Artifacts

Before running the workers for the first time, generate the baseline model files adhering to **SRS Section 3.1**:

```bash
python generate_models.py
```
This produces:
* `ml_engine/artifacts/spatial_cnn.h5`
* `ml_engine/artifacts/network_knn.pkl`
* `ml_engine/artifacts/fusion_ann.h5`

---

## Running the Application

For a fully decoupled runtime, run the workers and the API server in separate terminal windows:

### 1. Start the Inference Workers
Launch the dedicated background inference workers. Models will be loaded into memory **once at startup** (eliminating disk I/O latency):

```bash
python workers.py --worker all
```
*Options:*
* `--worker all`: Starts CNN, KNN, and ANN Fusion workers concurrently.
* `--worker cnn`: Runs only the video frame worker.
* `--worker knn`: Runs only the network telemetry worker.
* `--worker fusion`: Runs only the fusion scoring worker.

### 2. Start the Flask API Server
In a separate terminal, start the high-throughput ingestion server:

```bash
python app.py
```
The server will bind to `http://0.0.0.0:5000` and expose both REST and WebSocket channels.

### 3. Simulate Live Telemetry & Traffic
In a third terminal, run the simulated traffic script to send sample normal and anomalous telemetry to the API:

```bash
python simulate_traffic.py
```

---

## API Reference

### 1. Network Telemetry Ingestion (REST)
* **Endpoint**: `POST /api/v1/network/log`
* **Content-Type**: `application/json`
* **Response Status**: `202 Accepted`
* **Request Payload**:
  ```json
  {
    "source_id": "sensor_cyber_1",
    "src_ip": "45.33.32.156",
    "dst_ip": "10.0.0.5",
    "src_port": 64222,
    "dst_port": 22,
    "protocol": "TCP",
    "packet_size": 48,
    "packet_rate": 4500.0,
    "flags": "SYN"
  }
  ```
* **Sample Response**:
  ```json
  {
    "status": "enqueued",
    "queue": "network_stream",
    "source_id": "sensor_cyber_1",
    "enqueue_latency_ms": 1.25,
    "total_api_latency_ms": 2.11
  }
  ```

### 2. Video Frame Ingestion (WebSocket)
* **Namespace**: `/ws/stream/video`
* **Event**: `frame`
* **Payload**: Binary frame data (JPEG/PNG buffer) or a JSON dictionary containing base64 data and metadata (`source_id`, `timestamp`).
* **Acknowledgment Event**: `frame_ack`
  ```json
  {
    "status": "enqueued",
    "source_id": "camera_east_gate",
    "enqueue_latency_ms": 1.48,
    "total_latency_ms": 2.30
  }
  ```

### 3. Query Flagged Anomalies
* **Endpoint**: `GET /api/v1/events?limit=20`
* **Sample Response**:
  ```json
  {
    "count": 1,
    "events": [
      {
        "event_id": 1,
        "source_id": "sensor_cyber_1",
        "captured_at": "2026-09-25T05:24:12Z",
        "risk_score": 0.8842,
        "severity": "HIGH",
        "knn_distance": 4.12,
        "cnn_score": 1.25
      }
    ]
  }
  ```

### 4. Health Check
* **Endpoint**: `GET /api/v1/health`
* **Response Status**: `200 OK`

---

## Multimodal Fusion & Temporal Windowing

* **Temporal Window Buffer**: Network events and video frames arrive asynchronously. When an event arrives, the system queries the temporal state cache for recent counterpart data belonging to the same `source_id` within a configurable sliding window (default `2.0` seconds).
* **ANN Meta-Layer**:
  - Concatenates the $16$-dimensional CNN spatial embedding with the $1$-dimensional normalized KNN distance into a $17$-dimensional vector.
  - Passes the vector through `fusion_ann.h5` to output a unified continuous risk score $\in [0.0, 1.0]$.
* **Alerting Logic**:
  - `risk_score >= 0.90`: `CRITICAL`
  - `risk_score >= 0.70`: `HIGH` (flagged as Anomaly per SRS FR-3.2)
  - `risk_score >= 0.40`: `MEDIUM`
  - `risk_score < 0.40`: `LOW`
  - Any event with `risk_score >= 0.70` is automatically committed to the database and published to the `threat_alerts` Pub/Sub channel.

---

## Performance Targets & SLAs

* **NFR-1 (Ingestion Latency)**: Ingestion-to-enqueue latency strictly $\le 20$ ms at p95 (measured at $\approx 1$–$3$ ms).
* **NFR-1 (Inference Latency)**: End-to-end inference latency strictly $\le 500$ ms.
* **NFR-2 (Reliability)**: Redis connection includes `health_check_interval=30` and socket timeout handling; worker loops catch socket exceptions and safely continue processing.
* **NFR-3 (Decoupling)**: API threads never invoke ML/DL inference; heavy workloads are strictly offloaded to worker processes.

---

## Running Tests

Run the automated test suite using `pytest`:

```bash
pytest tests/test_pipeline.py -v
```

This verifies:
1. API Health endpoint operational status.
2. Ingestion latency and Pydantic validation.
3. Preload and inference of CNN, KNN, and ANN models.
4. Temporal window fusion logic.
5. SQLite anomaly database persistence and retrieval.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
