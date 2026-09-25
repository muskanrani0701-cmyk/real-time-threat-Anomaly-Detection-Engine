import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import keras
from keras import layers, models

import config

def build_and_save_models():
    artifacts_dir = Path(config.ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spatial CNN Model (spatial_cnn.h5)
    # SRS 3.1: Input: 224x224 RGB float32 tensor; Output: Feature vector.
    print("Generating CNN model (spatial_cnn.h5)...")
    cnn_input = layers.Input(shape=(224, 224, 3), name="frame_input")
    x = layers.Conv2D(16, (3, 3), strides=2, activation="relu", padding="same")(cnn_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    # Feature vector output (16 dimensions)
    feature_embedding = layers.Dense(16, activation="relu", name="cnn_embedding")(x)
    cnn_model = models.Model(inputs=cnn_input, outputs=feature_embedding, name="spatial_cnn")
    cnn_model.compile(optimizer="adam", loss="mse")
    cnn_model.save(config.CNN_MODEL_PATH)
    print(f"Saved CNN model to {config.CNN_MODEL_PATH}")

    # 2. Network KNN Model (network_knn.pkl)
    # SRS 3.1: Fitted Pipeline utilizing StandardScaler and NearestNeighbors/KNeighborsClassifier.
    # Features: [src_port, dst_port, packet_size, packet_rate, protocol_num]
    print("Generating KNN pipeline (network_knn.pkl)...")
    np.random.seed(42)
    # Baseline normal traffic: standard web/SSH/DNS ports, typical packet sizes
    normal_traffic_baseline = np.array([
        [443, 50000, 1200, 100, 6],   # HTTPS
        [80, 50001, 800, 80, 6],      # HTTP
        [53, 50002, 128, 20, 17],     # DNS
        [22, 50003, 350, 15, 6],      # SSH normal
        [443, 50004, 1400, 120, 6],   # HTTPS bulk
        [8080, 50005, 600, 50, 6],    # Dev web
        [53, 50006, 256, 30, 17],     # DNS query
        [123, 50007, 48, 5, 17],      # NTP
    ])
    # Expand with small noise to create baseline cluster
    expanded_baseline = np.vstack([
        normal_traffic_baseline + np.random.normal(0, [2, 100, 50, 5, 0], normal_traffic_baseline.shape)
        for _ in range(20)
    ])
    # Ensure positive ports and packet sizes
    expanded_baseline = np.clip(expanded_baseline, 1, None)

    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", NearestNeighbors(n_neighbors=5, metric="euclidean"))
    ])
    knn_pipeline.fit(expanded_baseline)
    joblib.dump(knn_pipeline, config.KNN_MODEL_PATH)
    print(f"Saved KNN pipeline to {config.KNN_MODEL_PATH}")

    # 3. ANN Fusion Model (fusion_ann.h5)
    # SRS 3.1: Input: Concatenated CNN (16) and KNN (1 distance) numerical outputs -> 17 dims.
    # Output: Sigmoid activation score representing unified threat level.
    print("Generating ANN Fusion model (fusion_ann.h5)...")
    fusion_input = layers.Input(shape=(17,), name="fusion_input")
    fx = layers.Dense(32, activation="relu")(fusion_input)
    fx = layers.Dropout(0.1)(fx)
    fx = layers.Dense(16, activation="relu")(fx)
    risk_output = layers.Dense(1, activation="sigmoid", name="risk_score")(fx)
    ann_model = models.Model(inputs=fusion_input, outputs=risk_output, name="fusion_ann")
    ann_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    ann_model.save(config.ANN_MODEL_PATH)
    print(f"Saved ANN model to {config.ANN_MODEL_PATH}")

if __name__ == "__main__":
    build_and_save_models()
