"""
Project Antigravity: TensorFlow / Keras 3 Triplet Training on Authentic Kaggle LFW Data
Strict Data Constraint: Authentic human faces only. Zero synthetic data.
"""

import os
import random
import logging
from pathlib import Path
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from model_tf import build_facenet_backbone, CosineTripletLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainTF")


class AuthenticLFWDataGenerator(keras.utils.PyDataset):
    """
    Keras PyDataset generating (Anchor, Positive, Negative) authentic face triplets.
    """
    def __init__(self, dataset_root: str, batch_size: int = 32, steps_per_epoch: int = 150, **kwargs):
        super().__init__(**kwargs)
        self.root = Path(dataset_root)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.identity_to_images = {}
        for item in self.root.rglob("*.jpg"):
            self.identity_to_images.setdefault(item.parent.name, []).append(item)
        for item in self.root.rglob("*.png"):
            self.identity_to_images.setdefault(item.parent.name, []).append(item)

        self.usable_identities = [k for k, v in self.identity_to_images.items() if len(v) >= 2]
        self.all_identities = list(self.identity_to_images.keys())

    def __len__(self):
        return self.steps_per_epoch

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize((160, 160))
        return np.array(img, dtype=np.float32)

    def __getitem__(self, idx):
        anchors, positives, negatives = [], [], []

        for _ in range(self.batch_size):
            # Select anchor and positive of same authentic individual
            person_a = random.choice(self.usable_identities)
            img_a_path, img_p_path = random.sample(self.identity_to_images[person_a], 2)

            # Select negative from different individual
            person_b = random.choice(self.all_identities)
            while person_b == person_a:
                person_b = random.choice(self.all_identities)
            img_n_path = random.choice(self.identity_to_images[person_b])

            anchors.append(self._load_image(img_a_path))
            positives.append(self._load_image(img_p_path))
            negatives.append(self._load_image(img_n_path))

        return {
            "anchor_input": np.array(anchors),
            "positive_input": np.array(positives),
            "negative_input": np.array(negatives),
        }, np.zeros((self.batch_size, 1))


def build_triplet_training_network(embedding_model: keras.Model) -> keras.Model:
    """Builds a Siamese 3-tower network sharing weights for Triplet Training."""
    input_shape = (160, 160, 3)
    in_a = keras.Input(shape=input_shape, name="anchor_input")
    in_p = keras.Input(shape=input_shape, name="positive_input")
    in_n = keras.Input(shape=input_shape, name="negative_input")

    emb_a = embedding_model(in_a)
    emb_p = embedding_model(in_p)
    emb_n = embedding_model(in_n)

    # Multi-backend Keras 3 stack operation
    stacked = keras.ops.stack([emb_a, emb_p, emb_n], axis=1)

    triplet_net = keras.Model(inputs=[in_a, in_p, in_n], outputs=stacked, name="TripletTrainingNetwork")
    return triplet_net


def train_model(data_dir: str = None, epochs: int = 15, batch_size: int = 32):
    base_dir = Path(__file__).resolve().parent
    if data_dir is None:
        data_dir = str(base_dir / "data" / "lfw_authentic")
    output_dir = base_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing FaceNet 512-D Embedding Model...")
    embedding_extractor = build_facenet_backbone(embedding_dim=512)
    triplet_net = build_triplet_training_network(embedding_extractor)

    loss_fn = CosineTripletLoss(margin=0.4)
    triplet_net.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn
    )

    if not Path(data_dir).exists():
        logger.warning(f"Data directory {data_dir} not found. Run fetch_kaggle_lfw.py first.")
        # Save base model architecture
        export_path = output_dir / "facenet_512_base.keras"
        embedding_extractor.save(str(export_path))
        logger.info(f"Saved base model to {export_path}")
        return

    logger.info("Loading authentic human dataset generator...")
    train_gen = AuthenticLFWDataGenerator(dataset_root=data_dir, batch_size=batch_size, steps_per_epoch=100)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "facenet_512_best.keras"),
            monitor="loss",
            save_best_only=True
        )
    ]

    logger.info(f"Starting training on authentic LFW triplets for {epochs} epochs...")
    triplet_net.fit(train_gen, epochs=epochs, callbacks=callbacks)

    # Export inference extractor
    final_path = output_dir / "facenet_512_final.keras"
    embedding_extractor.save(str(final_path))
    logger.info(f"Training complete. Standalone 512-D extractor saved to {final_path}")


if __name__ == "__main__":
    train_model()
