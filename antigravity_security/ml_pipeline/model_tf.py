"""
Project Antigravity: TensorFlow / Keras 3 FaceNet 512-D Embedding Model
Maps authentic 160x160 human face crops into a 512-dimensional Euclidean space
constrained to the unit hypersphere S^511 using strict L2 Normalization.
Strict Data Constraint: Authentic human faces only (Kaggle LFW).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications


def build_facenet_backbone(embedding_dim: int = 512, input_shape=(160, 160, 3)) -> keras.Model:
    """
    Constructs a high-performance deep metric learning backbone using
    ResNet50 / MobileNetV2 with a 512-D projection head and Unit L2 Normalization.
    """
    inputs = keras.Input(shape=input_shape, name="face_input_rgb")
    
    # Preprocessing: Scale pixel range to [-1.0, 1.0]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)

    # Pre-trained deep feature extractor
    base_model = applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg"
    )
    # Fine-tune upper residual stages
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    features = base_model(x)

    # 512-D Metric Projection Head
    dense_proj = layers.Dense(embedding_dim, use_bias=False, name="projection_512")(features)
    bn = layers.BatchNormalization(name="projection_bn")(dense_proj)

    # Crucial: Unit L2 Normalization ensures ||v||_2 = 1.0
    # On unit hypersphere, Cosine Similarity == Dot Product (v1 . v2)
    normalized_embeddings = layers.UnitNormalization(axis=-1, name="l2_unit_sphere")(bn)

    model = keras.Model(inputs=inputs, outputs=normalized_embeddings, name="FaceNet_512_Extractor")
    return model


class CosineTripletLoss(keras.losses.Loss):
    """
    Cosine Triplet Margin Loss:
      L(A, P, N) = max(0, cos(A, N) - cos(A, P) + margin)
    Forces genuine face cosine similarity to approach 1.0 and impostors to approach 0.0.
    """
    def __init__(self, margin: float = 0.4, name: str = "cosine_triplet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        # In triplet training, predictions are concatenated [batch, 3, 512]
        # or passed as (anchor, positive, negative)
        anchor = y_pred[:, 0, :]
        positive = y_pred[:, 1, :]
        negative = y_pred[:, 2, :]

        # Cosine similarity is the dot product of unit-normalized vectors
        sim_pos = keras.ops.sum(anchor * positive, axis=-1)
        sim_neg = keras.ops.sum(anchor * negative, axis=-1)

        loss = keras.ops.maximum(0.0, sim_neg - sim_pos + self.margin)
        return keras.ops.mean(loss)
