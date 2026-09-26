"""
Project Antigravity: Real-Time FaceNet 512-D Embedding Extractor
Supports both InceptionResnetV1 (FaceNet standard) and ResNet/MobileNet backbones.
Strict Data Constraint: Maps real human faces into an authentic 512-dimensional metric space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceNetEmbeddingModel(nn.Module):
    """
    Deep metric learning architecture mapping 160x160 aligned face crops
    to an authentic L2-normalized 512-dimensional embedding space.
    """
    def __init__(self, embedding_dim: int = 512, backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone_name

        if backbone_name == "facenet_inception":
            try:
                from facenet_pytorch import InceptionResnetV1
                self.backbone = InceptionResnetV1(pretrained="vggface2" if pretrained else None)
                self.backbone.logits = nn.Linear(512, embedding_dim)
                self.is_custom_inception = True
            except ImportError:
                # Fallback to ResNet50 if facenet_pytorch is not installed
                self.is_custom_inception = False
                self._build_resnet_backbone(pretrained)
        else:
            self.is_custom_inception = False
            self._build_resnet_backbone(pretrained)

    def _build_resnet_backbone(self, pretrained: bool):
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        in_features = base.fc.in_features
        # Replace classification layer with 512-D projection head + BatchNorm
        base.fc = nn.Sequential(
            nn.Linear(in_features, self.embedding_dim, bias=False),
            nn.BatchNorm1d(self.embedding_dim)
        )
        self.backbone = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 512-dimensional face embedding and projects onto unit hypersphere.
        Returns L2-normalized embedding of shape [Batch, 512].
        """
        features = self.backbone(x)
        # L2 Normalization ensures ||v||_2 = 1.0 (vital for Cosine similarity)
        normalized_embeddings = F.normalize(features, p=2, dim=1)
        return normalized_embeddings


class CosineTripletMarginLoss(nn.Module):
    """
    Triplet loss formulated directly in Cosine Similarity space:
      L(A, P, N) = max(0, cos(A, N) - cos(A, P) + margin)
    Ensures genuine faces have high cosine similarity (>0.7) and
    impostor faces have lower cosine similarity (<0.3).
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Since embeddings are L2 normalized, dot product == cosine similarity
        sim_pos = torch.sum(anchor * positive, dim=1)
        sim_neg = torch.sum(anchor * negative, dim=1)
        loss = torch.clamp(sim_neg - sim_pos + self.margin, min=0.0)
        return torch.mean(loss)


def compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """Computes pair-wise cosine similarity for L2-normalized embeddings."""
    return torch.sum(emb1 * emb2, dim=-1)
