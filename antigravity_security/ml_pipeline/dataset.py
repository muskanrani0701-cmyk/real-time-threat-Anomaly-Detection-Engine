"""
Project Antigravity: PyTorch Triplet Dataset for Authentic LFW Faces
Constructs Anchor-Positive-Negative triplets from authentic real-world identities.
Strict Data Constraint: Real human faces only.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_default_transforms(img_size: int = 160, is_train: bool = True) -> transforms.Compose:
    """Standard face image preprocessing pipeline for FaceNet/Inception-ResNet."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


class AuthenticLFWTripletDataset(Dataset):
    """
    PyTorch Dataset generating (Anchor, Positive, Negative) triplets
    from the authentic LFW dataset hierarchy: root/identity_name/*.jpg
    """
    def __init__(
        self,
        dataset_root: str,
        num_triplets: int = 10000,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True
    ):
        super().__init__()
        self.root = Path(dataset_root)
        self.num_triplets = num_triplets
        self.transform = transform or get_default_transforms(is_train=is_train)
        
        # Build map: identity -> list of image filepaths
        self.identity_to_images: Dict[str, List[Path]] = {}
        self._build_index()

        # Keep only identities with >= 2 images for Anchor-Positive pairs
        self.usable_identities = [k for k, v in self.identity_to_images.items() if len(v) >= 2]
        self.all_identities = list(self.identity_to_images.keys())

        if len(self.usable_identities) < 2:
            raise ValueError(
                f"Insufficient multi-image identities found in {self.root}. "
                f"Found {len(self.usable_identities)}, need at least 2."
            )

    def _build_index(self):
        """Scans dataset root for real human face directories."""
        for item in self.root.rglob("*.jpg"):
            identity = item.parent.name
            if identity not in self.identity_to_images:
                self.identity_to_images[identity] = []
            self.identity_to_images[identity].append(item)

        # Also search .png if present
        for item in self.root.rglob("*.png"):
            identity = item.parent.name
            if identity not in self.identity_to_images:
                self.identity_to_images[identity] = []
            self.identity_to_images[identity].append(item)

    def __len__(self) -> int:
        return self.num_triplets

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples:
          - Anchor (person A, photo 1)
          - Positive (person A, photo 2)
          - Negative (person B, photo 1)
        """
        # Pick random identity with at least 2 images
        anchor_identity = random.choice(self.usable_identities)
        pos_images = self.identity_to_images[anchor_identity]
        anchor_path, positive_path = random.sample(pos_images, 2)

        # Pick random different identity for negative
        negative_identity = random.choice(self.all_identities)
        while negative_identity == anchor_identity:
            negative_identity = random.choice(self.all_identities)
        
        negative_path = random.choice(self.identity_to_images[negative_identity])

        # Load RGB images
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


class AuthenticLFWVerificationDataset(Dataset):
    """
    Pairs dataset for evaluating ROC curve, EER, and verification accuracy.
    Generates balanced genuine pairs (label=1) and impostor pairs (label=0).
    """
    def __init__(
        self,
        dataset_root: str,
        num_pairs: int = 2000,
        transform: Optional[transforms.Compose] = None
    ):
        super().__init__()
        self.root = Path(dataset_root)
        self.num_pairs = num_pairs
        self.transform = transform or get_default_transforms(is_train=False)

        self.identity_to_images: Dict[str, List[Path]] = {}
        for item in self.root.rglob("*.jpg"):
            self.identity_to_images.setdefault(item.parent.name, []).append(item)

        self.usable_identities = [k for k, v in self.identity_to_images.items() if len(v) >= 2]
        self.all_identities = list(self.identity_to_images.keys())

        # Pre-generate deterministic pairs for consistent benchmark
        self.pairs: List[Tuple[Path, Path, int]] = []
        random.seed(42)
        for i in range(num_pairs):
            if i % 2 == 0:
                # Genuine pair (label = 1)
                ident = random.choice(self.usable_identities)
                p1, p2 = random.sample(self.identity_to_images[ident], 2)
                self.pairs.append((p1, p2, 1))
            else:
                # Impostor pair (label = 0)
                id1, id2 = random.sample(self.all_identities, 2)
                p1 = random.choice(self.identity_to_images[id1])
                p2 = random.choice(self.identity_to_images[id2])
                self.pairs.append((p1, p2, 0))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1_path, p2_path, label = self.pairs[idx]
        img1 = Image.open(p1_path).convert("RGB")
        img2 = Image.open(p2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)
