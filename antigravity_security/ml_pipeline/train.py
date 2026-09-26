"""
Project Antigravity: PyTorch Model Training & Fine-Tuning Pipeline
Trains the face embedding extractor on authentic Kaggle LFW face images using Triplet Loss.
Strict Data Constraint: Authentic human faces only. Zero synthetic data.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import AuthenticLFWTripletDataset, AuthenticLFWVerificationDataset
from model import FaceNetEmbeddingModel, CosineTripletMarginLoss, compute_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainPipeline")


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, (anchors, positives, negatives) in enumerate(loader):
        anchors = anchors.to(device, non_blocking=True)
        positives = positives.to(device, non_blocking=True)
        negatives = negatives.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Automatic Mixed Precision
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            emb_anchor = model(anchors)
            emb_pos = model(positives)
            emb_neg = model(negatives)
            loss = criterion(emb_anchor, emb_pos, emb_neg)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % 25 == 0:
            logger.info(f"Step [{step + 1}/{len(loader)}] - Triplet Loss: {loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_verification(model, val_loader, device, threshold: float = 0.65):
    """Evaluates face verification accuracy on authentic human face pairs."""
    model.eval()
    correct = 0
    total = 0
    pos_sims = []
    neg_sims = []

    for img1, img2, labels in val_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        emb1 = model(img1)
        emb2 = model(img2)
        sim = compute_similarity(emb1, emb2)

        predictions = (sim >= threshold).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pos_sims.extend(sim[labels == 1].cpu().tolist())
        neg_sims.extend(sim[labels == 0].cpu().tolist())

    accuracy = correct / max(total, 1)
    mean_pos = sum(pos_sims) / max(len(pos_sims), 1)
    mean_neg = sum(neg_sims) / max(len(neg_sims), 1)

    logger.info(f"Validation - Accuracy: {accuracy * 100:.2f}% | Mean Genuine Sim: {mean_pos:.3f} | Mean Impostor Sim: {mean_neg:.3f}")
    return accuracy


def export_onnx(model, output_path: Path, device):
    """Exports model to ONNX for ultra-low latency mobile/server deployment."""
    model.eval()
    dummy_input = torch.randn(1, 3, 160, 160, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["face_tensor"],
        output_names=["embedding_512"],
        dynamic_axes={"face_tensor": {0: "batch_size"}, "embedding_512": {0: "batch_size"}}
    )
    logger.info(f"Exported optimized ONNX model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train FaceNet on Authentic LFW Dataset")
    parser.add_argument("--data_dir", type=str, default="./data/lfw_authentic", help="Path to authentic LFW images")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.4, help="Cosine triplet loss margin")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing training on device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = FaceNetEmbeddingModel(embedding_dim=512, backbone_name="resnet50", pretrained=True).to(device)

    # Prepare Authentic Datasets
    if not Path(args.data_dir).exists():
        logger.warning(f"Data directory {args.data_dir} not found. Please run fetch_kaggle_lfw.py first.")
        logger.info("Initializing with simulated authentic structure verification.")
        return

    logger.info("Loading Authentic LFW Triplet Dataset...")
    train_dataset = AuthenticLFWTripletDataset(dataset_root=args.data_dir, num_triplets=5000, is_train=True)
    val_dataset = AuthenticLFWVerificationDataset(dataset_root=args.data_dir, num_pairs=1000)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    criterion = CosineTripletMarginLoss(margin=args.margin)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_ckpt = output_dir / "facenet_lfw_best.pth"

    logger.info(f"Beginning training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()
        logger.info(f"Epoch [{epoch}/{args.epochs}] Completed - Average Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        acc = evaluate_verification(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "accuracy": best_acc,
                "embedding_dim": 512
            }, best_ckpt)
            logger.info(f"[*] New best model checkpoint saved: {best_ckpt} (Accuracy: {best_acc * 100:.2f}%)")

    # Export ONNX
    onnx_path = output_dir / "facenet_lfw.onnx"
    export_onnx(model, onnx_path, device)
    logger.info("ML Pipeline Training & Export Finished Successfully.")


if __name__ == "__main__":
    main()
