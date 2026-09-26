"""
Project Antigravity: Evaluation, ROC-AUC, and Threshold Optimization
Evaluates authentic face verification pairs, calculates Equal Error Rate (EER),
and determines the optimal Cosine Similarity threshold for intruder detection.
"""

import argparse
import logging
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

from dataset import AuthenticLFWVerificationDataset
from model import FaceNetEmbeddingModel, compute_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalPipeline")


def compute_eer(y_true, y_scores):
    """
    Computes Equal Error Rate (EER) and the optimal threshold
    where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # Find the index where FPR and FNR are closest
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    optimal_threshold = thresholds[eer_idx]

    return eer, optimal_threshold, fpr, tpr


def main():
    parser = argparse.ArgumentParser(description="Evaluate FaceNet Model on LFW Authentic Pairs")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/facenet_lfw_best.pth")
    parser.add_argument("--data_dir", type=str, default="./data/lfw_authentic")
    parser.add_argument("--num_pairs", type=int, default=3000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running evaluation on device: {device}")

    # Load model
    model = FaceNetEmbeddingModel(embedding_dim=512, backbone_name="resnet50")
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded trained checkpoint from {args.checkpoint}")
    else:
        logger.warning(f"Checkpoint {args.checkpoint} not found. Running with base pretrained weights.")

    model.to(device)
    model.eval()

    if not Path(args.data_dir).exists():
        logger.error(f"Dataset root {args.data_dir} does not exist. Run fetch_kaggle_lfw.py first.")
        return

    val_dataset = AuthenticLFWVerificationDataset(dataset_root=args.data_dir, num_pairs=args.num_pairs)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    y_true = []
    y_scores = []

    logger.info("Computing embeddings on authentic human face verification pairs...")
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)

            emb1 = model(img1)
            emb2 = model(img2)
            sim = compute_similarity(emb1, emb2)

            y_scores.extend(sim.cpu().numpy())
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
    eer, optimal_thresh, _, _ = compute_eer(y_true, y_scores)

    logger.info("=" * 60)
    logger.info("AUTHENTIC FACE VERIFICATION BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"ROC-AUC Score:                {roc_auc:.4f}")
    logger.info(f"Equal Error Rate (EER):        {eer * 100:.2f}%")
    logger.info(f"Optimal Cosine Threshold:      {optimal_thresh:.4f}")
    logger.info(f"Operational Intruder Decision: If Cosine Sim < {optimal_thresh:.4f} -> FLAG INTRUDER")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
