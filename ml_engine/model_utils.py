"""
Model Utilities for Project Antigravity
Handles automatic transparent assembly of chunked model weights (< 50MB)
to comply with GitHub's 50.00 MB maximum file size recommendation.
"""

import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger("model_utils")


def ensure_model_weights(target_path: Path) -> Path:
    """
    Checks if target model exists. If not, transparently reconstructs it from .part* files.
    Guarantees that all tracked git files remain strictly under GitHub's 50.00 MB threshold.
    """
    target = Path(target_path)
    if target.exists() and target.stat().st_size > 0:
        return target

    part1 = target.with_name(target.name + ".part1")
    part2 = target.with_name(target.name + ".part2")

    if part1.exists() and part2.exists():
        logger.info(f"Recombining sub-50MB model chunks into {target.name}...")
        temp_target = target.with_name(target.name + ".tmp")
        with open(temp_target, "wb") as out_f:
            for p in [part1, part2]:
                with open(p, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f)
        temp_target.replace(target)
        logger.info(f"Reconstructed {target.name} ({target.stat().st_size / (1024*1024):.2f} MB) successfully.")
        return target

    # Check canonical fallback in ml_engine/artifacts/
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    canonical_model = artifacts_dir / target.name
    if canonical_model.exists() and canonical_model.stat().st_size > 0:
        return canonical_model

    canonical_part1 = artifacts_dir / (target.name + ".part1")
    canonical_part2 = artifacts_dir / (target.name + ".part2")
    if canonical_part1.exists() and canonical_part2.exists():
        ensure_model_weights(canonical_model)
        return canonical_model

    return target
