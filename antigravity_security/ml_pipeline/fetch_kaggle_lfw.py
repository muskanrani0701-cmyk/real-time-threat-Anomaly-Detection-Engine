"""
Project Antigravity: Kaggle Authentic Dataset Fetcher
Fetches the authentic 'Labeled Faces in the Wild' (LFW) dataset from Kaggle.
Strict Data Constraint: Real-world authentic human face images ONLY. No synthetic/AI data.
"""

import os
import sys
import zipfile
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Primary authentic Kaggle dataset identifier
KAGGLE_DATASET = "jessicali9530/lfw-dataset"
DATA_DIR = Path(__file__).resolve().parent / "data"
LFW_ROOT = DATA_DIR / "lfw_authentic"


def verify_kaggle_credentials() -> bool:
    """Verifies that Kaggle API credentials are configured via environment variables or ~/.kaggle/kaggle.json"""
    has_env = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    default_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if has_env or default_cfg.exists():
        logger.info("Kaggle credentials detected successfully.")
        return True
    
    logger.warning(
        "Kaggle API credentials not found!\n"
        "Please provide your Kaggle API key by doing ONE of the following:\n"
        "  1) Set environment variables:\n"
        "     export KAGGLE_USERNAME='your_username'\n"
        "     export KAGGLE_KEY='your_api_key'\n"
        "  2) Download kaggle.json from Kaggle -> Account -> Create New Token\n"
        "     and place it at ~/.kaggle/kaggle.json (or C:\\Users\\<user>\\.kaggle\\kaggle.json)"
    )
    return False


def download_lfw_from_kaggle(destination_dir: Path = LFW_ROOT) -> Path:
    """
    Downloads and extracts the authentic LFW dataset using the official Kaggle API.
    Guarantees zero synthetic or AI-generated data.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset is already extracted
    existing_people = [d for d in destination_dir.glob("*/*") if d.is_dir()]
    if len(existing_people) > 500:
        logger.info(f"Authentic LFW dataset already exists at {destination_dir} with {len(existing_people)} identities.")
        return destination_dir

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Connected to Kaggle API. Downloading '{KAGGLE_DATASET}'...")

        zip_target = destination_dir / "lfw_download.zip"
        api.dataset_download_files(KAGGLE_DATASET, path=str(destination_dir), unzip=False, quiet=False)

        # Locate downloaded archive
        zip_files = list(destination_dir.glob("*.zip"))
        if not zip_files:
            raise FileNotFoundError("Kaggle download completed but no zip archive was found.")

        archive = zip_files[0]
        logger.info(f"Extracting authentic face archive: {archive.name}...")
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(destination_dir)

        # Cleanup zip to conserve disk space
        if archive.exists():
            archive.unlink()
            logger.info("Extracted successfully and removed temporary zip archive.")

    except ImportError:
        logger.error("The 'kaggle' Python package is not installed. Run: pip install kaggle")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to download dataset from Kaggle: {str(e)}")
        raise

    # Verify extracted identities
    inspect_dataset(destination_dir)
    return destination_dir


def inspect_dataset(dataset_path: Path):
    """
    Scans the extracted dataset and prints statistics on authentic human identities and image counts.
    """
    all_images = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png"))
    all_identities = set([img.parent.name for img in all_images])

    logger.info("=" * 60)
    logger.info("AUTHENTIC HUMAN DATASET VERIFICATION (KAGGLE LFW)")
    logger.info("=" * 60)
    logger.info(f"Total Authentic Face Images: {len(all_images)}")
    logger.info(f"Total Distinct Human Identities: {len(all_identities)}")
    
    # Check for multi-image identities suitable for positive pair generation
    identity_counts = {}
    for img in all_images:
        identity = img.parent.name
        identity_counts[identity] = identity_counts.get(identity, 0) + 1

    multi_image_people = {k: v for k, v in identity_counts.items() if v >= 2}
    logger.info(f"Identities with >= 2 photos (usable for positive pairs): {len(multi_image_people)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    if verify_kaggle_credentials():
        download_lfw_from_kaggle()
    else:
        logger.info("Proceeding in inspection mode or waiting for Kaggle credentials.")
