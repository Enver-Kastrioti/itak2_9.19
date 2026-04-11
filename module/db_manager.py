#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import tarfile
import urllib.request
from pathlib import Path

import hashlib


INTERPROSCAN_REQUIRED_FILES = (
    "interproscan.sh",
    "interproscan-5.jar",
    "interproscan.properties",
)

INTERPROSCAN_REQUIRED_ENGINE_NONEMPTY_DIRS = (
    "bin",
    "lib",
)

INTERPROSCAN_REQUIRED_ENGINE_DIRS = (
    "work",
    "temp",
)

INTERPROSCAN_REQUIRED_NONEMPTY_DIRS = (
    "bin",
    "data",
    "lib",
)

INTERPROSCAN_REQUIRED_DIRS = (
    "work",
    "temp",
)

INTERPROSCAN_REQUIRED_DATASETS = (
    "cdd",
    "ncbifam",
    "panther",
    "pfam",
    "prosite",
    "smart",
)

HMM_SELF_BUILD_REQUIRED_FILES = (
    "self_build.hmm",
    "self_build.hmm.h3f",
    "self_build.hmm.h3i",
    "self_build.hmm.h3m",
    "self_build.hmm.h3p",
)

HMM_PK_REQUIRED_FILES = (
    "GA_table.txt",
    "PK_class_desc.txt",
    "Tfam_domain.hmm",
    "Tfam_domain.hmm.h3f",
    "Tfam_domain.hmm.h3i",
    "Tfam_domain.hmm.h3m",
    "Tfam_domain.hmm.h3p",
    "Plant_Pkinase_fam.hmm",
    "Plant_Pkinase_fam.hmm.h3f",
    "Plant_Pkinase_fam.hmm.h3i",
    "Plant_Pkinase_fam.hmm.h3m",
    "Plant_Pkinase_fam.hmm.h3p",
    "PlantsPHMM3_89.hmm",
    "PlantsPHMM3_89.hmm.h3f",
    "PlantsPHMM3_89.hmm.h3i",
    "PlantsPHMM3_89.hmm.h3m",
    "PlantsPHMM3_89.hmm.h3p",
    "Pkinase_sub_WNK1.hmm",
    "Pkinase_sub_WNK1.hmm.h3f",
    "Pkinase_sub_WNK1.hmm.h3i",
    "Pkinase_sub_WNK1.hmm.h3m",
    "Pkinase_sub_WNK1.hmm.h3p",
    "Pkinase_sub_MAK.hmm",
    "Pkinase_sub_MAK.hmm.h3f",
    "Pkinase_sub_MAK.hmm.h3i",
    "Pkinase_sub_MAK.hmm.h3m",
    "Pkinase_sub_MAK.hmm.h3p",
)


def _path_is_non_empty(path):
    path = Path(path)
    if not path.exists():
        return False

    if path.is_dir():
        try:
            next(path.iterdir())
        except StopIteration:
            return False
        except OSError:
            return False
        return True

    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def validate_interproscan_installation(interproscan_dir):
    """
    Validate that an InterProScan installation has the minimum structure required by iTAK3.

    Returns:
        list[str]: A list of integrity issues. Empty means the layout looks usable.
    """
    interproscan_dir = Path(interproscan_dir)
    issues = []

    if not interproscan_dir.exists():
        return [f"InterProScan directory does not exist: {interproscan_dir}"]
    if not interproscan_dir.is_dir():
        return [f"InterProScan path is not a directory: {interproscan_dir}"]

    for relative_name in INTERPROSCAN_REQUIRED_FILES:
        target = interproscan_dir / relative_name
        if not target.exists():
            issues.append(f"Missing required file: {target}")
            continue
        if not target.is_file():
            issues.append(f"Required file is not a regular file: {target}")
            continue
        if not _path_is_non_empty(target):
            issues.append(f"Required file is empty: {target}")

    for relative_name in INTERPROSCAN_REQUIRED_NONEMPTY_DIRS:
        target = interproscan_dir / relative_name
        if not target.exists():
            issues.append(f"Missing required directory: {target}")
            continue
        if not target.is_dir():
            issues.append(f"Required directory is not a directory: {target}")
            continue
        if not _path_is_non_empty(target):
            issues.append(f"Required directory is empty: {target}")

    for relative_name in INTERPROSCAN_REQUIRED_DIRS:
        target = interproscan_dir / relative_name
        if not target.exists():
            issues.append(f"Missing required directory: {target}")
            continue
        if not target.is_dir():
            issues.append(f"Required directory is not a directory: {target}")

    data_dir = interproscan_dir / "data"
    if data_dir.exists() and data_dir.is_dir() and _path_is_non_empty(data_dir):
        for dataset_name in INTERPROSCAN_REQUIRED_DATASETS:
            dataset_dir = data_dir / dataset_name
            if not dataset_dir.exists():
                issues.append(f"Missing InterProScan dataset directory: {dataset_dir}")
                continue
            if not dataset_dir.is_dir():
                issues.append(f"InterProScan dataset path is not a directory: {dataset_dir}")
                continue
            if not _path_is_non_empty(dataset_dir):
                issues.append(f"InterProScan dataset directory is empty: {dataset_dir}")
                continue

            has_non_empty_version_dir = False
            try:
                for child in dataset_dir.iterdir():
                    if child.is_dir() and _path_is_non_empty(child):
                        has_non_empty_version_dir = True
                        break
            except OSError:
                has_non_empty_version_dir = False

            if not has_non_empty_version_dir:
                issues.append(
                    f"InterProScan dataset directory has no non-empty version subdirectory: {dataset_dir}"
                )

    return issues


def validate_interproscan_engine_installation(interproscan_dir):
    """
    Validate the engine-side InterProScan layout without enforcing the bundled data layout.

    This is intended for future setups where the engine and the iTAK-managed data directory
    are configured separately.
    """
    interproscan_dir = Path(interproscan_dir)
    issues = []

    if not interproscan_dir.exists():
        return [f"InterProScan directory does not exist: {interproscan_dir}"]
    if not interproscan_dir.is_dir():
        return [f"InterProScan path is not a directory: {interproscan_dir}"]

    for relative_name in INTERPROSCAN_REQUIRED_FILES:
        target = interproscan_dir / relative_name
        if not target.exists():
            issues.append(f"Missing required file: {target}")
            continue
        if not target.is_file():
            issues.append(f"Required file is not a regular file: {target}")
            continue
        if not _path_is_non_empty(target):
            issues.append(f"Required file is empty: {target}")

    for relative_name in INTERPROSCAN_REQUIRED_ENGINE_NONEMPTY_DIRS:
        target = interproscan_dir / relative_name
        if not target.exists():
            issues.append(f"Missing required directory: {target}")
            continue
        if not target.is_dir():
            issues.append(f"Required directory is not a directory: {target}")
            continue
        if not _path_is_non_empty(target):
            issues.append(f"Required directory is empty: {target}")

    for relative_name in INTERPROSCAN_REQUIRED_ENGINE_DIRS:
        target = interproscan_dir / relative_name
        if not target.exists():
            issues.append(f"Missing required directory: {target}")
            continue
        if not target.is_dir():
            issues.append(f"Required directory is not a directory: {target}")

    return issues


def validate_interproscan_data_directory(data_dir):
    """
    Validate an iTAK-managed InterProScan data directory independently of the engine layout.
    """
    data_dir = Path(data_dir)
    issues = []

    if not data_dir.exists():
        return [f"InterProScan data directory does not exist: {data_dir}"]
    if not data_dir.is_dir():
        return [f"InterProScan data path is not a directory: {data_dir}"]
    if not _path_is_non_empty(data_dir):
        return [f"InterProScan data directory is empty: {data_dir}"]

    for dataset_name in INTERPROSCAN_REQUIRED_DATASETS:
        dataset_dir = data_dir / dataset_name
        if not dataset_dir.exists():
            issues.append(f"Missing InterProScan dataset directory: {dataset_dir}")
            continue
        if not dataset_dir.is_dir():
            issues.append(f"InterProScan dataset path is not a directory: {dataset_dir}")
            continue
        if not _path_is_non_empty(dataset_dir):
            issues.append(f"InterProScan dataset directory is empty: {dataset_dir}")
            continue

        has_non_empty_version_dir = False
        try:
            for child in dataset_dir.iterdir():
                if child.is_dir() and _path_is_non_empty(child):
                    has_non_empty_version_dir = True
                    break
        except OSError:
            has_non_empty_version_dir = False

        if not has_non_empty_version_dir:
            issues.append(
                f"InterProScan dataset directory has no non-empty version subdirectory: {dataset_dir}"
            )

    return issues


def _validate_required_files_in_directory(target_dir, required_files, label):
    target_dir = Path(target_dir)
    issues = []

    if not target_dir.exists():
        return [f"{label} directory does not exist: {target_dir}"]
    if not target_dir.is_dir():
        return [f"{label} path is not a directory: {target_dir}"]

    for relative_name in required_files:
        target = target_dir / relative_name
        if not target.exists():
            issues.append(f"Missing required {label} file: {target}")
            continue
        if not target.is_file():
            issues.append(f"Required {label} file is not a regular file: {target}")
            continue
        if not _path_is_non_empty(target):
            issues.append(f"Required {label} file is empty: {target}")

    return issues


def validate_hmm_self_build_directory(hmm_dir):
    return _validate_required_files_in_directory(hmm_dir, HMM_SELF_BUILD_REQUIRED_FILES, "self-build HMM")


def validate_hmm_pk_directory(hmm_dir):
    return _validate_required_files_in_directory(hmm_dir, HMM_PK_REQUIRED_FILES, "protein kinase HMM")

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_db_integrity(db_path, verbose=True):
    """
    Check if the db directory exists and contains necessary subdirectories.
    
    Args:
        db_path (Path): Path to the db directory.
        
    Returns:
        bool: True if db is complete, False otherwise.
    """
    db_path = Path(db_path)
    if not db_path.exists() or not db_path.is_dir():
        return False

    interproscan_dir = db_path / "interproscan"
    issues = validate_interproscan_installation(interproscan_dir)
    issues.extend(validate_hmm_pk_directory(db_path / "hmm_pk"))
    issues.extend(validate_hmm_self_build_directory(db_path / "hmm_self_build"))
    if issues:
        if verbose:
            for issue in issues:
                print(issue)
        return False

    return True

import subprocess

def download_file(url, output_path):
    """
    Download a file from a URL to a local path using wget -c.
    
    Args:
        url (str): URL to download from.
        output_path (Path): Local path to save the file.
    """
    print(f"Downloading from {url}...")
    try:
        # Check if wget is installed
        if shutil.which("wget") is None:
            print("Error: wget is not installed.")
            return False
            
        cmd = ["wget", "-c", url, "-O", str(output_path)]
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            print("\nDownload complete.")
            return True
        else:
            print(f"\nwget exited with code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading file: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

def extract_tar_gz(tar_path, extract_path):
    """
    Extract a tar.gz file.
    
    Args:
        tar_path (Path): Path to the tar.gz file.
        extract_path (Path): Directory to extract to.
    """
    print(f"Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False

def setup_db(project_root):
    """
    Main function to ensure db directory is ready.
    
    Args:
        project_root (Path): Root directory of the project.
        
    Returns:
        bool: True if db is ready, False otherwise.
    """
    project_root = Path(project_root)
    db_path = project_root / "db"
    tar_path = project_root / "db.tar.gz"
    
    # URL for db.tar.gz (Replace with actual URL if known, using a placeholder for now based on repo)
    # Assuming the file is released in the repo or hosted somewhere accessible.
    # Since I don't have the exact URL, I'll use a likely one or a placeholder.
    # User said "from githup download", so likely:
    # Keep the current release asset URL until a renamed iTAK3 release asset is published.
    DB_DOWNLOAD_URL = "https://github.com/Enver-Kastrioti/itak2_9.19/releases/download/iTAK2/db.tar.gz"
    EXPECTED_SHA256 = "220599853264a378abf6527b90004af35104848bf7240497d6901e45a8a47fbc"
    
    # 1. Check integrity
    if check_db_integrity(db_path):
        print("DB directory check passed.")
        return True
    
    print("DB directory missing or incomplete.")
    
    for broken_dir in (
        db_path / "interproscan",
        db_path / "hmm_pk",
        db_path / "hmm_self_build",
    ):
        if broken_dir.exists():
            print(f"Removing incomplete database directory: {broken_dir}")
            try:
                shutil.rmtree(broken_dir)
            except Exception as e:
                print(f"Failed to remove incomplete database directory: {e}")
                return False

    # 2. Check if tar.gz exists
    if not tar_path.exists():
        print(f"db.tar.gz not found at {tar_path}.")
        print("Attempting to download from GitHub...")
        if not download_file(DB_DOWNLOAD_URL, tar_path):
            print("Failed to download database.")
            return False
    else:
        print(f"Found db.tar.gz at {tar_path}.")
        
    # Verify SHA256 of the tarball (whether newly downloaded or existing)
    print("Verifying file integrity (SHA256)...")
    file_hash = calculate_sha256(tar_path)
    if file_hash != EXPECTED_SHA256:
        print(f"Error: SHA256 mismatch!")
        print(f"Expected: {EXPECTED_SHA256}")
        print(f"Actual:   {file_hash}")
        print("The file might be corrupted. Deleting it and aborting.")
        try:
            tar_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete corrupted file: {e}")
        return False
        
    print("SHA256 verification passed.")
        
    # 3. Extract
    if not extract_tar_gz(tar_path, project_root):
        return False
        
    # 4. Verify again
    if not check_db_integrity(db_path):
        print("DB integrity check failed after extraction.")
        return False
        
    # 5. Cleanup
    try:
        tar_path.unlink()
        print(f"Deleted {tar_path}.")
    except Exception as e:
        print(f"Warning: Could not delete {tar_path}: {e}")
        
    return True
