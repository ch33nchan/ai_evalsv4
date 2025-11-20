import os
import json
import requests
import zipfile
import argparse
from tqdm import tqdm

# Configuration
KEY_DIR = "/mnt/data1/srini/ai_evalsv4"
DATA_ROOT = os.path.expanduser("/mnt/data1/srini/ai_evalsv4/argus_data")

DATASETS = {
    "artifact": "awsaf49/artifact-dataset",
    "div2k": "soumikrakshit/div2k-high-resolution-images"
}

def get_kaggle_credentials(key_path):
    """Parses Kaggle credentials from the json file."""
    try:
        with open(key_path, 'r') as f:
            data = json.load(f)
        return data['username'], data['key']
    except FileNotFoundError:
        raise FileNotFoundError(f"Kaggle key not found at {key_path}")
    except KeyError:
        raise ValueError("Invalid kaggle.json format. Expected 'username' and 'key' fields.")

def download_with_progress(url, dest_path, auth):
    """Streams download with a tqdm progress bar."""
    response = requests.get(url, stream=True, auth=auth)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB chunks

    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def unzip_with_progress(zip_path, extract_to):
    """Unzips file with file-level progress tracking."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.infolist()
        with tqdm(total=len(members), desc=f"Extracting to {extract_to}", unit="file") as pbar:
            for member in members:
                zf.extract(member, extract_to)
                pbar.update(1)

def main():
    key_path = os.path.join(KEY_DIR, "kaggle.json")
    
    # 1. Authenticate
    try:
        username, key = get_kaggle_credentials(key_path)
        auth = (username, key)
        print(f"[INFO] Authenticated as {username}")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return

    # 2. Ensure directories
    os.makedirs(DATA_ROOT, exist_ok=True)

    # 3. Process Datasets
    for folder_name, dataset_id in DATASETS.items():
        target_dir = os.path.join(DATA_ROOT, folder_name)
        zip_name = f"{folder_name}.zip"
        zip_path = os.path.join(DATA_ROOT, zip_name)
        
        # Check if extracted folder exists and is populated
        if os.path.exists(target_dir) and os.listdir(target_dir):
            print(f"[INFO] {folder_name} dataset already exists. Skipping.")
            continue

        # Construct Kaggle API URL
        # Format: https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset-slug}
        url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_id}"
        
        print(f"[INFO] Processing {dataset_id}...")
        
        # Step A: Download
        try:
            download_with_progress(url, zip_path, auth)
        except Exception as e:
            print(f"[ERROR] Failed to download {dataset_id}: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue

        # Step B: Extract
        try:
            os.makedirs(target_dir, exist_ok=True)
            unzip_with_progress(zip_path, target_dir)
        except zipfile.BadZipFile:
            print(f"[ERROR] Downloaded file for {dataset_id} is corrupt.")
        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")
        finally:
            # Cleanup zip to save space
            if os.path.exists(zip_path):
                print(f"[INFO] Cleaning up {zip_name}...")
                os.remove(zip_path)

    print("[INFO] Data fetch pipeline complete.")

if __name__ == "__main__":
    main()