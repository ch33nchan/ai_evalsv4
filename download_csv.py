import os
import pandas as pd
import requests
import concurrent.futures
from tqdm import tqdm
from urllib.parse import urlparse

# --- Configuration ---
# CHANGE THIS to your actual CSV path
CSV_PATH = "/mnt/data1/srini/ai_evalsv4/csv\'s/query_result_2025-11-14T13_52_48.293337128Z.csv"
OUTPUT_DIR = "/mnt/data1/srini/ai_evalsv4/argus_data/csv_downloaded"
MAX_WORKERS = 16 # Number of parallel downloads (adjust based on CPU/Net)
TIMEOUT = 5

def download_single_image(args):
    url, idx = args
    try:
        # Generate safe filename
        ext = os.path.splitext(urlparse(url).path)[1]
        if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            ext = '.jpg' # Default
            
        save_path = os.path.join(OUTPUT_DIR, f"csv_fake_{idx}{ext}")
        
        if os.path.exists(save_path):
            return 0 # Skip existing

        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return 1
    except Exception:
        return 0
    return 0

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV not found at {CSV_PATH}")
        return

    print(f">>> Reading CSV {CSV_PATH}...")
    # Read only the image_url column to save RAM
    try:
        df = pd.read_csv(CSV_PATH, usecols=['Image URL'])
    except ValueError:
        # Fallback if column name is different, read first column
        print("Warning: 'image_url' column not found. Using first column.")
        df = pd.read_csv(CSV_PATH, usecols=[0], names=['Image URL'], header=0)

    urls = df["Image URL"].dropna().tolist()
    print(f"✅ Found {len(urls)} URLs.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f">>> Starting Download with {MAX_WORKERS} workers...")
    
    # Prepare args for map (url, index)
    download_args = [(url, i) for i, url in enumerate(urls)]
    
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(download_single_image, download_args), total=len(urls), unit="img"))
        success_count = sum(results)

    print(f"\n>>> Download Complete. Successfully saved {success_count} images to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()