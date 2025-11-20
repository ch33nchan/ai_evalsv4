import os
import io
import requests
import pandas as pd
import asyncio
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm

# --- Configuration ---
OUTPUT_DIR = "/mnt/data1/srini/ai_evalsv4/argus_data/open_images_real"
TARGET_COUNT = 150000  # Aim for 150k to match your 170k Fakes
CONCURRENT_LIMIT = 300 # Adjust based on bandwidth
TIMEOUT_SEC = 15

# URL to the official Open Images V7 Image ID list (Train split)
INDEX_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
# S3 Bucket Base URL for Open Images
S3_BASE_URL = "https://s3.amazonaws.com/open-images-dataset/train"

async def download_s3_image(session, image_id, semaphore):
    url = f"{S3_BASE_URL}/{image_id}.jpg"
    save_path = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")
    
    if os.path.exists(save_path):
        return False

    async with semaphore:
        try:
            async with session.get(url, timeout=TIMEOUT_SEC) as response:
                if response.status == 200:
                    content = await response.read()
                    async with aiofiles.open(save_path, 'wb') as f:
                        await f.write(content)
                    return True
        except Exception:
            pass
        return False

async def main():
    print(f">>> Fetching Open Images Index from {INDEX_URL}...")
    
    # 1. Download Index in Memory
    response = requests.get(INDEX_URL)
    if response.status_code != 200:
        print("❌ Failed to retrieve image index.")
        return
        
    # 2. Parse CSV
    print(">>> Parsing CSV...")
    df = pd.read_csv(io.BytesIO(response.content))
    
    # 3. Sample Random Images
    if len(df) > TARGET_COUNT:
        print(f">>> Sampling {TARGET_COUNT} images from {len(df)} available...")
        df_sample = df.sample(n=TARGET_COUNT, random_state=42)
    else:
        df_sample = df
    
    image_ids = df_sample["ImageID"].tolist()
    print(f"✅ Prepared {len(image_ids)} Image IDs.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 4. Async Download
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    connector = aiohttp.TCPConnector(limit=None)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_s3_image(session, iid, semaphore) for iid in image_ids]
        
        print(f">>> Starting Download ({CONCURRENT_LIMIT} threads)...")
        results = await tqdm.gather(*tasks, unit="img", desc="Fetching Real Data")
        
    print(f"\n>>> Download Complete. Saved {sum(results)} images to {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
    asyncio.run(main())