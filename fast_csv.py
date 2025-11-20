import os
import asyncio
import aiohttp
import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm
from urllib.parse import urlparse

# --- Configuration ---
CSV_PATH = "/mnt/data1/srini/ai_evalsv4/csv\'s/query_result_2025-11-14T13_52_48.293337128Z.csv"
OUTPUT_DIR = "/mnt/data1/srini/ai_evalsv4/argus_data/csv_downloaded"
CONCURRENT_LIMIT = 1000  # Safe limit. Increase to 500 if you have fiber internet.
TIMEOUT_SEC = 10
# ---------------------

async def download_image(session, url, idx, semaphore):
    async with semaphore:  # Limits active connections
        try:
            # Parse extension safely
            parsed = urlparse(url)
            path = parsed.path
            ext = os.path.splitext(path)[1]
            if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                ext = '.jpg'
            
            filename = f"csv_fake_{idx}{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            # Skip if exists
            if os.path.exists(filepath):
                return False

            async with session.get(url, timeout=TIMEOUT_SEC) as response:
                if response.status == 200:
                    content = await response.read()
                    async with aiofiles.open(filepath, 'wb') as f:
                        await f.write(content)
                    return True
        except Exception:
            return False
        return False

async def main():
    print(f">>> Reading CSV: {CSV_PATH}")
    # Efficiently read only the URL column
    try:
        df = pd.read_csv(CSV_PATH, usecols=['Image URL'])
    except ValueError:
        # Fallback if no header or different name
        df = pd.read_csv(CSV_PATH, usecols=[0], names=['Image URL'], header=0)
        
    urls = df["Image URL"].dropna().tolist()
    total_urls = len(urls)
    print(f"✅ Loaded {total_urls} URLs.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Semaphore controls the flood
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    # Connection Pooling
    connector = aiohttp.TCPConnector(limit=None, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, url in enumerate(urls):
            task = download_image(session, url, i, semaphore)
            tasks.append(task)
        
        print(f">>> BLASTING downloads with {CONCURRENT_LIMIT} concurrent connections...")
        
        # Execute with progress bar
        results = await tqdm.gather(*tasks, unit="img", desc="Downloading")

    success_count = sum(results)
    print(f"\n>>> COMPLETE. Downloaded {success_count}/{total_urls} images.")

if __name__ == "__main__":
    # Use uvloop if available (linux only) for extra speed
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
        
    asyncio.run(main())