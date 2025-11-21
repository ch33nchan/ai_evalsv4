import os

# Configuration
DATA_ROOT = "/mnt/data1/srini/ai_evalsv4/argus_data"

SOURCES = {
    "REAL": [
        "open_images_real",
        "div2k/DIV2K_train_HR",
        "artifact/ffhq",
        "artifact/coco",
        "artifact/imagenet"
    ],
    "FAKE": [
        "csv_downloaded",
        "artifact/stable_diffusion",
        "artifact/stylegan3",
        "artifact/midjourney",
        "artifact/glide",
        "artifact/latent_diffusion"
    ]
}

def count_files_fast(dir_path):
    """Counts images recursively using os.scandir for speed."""
    count = 0
    if not os.path.exists(dir_path):
        return 0
        
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                count += 1
    return count

def main():
    print(f"{'CATEGORY':<20} | {'SOURCE FOLDER':<30} | {'COUNT':>10}")
    print("-" * 66)

    total_real = 0
    total_fake = 0

    # Count Real
    for src in SOURCES["REAL"]:
        path = os.path.join(DATA_ROOT, src)
        n = count_files_fast(path)
        total_real += n
        print(f"{'REAL':<20} | {src:<30} | {n:>10,}")

    print("-" * 66)

    # Count Fake
    for src in SOURCES["FAKE"]:
        path = os.path.join(DATA_ROOT, src)
        n = count_files_fast(path)
        total_fake += n
        print(f"{'FAKE':<20} | {src:<30} | {n:>10,}")

    print("=" * 66)
    print(f"TOTAL REAL: {total_real:,}")
    print(f"TOTAL FAKE: {total_fake:,}")
    
    balance_ratio = total_real / (total_fake + 1e-6)
    print(f"BALANCE RATIO (Real/Fake): {balance_ratio:.2f} (Ideal is 1.0)")

if __name__ == "__main__":
    main()