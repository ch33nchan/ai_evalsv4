import os
import numpy as np
from PIL import Image

base_dir = os.path.expanduser("/argus_data")
paths = {
    "real": os.path.join(base_dir, "div2k", "DIV2K_train_HR"),
    "fake": os.path.join(base_dir, "artifact", "fake")
}

def create_dummy():
    print(f">>> Creating dummy data in {base_dir}...")
    for label, path in paths.items():
        os.makedirs(path, exist_ok=True)
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img.save(os.path.join(path, f"{label}_dummy_{i}.png"))
    print(">>> Dummy data ready.")

if __name__ == "__main__":
    create_dummy()
