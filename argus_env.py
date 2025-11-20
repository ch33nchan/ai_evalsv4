import os
import glob
import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from torchvision import transforms
from PIL import Image

# --- Configuration ---
# CRITICAL FIX: Pointing to the actual data location on the mount
DATA_ROOT = "/mnt/data1/srini/ai_evalsv4/argus_data"

# Device mapping: Environment uses Logical cuda:0 (Physical GPU 1)
DEVICE_ENV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ArgusEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, dataset_split='train', max_steps=20):
        super(ArgusEnv, self).__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = spaces.Discrete(7)
        self.obs_dim = 384 + 10 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        print(f"Loading DINOv2 on {DEVICE_ENV}...")
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE_ENV)
        self.dino.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_paths = self._load_dataset()
        print(f"ArgusEnv initialized with {len(self.image_paths)} images.")

        self.current_image = None
        self.current_label = None
        self.window = [0, 0, 1.0]
        self.history_buffer = np.zeros(10)

    def _load_dataset(self):
        files = []
        
        # --- 1. REAL IMAGES ---
        # Path structure based on your ls output
        real_sources = [
            os.path.join(DATA_ROOT, "div2k", "DIV2K_train_HR"),
            os.path.join(DATA_ROOT, "artifact", "ffhq"),
            os.path.join(DATA_ROOT, "artifact", "coco"),
            os.path.join(DATA_ROOT, "artifact", "imagenet")
        ]
        
        print(f">>> Scanning REAL folders in {DATA_ROOT}...")
        for d in real_sources:
            if not os.path.exists(d):
                print(f"   [WARNING] Folder not found: {d}")
                continue
                
            # Scan for images (png, jpg, jpeg)
            found = glob.glob(os.path.join(d, "**", "*.png"), recursive=True) + \
                    glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True) + \
                    glob.glob(os.path.join(d, "**", "*.jpeg"), recursive=True)
            
            found = found[:5000] 
            files.extend([(f, 0) for f in found])
            print(f"   Found {len(found)} in {os.path.basename(d)}")

        # --- 2. FAKE IMAGES ---
        fake_sources = [
            os.path.join(DATA_ROOT, "artifact", "stable_diffusion"),
            os.path.join(DATA_ROOT, "artifact", "stylegan3"),
            os.path.join(DATA_ROOT, "artifact", "latent_diffusion"),
            os.path.join(DATA_ROOT, "artifact", "glide")
        ]

        print(">>> Scanning FAKE folders...")
        for d in fake_sources:
            if not os.path.exists(d): 
                print(f"   [WARNING] Folder not found: {d}")
                continue
                
            found = glob.glob(os.path.join(d, "**", "*.png"), recursive=True) + \
                    glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True) + \
                    glob.glob(os.path.join(d, "**", "*.jpeg"), recursive=True)
            
            found = found[:5000]
            files.extend([(f, 1) for f in found])
            print(f"   Found {len(found)} in {os.path.basename(d)}")

        if len(files) == 0:
            print("❌ CRITICAL ERROR: Still no images found. Check if the drive is mounted.")
            # Create a dummy image in memory to allow debugging instead of crashing
            return [("dummy", 0)]

        random.shuffle(files)
        return files

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Safety check for dummy/empty
        if not self.image_paths or self.image_paths[0][0] == "dummy":
             return np.zeros(self.obs_dim, dtype=np.float32), {}

        # Try loading an image. If corrupt, skip to next.
        for _ in range(10): # Try 10 times
            path, label = random.choice(self.image_paths)
            self.current_label = label
            try:
                self.current_image = Image.open(path).convert("RGB")
                break # Success
            except Exception:
                continue
        
        # If still None (super unlucky), fail gracefully
        if self.current_image is None:
             return np.zeros(self.obs_dim, dtype=np.float32), {}

        self.window = [0.5, 0.5, 1.0] 
        self.current_step = 0
        self.history_buffer = np.zeros(10)
        
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        reward = -0.1 

        step_size = 0.1 * self.window[2] 
        if action == 0: self.window[1] = max(0, self.window[1] - step_size)
        elif action == 1: self.window[1] = min(1, self.window[1] + step_size)
        elif action == 2: self.window[0] = max(0, self.window[0] - step_size)
        elif action == 3: self.window[0] = min(1, self.window[0] + step_size)
        elif action == 4: self.window[2] = max(0.1, self.window[2] * 0.5)
        elif action == 5: self.window[2] = min(1.0, self.window[2] * 2.0)
        elif action == 6: 
            terminated = True
            if self.current_label == 1: reward = 10.0 
            else: reward = -10.0 

        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated and self.current_label == 0: reward = 5.0 

        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        if self.current_image is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        w, h = self.current_image.size
        cx, cy = self.window[0] * w, self.window[1] * h
        cw, ch = w * self.window[2], h * self.window[2]
        
        crop = self.current_image.crop((
            max(0, cx - cw/2), max(0, cy - ch/2),
            min(w, cx + cw/2), min(h, cy + ch/2)
        ))
        
        with torch.no_grad():
            t_crop = self.transform(crop).unsqueeze(0).to(DEVICE_ENV)
            embedding = self.dino(t_crop).cpu().numpy()[0]
            
        return np.concatenate([embedding, self.history_buffer])
