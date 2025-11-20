import os
import glob
import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from torchvision import transforms
from PIL import Image

DATA_ROOT = "/mnt/data1/srini/ai_evalsv4/argus_data"
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
        print(f"ArgusEnv initialized with {len(self.image_paths)} balanced images.")

        self.current_image = None
        self.current_label = None
        self.window = [0, 0, 1.0]
        self.history_buffer = np.zeros(10)

    def _load_dataset(self):
        # Same loader logic as before - keep it robust
        fake_files = []
        fake_sources = [
            os.path.join(DATA_ROOT, "csv_downloaded"),
            os.path.join(DATA_ROOT, "artifact", "stable_diffusion"),
            os.path.join(DATA_ROOT, "artifact", "midjourney"),
            os.path.join(DATA_ROOT, "artifact", "stylegan3")
        ]
        for d in fake_sources:
            if os.path.exists(d):
                found = glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True) + \
                        glob.glob(os.path.join(d, "**", "*.png"), recursive=True)
                fake_files.extend([(f, 1) for f in found])

        real_files = []
        real_sources = [
            os.path.join(DATA_ROOT, "open_images_real"),
            os.path.join(DATA_ROOT, "div2k", "DIV2K_train_HR"),
            os.path.join(DATA_ROOT, "artifact", "ffhq"),
            os.path.join(DATA_ROOT, "artifact", "coco")
        ]
        for d in real_sources:
            if os.path.exists(d):
                found = glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True) + \
                        glob.glob(os.path.join(d, "**", "*.png"), recursive=True)
                real_files.extend([(f, 0) for f in found])

        if not fake_files or not real_files: return [("dummy", 0)]

        # Balance
        limit = min(len(real_files), len(fake_files))
        print(f">>> Balancing to {limit} per class...")
        random.shuffle(real_files)
        random.shuffle(fake_files)
        data = real_files[:limit] + fake_files[:limit]
        random.shuffle(data)
        return data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.image_paths or self.image_paths[0][0] == "dummy": return np.zeros(self.obs_dim, dtype=np.float32), {}

        for _ in range(10):
            path, label = random.choice(self.image_paths)
            self.current_label = label
            try:
                self.current_image = Image.open(path).convert("RGB")
                if self.current_image.size[0] < 32: continue
                break
            except: continue
        
        if self.current_image is None: return np.zeros(self.obs_dim, dtype=np.float32), {}

        self.window = [0.5, 0.5, 1.0] 
        self.current_step = 0
        self.history_buffer = np.zeros(10)
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        reward = -0.5 # Moderate step cost

        step_size = 0.1 * self.window[2] 
        if action == 0: self.window[1] = max(0, self.window[1] - step_size)
        elif action == 1: self.window[1] = min(1, self.window[1] + step_size)
        elif action == 2: self.window[0] = max(0, self.window[0] - step_size)
        elif action == 3: self.window[0] = min(1, self.window[0] + step_size)
        elif action == 4: self.window[2] = max(0.1, self.window[2] * 0.5)
        elif action == 5: self.window[2] = min(1.0, self.window[2] * 2.0)
        elif action == 6: # ACCUSE
            terminated = True
            if self.current_label == 1:
                reward = 100.0 # Valid Catch
            else:
                reward = -100.0 # SEVERE PENALTY for guessing

        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated and self.current_label == 0:
                reward = 20.0 # Reward for correctly NOT accusing (Real)

        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        if self.current_image is None: return np.zeros(self.obs_dim, dtype=np.float32)
        w, h = self.current_image.size
        cx, cy = self.window[0] * w, self.window[1] * h
        cw, ch = w * self.window[2], h * self.window[2]
        crop = self.current_image.crop((max(0, cx - cw/2), max(0, cy - ch/2), min(w, cx + cw/2), min(h, cy + ch/2)))
        with torch.no_grad():
            t_crop = self.transform(crop).unsqueeze(0).to(DEVICE_ENV)
            embedding = self.dino(t_crop).cpu().numpy()[0]
        return np.concatenate([embedding, self.history_buffer])
