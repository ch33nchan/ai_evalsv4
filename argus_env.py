import os
import glob
import random
import numpy as np
import torch
import timm
import gymnasium as gym
from gymnasium import spaces
from torchvision import transforms
from PIL import Image

# --- Configuration ---
DATA_ROOT = "/mnt/data1/srini/ai_evalsv4/argus_data"
# DINOv3 runs on Logical cuda:0 (Physical GPU 1)
DEVICE_ENV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ArgusEnv(gym.Env):
    """
    The Active Vision Forensic Environment (DINOv3 Edition).
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, dataset_split='train', max_steps=20):
        super(ArgusEnv, self).__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = spaces.Discrete(7)
        
        # Observation: 
        # [Embedding (1024)] -> ViT-Large DINOv3 is 1024 dim
        # [History (20)]     -> Last 5 steps context
        self.embed_dim = 1024 
        self.history_dim = 20
        self.obs_dim = self.embed_dim + self.history_dim
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        print(f"Loading DINOv3 (ViT-Large LVD-1689M) on {DEVICE_ENV}...")
        # UPGRADE: DINOv3 Backbone via timm
        self.dino = timm.create_model(
            'vit_large_patch16_dinov3.lvd1689m', 
            pretrained=True, 
            num_classes=0, # Return global pool/CLS token
        ).to(DEVICE_ENV)
        self.dino.eval()
        
        # DINOv3 uses standard ImageNet stats
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
        self.history_buffer = np.zeros(self.history_dim)
        self.recent_actions = []

    def _load_dataset(self):
        # (Robust loader logic)
        fake_files = []
        fake_sources = [
            os.path.join(DATA_ROOT, "csv_downloaded"),
            os.path.join(DATA_ROOT, "artifact", "stable_diffusion"),
            os.path.join(DATA_ROOT, "artifact", "stylegan3"),
            os.path.join(DATA_ROOT, "artifact", "midjourney"),
            os.path.join(DATA_ROOT, "artifact", "glide")
        ]
        for d in fake_sources:
            if os.path.exists(d):
                found = glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True) + \
                        glob.glob(os.path.join(d, "**", "*.png"), recursive=True) + \
                        glob.glob(os.path.join(d, "**", "*.jpeg"), recursive=True)
                fake_files.extend([(f, 1) for f in found])
        
        real_files = []
        real_sources = [
            os.path.join(DATA_ROOT, "open_images_real"),
            os.path.join(DATA_ROOT, "div2k", "DIV2K_train_HR"),
            os.path.join(DATA_ROOT, "artifact", "ffhq"),
            os.path.join(DATA_ROOT, "artifact", "coco"),
            os.path.join(DATA_ROOT, "artifact", "imagenet")
        ]
        for d in real_sources:
            if os.path.exists(d):
                found = glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True) + \
                        glob.glob(os.path.join(d, "**", "*.png"), recursive=True) + \
                        glob.glob(os.path.join(d, "**", "*.jpeg"), recursive=True)
                real_files.extend([(f, 0) for f in found])

        if not fake_files: return [("dummy", 0)]

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
        self.history_buffer = np.zeros(self.history_dim)
        self.recent_actions = []
        
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        reward = -1.0 

        # History Update
        current_state = [self.window[0], self.window[1], self.window[2], float(action)/7.0]
        self.recent_actions.append(current_state)
        if len(self.recent_actions) > 5: self.recent_actions.pop(0)
        flat_hist = np.array(self.recent_actions).flatten()
        if len(flat_hist) < self.history_dim:
            padding = np.zeros(self.history_dim - len(flat_hist))
            flat_hist = np.concatenate([padding, flat_hist])
        self.history_buffer = flat_hist

        step_size = 0.1 * self.window[2] 
        if action == 0: self.window[1] = max(0, self.window[1] - step_size)
        elif action == 1: self.window[1] = min(1, self.window[1] + step_size)
        elif action == 2: self.window[0] = max(0, self.window[0] - step_size)
        elif action == 3: self.window[0] = min(1, self.window[0] + step_size)
        elif action == 4: self.window[2] = max(0.1, self.window[2] * 0.5)
        elif action == 5: self.window[2] = min(1.0, self.window[2] * 2.0)
        elif action == 6: 
            terminated = True
            if self.current_label == 1:
                reward = 200.0 # Bounty (DINOv3 is smarter, we don't need +500 to force it)
            else:
                reward = -100.0 # Penalty

        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated and self.current_label == 0:
                reward = 20.0 # Reward for patience (Real)

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