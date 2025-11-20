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
# Define where your data lives (based on the structure we discussed)
DATA_ROOT = os.path.expanduser("/argus_data")
DEVICE_ENV = torch.device("cuda:0")  # GPU for DINOv2 & Patches

class ArgusEnv(gym.Env):
    """
    The Active Vision Forensic Environment.
    The agent controls a 'camera' scanning a high-res image to find AI artifacts.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, dataset_split='train', max_steps=20):
        super(ArgusEnv, self).__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # --- 1. Action Space (Discrete) ---
        # 0: Pan Up, 1: Pan Down, 2: Pan Left, 3: Pan Right
        # 4: Zoom In (2x), 5: Zoom Out (0.5x)
        # 6: Accuse (Terminate & Predict)
        self.action_space = spaces.Discrete(7)

        # --- 2. Observation Space (Embedding) ---
        # DINOv2 Small outputs a vector of size 384.
        # We stack: [Current View Embedding (384) + History Vector (10)]
        self.obs_dim = 384 + 10 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # --- 3. Load Models (The "Eye") ---
        print(f"Loading DINOv2 on {DEVICE_ENV}...")
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE_ENV)
        self.dino.eval()
        
        # Pre-processing for DINO
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # DINO expects 224 patches
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- 4. Load Data Paths ---
        self.image_paths = self._load_dataset(dataset_split)
        print(f"ArgusEnv initialized with {len(self.image_paths)} images.")

        # State variables
        self.current_image = None # The full PIL image
        self.current_label = None # 0 for Real, 1 for Fake
        self.window = [0, 0, 1.0] # [x_center (0-1), y_center (0-1), zoom_level]
        self.history_buffer = np.zeros(10)

    def _load_dataset(self, split):
        """
        Scans the directory structure for images.
        Adapts to the folders we created earlier (artifact, div2k, ffhq).
        """
        files = []
        
        # 1. Add Real Images (Label 0)
        real_dirs = [
            os.path.join(DATA_ROOT, "div2k", "DIV2K_train_HR"),
            os.path.join(DATA_ROOT, "ffhq", "images")
        ]
        for d in real_dirs:
            if os.path.exists(d):
                found = glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))
                files.extend([(f, 0) for f in found]) # Tuple: (path, label)

        # 2. Add Fake Images (Label 1)
        # Note: Modify this path once your Artifact/Flux download finishes
        fake_dirs = [
            os.path.join(DATA_ROOT, "artifact", "fake"), 
            os.path.join(DATA_ROOT, "my_flux_gen")
        ]
        for d in fake_dirs:
            if os.path.exists(d):
                found = glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))
                files.extend([(f, 1) for f in found])

        random.shuffle(files)
        return files

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a random image
        path, label = random.choice(self.image_paths)
        self.current_label = label
        
        # Load Image (Keep in RAM, send crop to GPU later)
        self.current_image = Image.open(path).convert("RGB")
        
        # Reset View (Start at center, fully zoomed out)
        self.window = [0.5, 0.5, 1.0] 
        self.current_step = 0
        self.history_buffer = np.zeros(10)
        
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        reward = -0.1 # Time penalty (encourage speed)

        # --- Execute Action ---
        # Move Step Size depends on zoom (finer moves when zoomed in)
        step_size = 0.1 * self.window[2] 
        
        if action == 0: # Up
            self.window[1] = max(0, self.window[1] - step_size)
        elif action == 1: # Down
            self.window[1] = min(1, self.window[1] + step_size)
        elif action == 2: # Left
            self.window[0] = max(0, self.window[0] - step_size)
        elif action == 3: # Right
            self.window[0] = min(1, self.window[0] + step_size)
        elif action == 4: # Zoom In
            self.window[2] = max(0.1, self.window[2] * 0.5) # Smaller window = Zoom In
        elif action == 5: # Zoom Out
            self.window[2] = min(1.0, self.window[2] * 2.0)
        elif action == 6: # Accuse!
            terminated = True
            # The Critical Reward Logic
            # If Agent accuses 'Fake' (action 6) and it IS Fake (label 1) -> Big Reward
            # But wait... the agent needs to say "Real" or "Fake".
            # For simplicity: Action 6 means "I think this is FAKE".
            # If the agent runs out of time without Action 6, it implicitly votes "REAL".
            
            if self.current_label == 1:
                reward = 10.0 # Caught a fake!
                print(">>> Agent CAUGHT a Fake!")
            else:
                reward = -10.0 # False accusation of a Real image
                print(">>> Agent False Positive (Accused Real)")

        # Check max steps
        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated and self.current_label == 0:
                # Correctly identified Real by not accusing it
                reward = 5.0 

        # Get new observation
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        """
        1. Crop the image based on self.window
        2. Send crop to GPU
        3. Run DINOv2
        """
        # Calculate pixel coordinates
        w, h = self.current_image.size
        cx, cy = self.window[0] * w, self.window[1] * h
        
        # Window size (zoom level 1.0 = full image)
        crop_w = w * self.window[2]
        crop_h = h * self.window[2]
        
        # Box (left, top, right, bottom)
        left = max(0, cx - crop_w/2)
        top = max(0, cy - crop_h/2)
        right = min(w, cx + crop_w/2)
        bottom = min(h, cy + crop_h/2)
        
        crop = self.current_image.crop((left, top, right, bottom))
        
        # Feature Extraction (On GPU 0)
        with torch.no_grad():
            t_crop = self.transform(crop).unsqueeze(0).to(DEVICE_ENV)
            embedding = self.dino(t_crop).cpu().numpy()[0] # Move back to CPU for SB3
            
        # Combine with history (simple placeholder for now)
        full_obs = np.concatenate([embedding, self.history_buffer])
        return full_obs