import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from argus_env import ArgusEnv

# --- Configuration ---
# Hardware: Visible Devices 1 and 2. 
# Inside script: cuda:0 = Phyiscal GPU 1, cuda:1 = Physical GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
DEVICE_AGENT = "cuda:0" # Run inference on the same GPU as DINOv2 for speed
MODEL_PATH = "argus_final_model"
CSV_DIR = "/mnt/data1/srini/ai_evalsv4/argus_data/csv_downloaded"

def generate_overlay(image_pil, history, verdict):
    """
    Generates a heatmap overlay.
    - FAKE verdict = RED path
    - REAL verdict = GREEN path
    """
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = np.zeros_like(img, dtype=np.float32)
    
    # Color Logic (BGR)
    if verdict == "FAKE":
        color_base = [0, 0, 255] # Red
    else:
        color_base = [0, 255, 0] # Green

    for i, box in enumerate(history):
        x1, y1, x2, y2 = map(int, box)
        
        # Clamp coordinates
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Intensity increases with time (Agent focusing)
        intensity = 0.2 + (i / len(history)) * 0.5
        
        for c in range(3):
            if color_base[c] > 0: 
                mask[y1:y2, x1:x2, c] += (intensity * 255)

    mask = np.clip(mask, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.addWeighted(img, 0.7, mask, 0.6, 0), cv2.COLOR_BGR2RGB)

def run():
    # 1. Pick Random CSV Image
    print(f">>> Scanning {CSV_DIR}...")
    if not os.path.exists(CSV_DIR):
        print(f"❌ Directory not found: {CSV_DIR}")
        return

    all_files = [
        os.path.join(CSV_DIR, f) 
        for f in os.listdir(CSV_DIR) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    if not all_files:
        print("❌ No images found in CSV directory.")
        return
    
    target_img_path = random.choice(all_files)
    print(f">>> Selected Target: {os.path.basename(target_img_path)}")
    
    # 2. Setup Agent
    # We use a dummy env just to load the DINOv2 encoder and structure
    print(">>> Loading Agent...")
    env = ArgusEnv(dataset_split='train')
    
    try:
        model = PPO.load(MODEL_PATH, device=DEVICE_AGENT)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # 3. Reset Environment with Target Image
    env.reset()
    try:
        env.current_image = Image.open(target_img_path).convert("RGB")
    except Exception as e:
        print(f"❌ Corrupt image selected: {e}")
        return

    env.window = [0.5, 0.5, 1.0] # Start Center, Zoom 1x
    env.current_step = 0
    env.history_buffer = np.zeros(10)
    
    obs = env._get_observation()
    history = []
    verdict = "REAL"
    confidence = 0.0
    
    # 4. Run Investigation Loop (Max 20 steps)
    max_steps = 20
    print(">>> Agent Investigating...")
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        
        # Record current view (Before moving)
        w, h = env.current_image.size
        cx, cy = env.window[0]*w, env.window[1]*h
        cw, ch = w*env.window[2], h*env.window[2]
        history.append([cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2])
        
        # Step the logic manually (since we don't use env.step() fully here)
        step_size = 0.1 * env.window[2]
        if action == 0: env.window[1] = max(0, env.window[1] - step_size)
        elif action == 1: env.window[1] = min(1, env.window[1] + step_size)
        elif action == 2: env.window[0] = max(0, env.window[0] - step_size)
        elif action == 3: env.window[0] = min(1, env.window[0] + step_size)
        elif action == 4: env.window[2] = max(0.1, env.window[2] * 0.5)
        elif action == 5: env.window[2] = min(1.0, env.window[2] * 2.0)
        elif action == 6: 
            verdict = "FAKE"
            confidence = (1.0 - (step / max_steps)) * 100
            break
            
        obs = env._get_observation()

    if verdict == "REAL":
        confidence = 100.0 # Agent is confident it found nothing

    # 5. Save Visualization
    print(f">>> Verdict: {verdict} ({confidence:.1f}%)")
    
    heatmap = generate_overlay(env.current_image, history, verdict)
    
    # Create side-by-side plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(env.current_image)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap)
    plt.title(f"Agent Verdict: {verdict}")
    plt.axis('off')
    
    save_name = "csv_test_result.png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"✅ Result saved to {save_name}")

if __name__ == "__main__":
    run()