import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from argus_env import ArgusEnv

# Config
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
MODEL_PATH = "argus_final_model"
DEVICE = "cuda:0"

def overlay_heatmap(image_pil, history):
    """
    Creates a heatmap overlay based on where the agent looked.
    history: list of [x1, y1, x2, y2] crop coordinates
    """
    # Convert PIL to numpy (OpenCV format)
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Create a blank mask
    mask = np.zeros_like(img, dtype=np.float32)
    
    # "Paint" the mask where the agent looked
    # Later steps get higher intensity
    for i, box in enumerate(history):
        x1, y1, x2, y2 = map(int, box)
        
        # Intensity increases with steps (Agent focuses harder)
        intensity = 0.3 + (i / len(history)) * 0.7
        
        # Add red color to the looked-at region
        # BGR format: Red is channel 2
        mask[y1:y2, x1:x2, 2] += (intensity * 255) 
    
    # Normalize mask to 0-255
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    
    # Blend original image with mask
    heatmap = cv2.addWeighted(img, 0.7, mask, 0.5, 0)
    
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

def run_investigation(image_path):
    env = ArgusEnv()
    model = PPO.load(MODEL_PATH)
    
    # Manually set the image
    env.reset()
    env.current_image = Image.open(image_path).convert("RGB")
    env.window = [0.5, 0.5, 1.0]
    env.current_step = 0
    env.history_buffer = np.zeros(10)
    
    obs = env._get_observation()
    history = []
    verdict = "REAL"
    confidence = 0
    
    print(f"🔍 Investigating {os.path.basename(image_path)}...")
    
    for step in range(30):
        action, _ = model.predict(obs, deterministic=True)
        
        # Record current view
        w, h = env.current_image.size
        cx, cy = env.window[0]*w, env.window[1]*h
        cw, ch = w*env.window[2], h*env.window[2]
        history.append([cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2])
        
        obs, reward, terminated, _, _ = env.step(action)
        
        if action == 6: # Accuse
            verdict = "FAKE"
            confidence = (1.0 - (step / 30)) * 100
            break
            
    # Generate Heatmap
    heatmap_img = overlay_heatmap(env.current_image, history)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image\nGround Truth: ?")
    plt.imshow(env.current_image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Agent Heatmap\nVerdict: {verdict} ({confidence:.1f}%)")
    plt.imshow(heatmap_img)
    plt.axis('off')
    
    save_path = f"heatmap_{os.path.basename(image_path)}"
    plt.savefig(save_path)
    print(f"✅ Saved analysis to {save_path}")

if __name__ == "__main__":
    # Pick a random test image from your dataset to test
    test_dir = "/mnt/data1/srini/ai_evalsv4/argus_data/artifact/stable_diffusion/stable-face/Male/"
    if os.path.exists(test_dir):
        random_img = os.path.join(test_dir, os.listdir(test_dir)[0])
        run_investigation(random_img)
    else:
        print("Please specify an image path in the script.")