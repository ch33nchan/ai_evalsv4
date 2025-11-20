import os
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 
from argus_env import ArgusEnv

dummy_env = ArgusEnv(dataset_split='train')
MODEL_PATH = "argus_final_model"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, env=dummy_env, device=device)
else:
    model = PPO("MlpPolicy", dummy_env, device=device)

def analyze(img):
    if img is None: return None, "0", "N/A"
    dummy_env.current_image = img.convert("RGB")
    dummy_env.window = [0.5, 0.5, 1.0]
    dummy_env.current_step = 0
    obs = dummy_env._get_observation()
    
    history = []
    score = 0
    verdict = "REAL"
    
    for step in range(20):
        action, _ = model.predict(obs, deterministic=True)
        w, h = dummy_env.current_image.size
        cx, cy = dummy_env.window[0]*w, dummy_env.window[1]*h
        cw, ch = w*dummy_env.window[2], h*dummy_env.window[2]
        history.append([cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2])
        
        obs, _, _, _, _ = dummy_env.step(action)
        if action == 6:
            verdict = "FAKE"
            score = (1.0 - step/20)*100
            break
            
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    for box in history: draw.rectangle(box, outline="red", width=3)
    return vis, f"{score:.1f}", verdict

demo = gr.Interface(fn=analyze, inputs=gr.Image(type="pil"), outputs=[gr.Image(), "text", "text"])
if __name__ == "__main__": demo.launch(server_name="0.0.0.0", server_port=7860)
