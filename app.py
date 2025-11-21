import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from stable_baselines3 import PPO
from argus_env import ArgusEnv

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 
DEVICE_AGENT = "cuda:0"
MODEL_PATH = "argus_final_model"

print(f">>> Loading ARGUS Agent from {MODEL_PATH}...")
dummy_env = ArgusEnv(dataset_split='train')

try:
    model = PPO.load(MODEL_PATH, device=DEVICE_AGENT)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def calculate_ainess_breakdown(verdict, prob_fake, steps, max_steps, min_zoom):
    """
    Decomposes the score into measurable components.
    """
    # 1. Raw Confidence (0-100)
    score_conf = prob_fake * 100

    # 2. Speed/Obviousness (0-100)
    score_urgency = (1.0 - (steps / max_steps)) * 100
    
    # 3. Scale Classification
    artifact_scale = "Global Structure"
    if min_zoom < 0.6: artifact_scale = "Regional Feature"
    if min_zoom < 0.3: artifact_scale = "Micro-Texture"

    # 4. Weighted Score
    if verdict == "FAKE":
        final_score = (score_conf * 0.6) + (score_urgency * 0.4)
    elif verdict == "SUSPICIOUS":
        final_score = 45.0 + (prob_fake * 10)
    else:
        final_score = (1.0 - prob_fake) * 15 
        
    # STRUCTURED REPORT (Matches keys used in analyze_image)
    return {
        "AI_Ness_Evaluation": {
            "Score": f"{final_score:.1f}",
            "Verdict": verdict,
            "Risk_Level": "High" if final_score > 80 else ("Medium" if final_score > 40 else "Low")
        },
        "Metrics_Breakdown": {
            "Model_Confidence": f"{score_conf:.1f}%",
            "Detection_Speed": f"{score_urgency:.1f}% ({steps} steps)",
            "Artifact_Scale": artifact_scale,
            "Min_Zoom_Level": f"{min_zoom:.2f}x"
        }
    }

def generate_overlay(image_pil, history, verdict):
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = np.zeros_like(img, dtype=np.float32)

    # BGR Colors
    if verdict == "FAKE":
        color = [0, 0, 255]   # Red
    elif verdict == "SUSPICIOUS":
        color = [0, 255, 255] # Yellow
    else:
        color = [0, 255, 0]   # Green

    for i, box in enumerate(history):
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = img.shape
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        intensity = 0.2 + (i / len(history)) * 0.5
        
        for c in range(3):
            if color[c] > 0:
                mask[y1:y2, x1:x2, c] += (intensity * 255)

    mask = np.clip(mask, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.addWeighted(img, 0.7, mask, 0.6, 0), cv2.COLOR_BGR2RGB)

def analyze_image(image_input):
    if image_input is None: return None, {}, "N/A"

    # 1. Reset Env Logic
    dummy_env.current_image = image_input.convert("RGB")
    dummy_env.window = [0.5, 0.5, 1.0]
    dummy_env.current_step = 0
    
    # Reset History Buffer
    dummy_env.history_buffer = np.zeros(20) 
    dummy_env.recent_actions = [] 
    
    obs = dummy_env._get_observation()
    
    history_boxes = []
    verdict = "REAL"
    
    # Tracking
    max_prob_fake = 0.0
    min_zoom = 1.0
    action_log = []

    max_steps = 20
    steps_taken = max_steps

    # 2. Investigation Loop
    for step in range(max_steps):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs).unsqueeze(0).to(DEVICE_AGENT)
            dist = model.policy.get_distribution(t_obs)
            probs = dist.distribution.probs.cpu().numpy()[0]
            action = np.argmax(probs)
            
            p_fake = float(probs[6])
            if p_fake > max_prob_fake: max_prob_fake = p_fake
        
        # Metadata update
        current_zoom = dummy_env.window[2]
        if current_zoom < min_zoom: min_zoom = current_zoom
        
        w, h = dummy_env.current_image.size
        cx, cy = dummy_env.window[0]*w, dummy_env.window[1]*h
        cw, ch = w*dummy_env.window[2], h*dummy_env.window[2]
        history_boxes.append([cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2])
        
        # Update History Buffer
        current_state = [dummy_env.window[0], dummy_env.window[1], dummy_env.window[2], float(action)/7.0]
        dummy_env.recent_actions.append(current_state)
        if len(dummy_env.recent_actions) > 5: dummy_env.recent_actions.pop(0)
        
        flat_hist = np.array(dummy_env.recent_actions).flatten()
        if len(flat_hist) < 20:
            padding = np.zeros(20 - len(flat_hist))
            flat_hist = np.concatenate([padding, flat_hist])
        dummy_env.history_buffer = flat_hist

        # Apply Movement
        step_size = 0.1 * dummy_env.window[2]
        if action == 0: dummy_env.window[1] = max(0, dummy_env.window[1] - step_size)
        elif action == 1: dummy_env.window[1] = min(1, dummy_env.window[1] + step_size)
        elif action == 2: dummy_env.window[0] = max(0, dummy_env.window[0] - step_size)
        elif action == 3: dummy_env.window[0] = min(1, dummy_env.window[0] + step_size)
        elif action == 4: dummy_env.window[2] = max(0.1, dummy_env.window[2] * 0.5)
        elif action == 5: dummy_env.window[2] = min(1.0, dummy_env.window[2] * 2.0)
        elif action == 6: 
            if p_fake > 0.60:
                verdict = "FAKE"
                steps_taken = step + 1
                break
        
        action_str = ['Up','Down','Left','Right','In','Out','Accuse'][action]
        action_log.append(f"{step+1}: {action_str} ({p_fake*100:.0f}%)")
        
        obs = dummy_env._get_observation()

    # Post-Process
    if verdict == "REAL":
        if max_prob_fake > 0.50: 
            verdict = "SUSPICIOUS"

    # 3. Reports
    heatmap = generate_overlay(image_input, history_boxes, verdict)
    report = calculate_ainess_breakdown(verdict, max_prob_fake, steps_taken, max_steps, min_zoom)
    report["Action_Log_Tail"] = action_log[-5:]
    
    # Correct key access matching the structure above
    summary = f"{verdict} ({report['AI_Ness_Evaluation']['Score']}/100)"

    return heatmap, report, summary

# --- UI ---
css = ".json-holder {height: 100%;}"

with gr.Blocks(title="ARGUS v6", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("### ARGUS: DINOv3 Forensic Analysis")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Target Image")
            btn = gr.Button("Run Analysis", variant="primary")
            
        with gr.Column(scale=2):
            output_img = gr.Image(label="Active Vision Heatmap")
            
    with gr.Row():
        out_json = gr.JSON(label="Forensic Scorecard")
        out_txt = gr.Textbox(label="Verdict Summary", scale=0)

    btn.click(analyze_image, input_img, [output_img, out_json, out_txt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7575,share=True)