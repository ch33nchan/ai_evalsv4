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
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def generate_forensic_report(verdict, max_prob_fake, steps, max_steps, min_zoom):
    """
    Generates a detailed metric evaluation (0-100 Scale) instead of just a verdict.
    """
    # 1. Base Score Calculation
    # If Fake: Score starts high, boosted by speed
    # If Real: Score is based on 'Peak Suspicion' (max_prob_fake)
    if verdict == "FAKE":
        base_score = max_prob_fake * 100
        speed_bonus = (1.0 - (steps / max_steps)) * 20 # Up to +20 pts for speed
        final_score = min(100.0, base_score + speed_bonus)
        risk_level = "CRITICAL"
    elif verdict == "SUSPICIOUS":
        final_score = 45.0 + (max_prob_fake * 10) # Range 45-55
        risk_level = "MODERATE"
    else:
        # Even if Real, how suspicious was it?
        # Scale 0-30 based on max probability seen
        final_score = max_prob_fake * 40 
        risk_level = "LOW"

    # 2. Artifact Classification
    if min_zoom < 0.3:
        scan_type = "Micro-Texture Analysis"
    elif min_zoom < 0.7:
        scan_type = "Regional Feature Analysis"
    else:
        scan_type = "Global Structure Analysis"

    # 3. Structured JSON Output
    return {
        "AI_Ness_Evaluation": {
            "Score": round(final_score, 2),
            "Risk_Level": risk_level,
            "Verdict_Label": verdict
        },
        "Confidence_Metrics": {
            "Peak_Suspicion": f"{max_prob_fake*100:.2f}%",
            "Decision_Step": f"{steps}/{max_steps}",
            "Investigation_Efficiency": f"{(1.0 - steps/max_steps)*100:.1f}%"
        },
        "Behavioral_Forensics": {
            "Scan_Depth": f"{min_zoom:.2f}x",
            "Focus_Strategy": scan_type
        }
    }

def generate_overlay(image_pil, history, verdict):
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = np.zeros_like(img, dtype=np.float32)

    # Heatmap Colors (BGR)
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
        
        # Time-decay intensity (Newer = Brighter)
        intensity = 0.1 + (i / len(history)) * 0.6
        
        for c in range(3):
            if color[c] > 0:
                mask[y1:y2, x1:x2, c] += (intensity * 255)

    mask = np.clip(mask, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.addWeighted(img, 0.7, mask, 0.6, 0), cv2.COLOR_BGR2RGB)

def analyze_image(image_input):
    if image_input is None: return None, {}, "N/A"

    # 1. Setup Environment
    dummy_env.current_image = image_input.convert("RGB")
    dummy_env.window = [0.5, 0.5, 1.0]
    dummy_env.current_step = 0
    dummy_env.history_buffer = np.zeros(10)
    obs = dummy_env._get_observation()
    
    history_boxes = []
    verdict = "REAL"
    
    # Tracking Variables
    max_prob_fake = 0.0
    min_zoom = 1.0
    action_log = []

    max_steps = 20
    steps_taken = max_steps

    # 2. Agent Loop
    for step in range(max_steps):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs).unsqueeze(0).to(DEVICE_AGENT)
            dist = model.policy.get_distribution(t_obs)
            probs = dist.distribution.probs.cpu().numpy()[0]
            action = np.argmax(probs)
            
            # Track suspicion even if not accusing
            p_fake = float(probs[6])
            if p_fake > max_prob_fake:
                max_prob_fake = p_fake
        
        # Update Metadata
        if dummy_env.window[2] < min_zoom: min_zoom = dummy_env.window[2]
        
        # Record Path
        w, h = dummy_env.current_image.size
        cx, cy = dummy_env.window[0]*w, dummy_env.window[1]*h
        cw, ch = w*dummy_env.window[2], h*dummy_env.window[2]
        history_boxes.append([cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2])
        
        # Step Logic
        step_size = 0.1 * dummy_env.window[2]
        if action == 0: dummy_env.window[1] = max(0, dummy_env.window[1] - step_size)
        elif action == 1: dummy_env.window[1] = min(1, dummy_env.window[1] + step_size)
        elif action == 2: dummy_env.window[0] = max(0, dummy_env.window[0] - step_size)
        elif action == 3: dummy_env.window[0] = min(1, dummy_env.window[0] + step_size)
        elif action == 4: dummy_env.window[2] = max(0.1, dummy_env.window[2] * 0.5)
        elif action == 5: dummy_env.window[2] = min(1.0, dummy_env.window[2] * 2.0)
        elif action == 6: 
            # Confidence Threshold
            if p_fake > 0.60:
                verdict = "FAKE"
                steps_taken = step + 1
                break
        
        action_str = ['Up','Down','Left','Right','In','Out','Accuse'][action]
        action_log.append(f"Step {step+1}: {action_str} (Suspicion: {p_fake*100:.1f}%)")
        
        obs = dummy_env._get_observation()

    # Post-Process: "Soft" Verdicts
    if verdict == "REAL":
        if max_prob_fake > 0.45: 
            verdict = "SUSPICIOUS"

    # 3. Generate Output
    report = generate_forensic_report(verdict, max_prob_fake, steps_taken, max_steps, min_zoom)
    report["Live_Action_Log"] = action_log[-5:] # Show last 5 moves
    
    heatmap = generate_overlay(image_input, history_boxes, verdict)
    
    # Simple summary string for the textbox
    summary_str = f"{report['AI_Ness_Evaluation']['Score']}/100 ({verdict})"

    return heatmap, report, summary_str

# --- UI Layout ---
with gr.Blocks(title="ARGUS Evaluation Suite", theme=gr.themes.Soft()) as demo:
    gr.Markdown("### 👁️ ARGUS: Active Vision AI Evaluation")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Input Image")
            btn = gr.Button("Generate Evaluation", variant="primary")
            
        with gr.Column(scale=1):
            output_img = gr.Image(label="Attention Heatmap")
            
    with gr.Row():
        # The JSON output is now the star of the show
        out_json = gr.JSON(label="Forensic Metrics Scorecard")
        
    # Hidden textbox for simple return value handling if needed
    out_txt = gr.Textbox(label="Quick Score", visible=True)

    btn.click(analyze_image, input_img, [output_img, out_json, out_txt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7500,share=True)