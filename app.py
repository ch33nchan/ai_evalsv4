import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from stable_baselines3 import PPO
from argus_env import ArgusEnv

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 
DEVICE_AGENT = "cuda:0"
MODEL_PATH = "argus_final_model"

print(f">>> Loading Agent...")
dummy_env = ArgusEnv(dataset_split='train')

try:
    model = PPO.load(MODEL_PATH, device=DEVICE_AGENT)
    print(" Model loaded.")
except:
    print(" Model error. Ensure argus_final_model.zip exists.")
    exit()

# --- Signal Processing Forensics ---

def compute_spectral_anomaly(image_pil):
    """
    Angle 1: Frequency Domain (Grid Artifacts)
    """
    img = np.array(image_pil.convert("L"))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = magnitude.shape
    center_x, center_y = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
    
    tbin = np.bincount(r.ravel(), magnitude.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    
    # Check high-frequency tail for anomalies
    tail = radial_profile[int(len(radial_profile)*0.15):]
    if len(tail) < 2: return 0.0
    
    variance = np.std(np.log(tail + 1e-8))
    # Normalization heuristic
    anomaly_score = min(100, variance * 65) 
    return anomaly_score

def compute_texture_entropy(image_pil):
    """
    Angle 2: Texture Analysis (LBP Entropy)
    Detects "AI Smoothing" or "Digital Smudging"
    """
    img = np.array(image_pil.convert("L"))
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    ent = entropy(hist + 1e-8, base=2)
    
    # Inverse mapping: Real photos have high entropy (>5.0). AI often <4.0.
    # We map lower entropy to higher "Slop Score"
    slop_likelihood = max(0, (6.0 - ent) * 25)
    return min(100, slop_likelihood)

def generate_overlay(image_pil, history, verdict):
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = np.zeros_like(img, dtype=np.float32)

    if verdict == "AI GENERATED": color = [0, 0, 255]      # Red
    elif verdict == "SUS": color = [0, 255, 255]    # Yellow
    else: color = [0, 255, 0]                              # Green

    for i, box in enumerate(history):
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        intensity = 0.2 + (i / len(history)) * 0.5
        for c in range(3):
            if color[c] > 0: mask[y1:y2, x1:x2, c] += (intensity * 255)

    mask = np.clip(mask, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.addWeighted(img, 0.7, mask, 0.6, 0), cv2.COLOR_BGR2RGB)

def analyze_image(image_input):
    if image_input is None: return None, {}, "N/A"

    # 1. Signal Forensics (Hard Metrics)
    spectral_score = compute_spectral_anomaly(image_input)
    texture_score = compute_texture_entropy(image_input)
    
    # 2. Cognitive Agent (Behavioral Metric)
    dummy_env.current_image = image_input.convert("RGB")
    dummy_env.window = [0.5, 0.5, 1.0]
    dummy_env.current_step = 0
    dummy_env.history_buffer = np.zeros(20)
    dummy_env.recent_actions = []
    obs = dummy_env._get_observation()
    
    history = []
    max_prob_fake = 0.0
    action_log = []
    max_steps = 20
    steps_taken = max_steps

    for step in range(max_steps):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs).unsqueeze(0).to(DEVICE_AGENT)
            dist = model.policy.get_distribution(t_obs)
            probs = dist.distribution.probs.cpu().numpy()[0]
            action = np.argmax(probs)
            p_fake = float(probs[6])
            
            if p_fake > max_prob_fake: max_prob_fake = p_fake
        
        # Movement Logic
        w, h = dummy_env.current_image.size
        cx, cy = dummy_env.window[0]*w, dummy_env.window[1]*h
        cw, ch = w*dummy_env.window[2], h*dummy_env.window[2]
        history.append([cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2])

        step_size = 0.1 * dummy_env.window[2]
        if action == 0: dummy_env.window[1] = max(0, dummy_env.window[1] - step_size)
        elif action == 1: dummy_env.window[1] = min(1, dummy_env.window[1] + step_size)
        elif action == 2: dummy_env.window[0] = max(0, dummy_env.window[0] - step_size)
        elif action == 3: dummy_env.window[0] = min(1, dummy_env.window[0] + step_size)
        elif action == 4: dummy_env.window[2] = max(0.1, dummy_env.window[2] * 0.5)
        elif action == 5: dummy_env.window[2] = min(1.0, dummy_env.window[2] * 2.0)
        elif action == 6: 
            if p_fake > 0.60:
                steps_taken = step + 1
                break
        
        action_log.append(f"{step+1}: {['Up','Down','Left','Right','In','Out','Accuse'][action]}")
        obs = dummy_env._get_observation()

    agent_score = max_prob_fake * 100

    # --- 3. SMART FUSION LOGIC (The Fix) ---
    
    # Base average
    slop_score = (agent_score * 0.4) + (spectral_score * 0.3) + (texture_score * 0.3)
    
    # RED FLAG OVERRIDE: If any single metric is extremely high, boost the score.
    # This prevents a confused agent (0%) from hiding a blatant texture artifact (75%).
    max_signal = max(agent_score, spectral_score, texture_score)
    
    if max_signal > 70.0:
        # Boost score to at least the 'Suspicious' range (50+)
        slop_score = max(slop_score, (max_signal * 0.8))
        override_reason = "High Confidence Signal Detected"
    else:
        override_reason = "None"

    # Final Verdict
    if slop_score > 65: verdict = "AI GENERATED"
    elif slop_score > 45: verdict = "SUS"
    else: verdict = "AUTHENTIC"

    heatmap = generate_overlay(image_input, history, verdict)
    
    report = {
        "Final_Analysis": {
            "AI_Slop_Score": f"{slop_score:.1f} / 100",
            "Verdict": verdict,
            "Override_Trigger": override_reason
        },
        "Metric_Breakdown": {
            "Cognitive_Agent_Confidence": f"{agent_score:.1f}% (Visual Structure)",
            "Spectral_Anomaly_Score": f"{spectral_score:.1f}% (Frequency Artifacts)",
            "Texture_Coherence_Score": f"{texture_score:.1f}% (Digital Smoothing)"
        },
        "Investigation_Stats": {
            "Steps_Taken": steps_taken,
            "Agent_Focus": "Micro" if dummy_env.window[2] < 0.3 else "Global"
        }
    }

    return heatmap, report, f"{verdict} ({slop_score:.1f})"

# --- UI ---
with gr.Blocks(title="AI Image Analysis", theme=gr.themes.Base()) as demo:
    gr.Markdown("### AI Image Analysis")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Gaze Heatmap")
            
    out_json = gr.JSON(label=" Breakdown")
    out_txt = gr.Textbox(label="Summary", scale=0)

    btn.click(analyze_image, inp, [out_img, out_json, out_txt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7777,share=True)