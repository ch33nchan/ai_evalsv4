import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from sklearn.metrics import classification_report, confusion_matrix
from argus_env import ArgusEnv

# Config
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
AGENT_DEVICE = "cuda:0" # Load onto GPU 1 for inference
MODEL_PATH = "argus_final_model"

def evaluate():
    print(f">>> Loading {MODEL_PATH}...")
    # Use the environment to load the balanced dataset
    env = ArgusEnv(dataset_split='train') 
    model = PPO.load(MODEL_PATH, device=AGENT_DEVICE)

    y_true = []
    y_pred = []
    
    # Test on 200 random samples (approx 100 Real / 100 Fake)
    n_episodes = 200
    print(f">>> Running Evaluation on {n_episodes} episodes...")
    
    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        verdict = 0 # Default Real (0)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
            if action == 6: # Accuse
                verdict = 1 # Fake
            
            done = terminated or truncated

        y_true.append(env.current_label)
        y_pred.append(verdict)

    # Report
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))
    
    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_true, y_pred)
    print(f"True Negatives (Real as Real): {cm[0][0]}")
    print(f"False Positives (Real as Fake): {cm[0][1]}")
    print(f"False Negatives (Fake as Real): {cm[1][0]}")
    print(f"True Positives (Fake as Fake): {cm[1][1]}")

if __name__ == "__main__":
    evaluate()