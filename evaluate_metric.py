import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from argus_env import ArgusEnv

# Hardware Config
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
MODEL_PATH = "argus_final_model"

def evaluate():
    print(f">>> Loading {MODEL_PATH}...")
    # Use 'test' split if you implemented it, otherwise 'train' for sanity check
    # Ideally, you modify ArgusEnv to accept a 'test' folder in _load_dataset
    env = ArgusEnv(dataset_split='train') 
    model = PPO.load(MODEL_PATH)

    n_episodes = 100
    correct = 0
    false_positives = 0
    false_negatives = 0
    detected_fakes = 0
    
    print(f">>> Running Evaluation on {n_episodes} random episodes...")
    
    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        verdict = "REAL" # Default if timeout
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if action == 6: # Accuse
                verdict = "FAKE"
            
            done = terminated or truncated

        # Ground Truth Check
        is_actually_fake = (env.current_label == 1)
        voted_fake = (verdict == "FAKE")

        if voted_fake == is_actually_fake:
            correct += 1
        elif voted_fake and not is_actually_fake:
            false_positives += 1
        elif not voted_fake and is_actually_fake:
            false_negatives += 1

    # Metrics
    accuracy = (correct / n_episodes) * 100
    print(f"\n=== RESULTS ({n_episodes} samples) ===")
    print(f" Accuracy: {accuracy:.2f}%")
    print(f"False Positives (Called Real 'Fake'): {false_positives}")
    print(f" False Negatives (Missed the Fake):   {false_negatives}")

if __name__ == "__main__":
    evaluate()