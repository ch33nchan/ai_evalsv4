import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

# --- 1. Hardware Setup ---
# Restrict Python to only see Physical GPUs 1 and 2.
# They will be re-indexed as:
# Physical GPU 1 -> Logical "cuda:0" (Used by Environment/DINOv2)
# Physical GPU 2 -> Logical "cuda:1" (Used by PPO Agent)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 

# Define the device for the Policy Network
AGENT_DEVICE = "cuda:1"

# Import local environment (must be in same directory)
from argus_env import ArgusEnv

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__(verbose=0)
        self.pbar = tqdm(total=total_timesteps, desc="Argus Training", unit="step")
        
    def _on_step(self) -> bool:
        self.pbar.update(1)
        
        # Retrieve episode infos
        if len(self.model.ep_info_buffer) > 0:
            # FIX: Convert deque to list before slicing
            ep_infos = list(self.model.ep_info_buffer)
            recent_eps = ep_infos[-10:]
            
            # Calculate moving average reward
            if len(recent_eps) > 0:
                avg_reward = sum([ep['r'] for ep in recent_eps]) / len(recent_eps)
                self.pbar.set_postfix({'avg_reward': f'{avg_reward:.2f}'})
        
        return True

    def _on_training_end(self):
        self.pbar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "debug"], help="Select execution mode")
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    args = parser.parse_args()

    # Hardware verification
    print(f"DEBUG: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"DEBUG: PPO Agent target device = {AGENT_DEVICE}")

    # Configuration based on mode
    if args.mode == "debug":
        print(">>> MODE: DEBUG (Short run, small batches)")
        total_steps = 2000
        n_steps_per_update = 128 
    else:
        print(f">>> MODE: TRAIN (Full run: {args.steps} steps)")
        total_steps = args.steps
        n_steps_per_update = 2048 

    # 1. Initialize Environment
    # This automatically loads DINOv2 onto 'cuda:0' (Physical GPU 1) defined in argus_env.py
    print(">>> Initializing Environment...")
    env = Monitor(ArgusEnv(dataset_split='train'))

    # 2. Initialize Agent
    # We explicitly map the policy network to 'cuda:1' (Physical GPU 2)
    print(f">>> Initializing PPO Agent on {AGENT_DEVICE}...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        n_steps=n_steps_per_update, 
        device=AGENT_DEVICE, 
        tensorboard_log="./argus_logs/"
    )

    # 3. Start Training
    print(">>> Starting Training Loop...")
    model.learn(total_timesteps=total_steps, callback=TqdmCallback(total_steps))
    
    # 4. Save Artifacts
    save_name = "argus_final_model"
    model.save(save_name)
    print(f">>> Training Complete. Model saved to {save_name}.zip")

if __name__ == "__main__":
    main()