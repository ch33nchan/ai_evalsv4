import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
AGENT_DEVICE = "cuda:1"
# DINOv3 is powerful, so we can train for fewer steps to see results
STEPS = 300000 

from argus_env import ArgusEnv

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__(verbose=0)
        self.pbar = tqdm(total=total_timesteps, desc="Training DINOv3 Agent", unit="step")
    def _on_step(self) -> bool:
        self.pbar.update(1)
        if len(self.model.ep_info_buffer) > 0:
            ep_infos = list(self.model.ep_info_buffer)
            recent_eps = ep_infos[-10:]
            if len(recent_eps) > 0:
                avg = sum([ep['r'] for ep in recent_eps]) / len(recent_eps)
                self.pbar.set_postfix({'avg_reward': f'{avg:.2f}'})
        return True

def main():
    print(">>> Initializing DINOv3 Environment...")
    env = Monitor(ArgusEnv(dataset_split='train'))
    
    print(f">>> Starting Fresh PPO Training on {AGENT_DEVICE}...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device=AGENT_DEVICE, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01
    )
    
    model.learn(total_timesteps=STEPS, callback=TqdmCallback(STEPS))
    model.save("argus_final_model")
    print(">>> Training Complete. Model saved to argus_final_model.zip")

if __name__ == "__main__":
    main()