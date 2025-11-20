import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
AGENT_DEVICE = "cuda:1"
STEPS = 100000 # Shorter run to correct behavior

from argus_env import ArgusEnv

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__(verbose=0)
        self.pbar = tqdm(total=total_timesteps, desc="Corrective Finetuning", unit="step")
    def _on_step(self) -> bool:
        self.pbar.update(1)
        if len(self.model.ep_info_buffer) > 0:
            ep_infos = list(self.model.ep_info_buffer)
            recent_eps = ep_infos[-10:]
            if len(recent_eps) > 0:
                avg_reward = sum([ep['r'] for ep in recent_eps]) / len(recent_eps)
                self.pbar.set_postfix({'avg_reward': f'{avg_reward:.2f}'})
        return True

def main():
    print(f">>> Loading Agent for CORRECTION...")
    env = Monitor(ArgusEnv(dataset_split='train'))
    
    try:
        model = PPO.load("argus_final_model", env=env, device=AGENT_DEVICE)
    except:
        model = PPO("MlpPolicy", env, verbose=1, device=AGENT_DEVICE)

    # Force Exploration to break "Always Fake" bias
    model.ent_coef = 0.05 
    model.learning_rate = 0.0002
    
    print(f">>> Starting Correction Run (Penalty enabled)...")
    model.learn(total_timesteps=STEPS, callback=TqdmCallback(STEPS))
    
    model.save("argus_final_model")
    print(">>> Correction Complete.")

if __name__ == "__main__":
    main()
