"""Run: python train.py --steps 100000"""
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from nav_env import NavigationEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=100000)
    p.add_argument('--envs',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    Path('models').mkdir(exist_ok=True)
    env=make_vec_env(lambda:Monitor(NavigationEnv()),n_envs=args.envs,seed=args.seed)
    model=PPO('MlpPolicy',env,verbose=1,device='cpu',seed=args.seed,
              n_steps=512,batch_size=128,learning_rate=3e-4,tensorboard_log=None)
    model.learn(total_timesteps=args.steps,progress_bar=False)
    model.save('models/moorebot_ppo')
    print('Saved models/moorebot_ppo.zip')
    env.close()

if __name__=='__main__':main()
