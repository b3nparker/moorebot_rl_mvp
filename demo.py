"""Visualize an untrained or trained policy; optional video recording via matplotlib is not required."""
import argparse
import numpy as np
from pathlib import Path
from nav_env import NavigationEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--random',action='store_true',help='Run random policy as environment sanity test')
    p.add_argument('--render',action='store_true',help='Show live matplotlib visualization')
    p.add_argument('--seed',type=int,default=1)
    args=p.parse_args()
    model=None
    if not args.random:
        from stable_baselines3 import PPO
        if not Path('models/moorebot_ppo.zip').exists():
            raise SystemExit('No trained policy. Run: python train.py --steps 100000 (or use --random).')
        model=PPO.load('models/moorebot_ppo')
    env=NavigationEnv()
    successes=0; collisions=0
    for episode in range(args.episodes):
        obs,_=env.reset(seed=args.seed+episode)
        done=False; steps=0
        while not done:
            if model is None:
                action=env.action_space.sample()
            else:
                action,_=model.predict(obs,deterministic=True)
            obs,reward,terminated,truncated,info=env.step(action)
            steps+=1
            if args.render: env.render()
            done=terminated or truncated
        successes+=info['success']; collisions+=info['collision']
        print(f'Episode {episode+1}: success={info["success"]}, collision={info["collision"]}, steps={steps}, final_distance={info["distance"]:.2f} m')
    print(f'Success rate: {successes}/{args.episodes}; collisions: {collisions}/{args.episodes}')
    if args.render:
        input('Press Enter to close visualization...')
    env.close()

if __name__=='__main__':main()
