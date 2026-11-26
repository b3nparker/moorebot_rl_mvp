"""Repeatable PPO vs random evaluation of the Moorebot 2D navigation simulator.
Run from the same directory as nav_env.py, with this file copied there.
"""
import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO
from nav_env import NavigationEnv


def evaluate(model, seeds, random=False):
    env = NavigationEnv()
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        if random:
            env.action_space.seed(seed + 100000)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action = env.action_space.sample() if random else model.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
        results.append({
            'policy': 'random' if random else 'ppo',
            'seed': seed,
            'success': int(info['success']),
            'collision': int(info['collision']),
            'timeout': int(truncated and not terminated),
            'steps': steps,
            'final_distance_m': round(info['distance'], 4),
        })
    env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/moorebot_ppo_best.zip')
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--output', default='evaluation_results.csv')
    args = p.parse_args()
    if args.episodes <= 0:
        p.error('--episodes must be positive')
    if not Path(args.model).is_file():
        p.error('model not found: ' + args.model)
    model = PPO.load(args.model, device='cpu')
    env = NavigationEnv()
    if model.observation_space.shape != env.observation_space.shape:
        p.error('model and environment observation sizes do not match')
    env.close()
    seeds = range(args.seed, args.seed + args.episodes)
    trained = evaluate(model, seeds)
    random = evaluate(None, seeds, random=True)
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trained[0]))
        w.writeheader()
        w.writerows(trained + random)
    for name, rows in [('PPO', trained), ('Random', random)]:
        n = len(rows)
        successes = sum(row['success'] for row in rows)
        collisions = sum(row['collision'] for row in rows)
        timeouts = sum(row['timeout'] for row in rows)
        successful_steps = [row['steps'] for row in rows if row['success']]
        avg_steps = f'{sum(successful_steps) / len(successful_steps):.1f}' if successful_steps else 'n/a'
        print(f'{name}: success={successes}/{n} ({successes/n:.1%}), '
              f'collision={collisions}/{n} ({collisions/n:.1%}), '
              f'timeout={timeouts}/{n} ({timeouts/n:.1%}), '
              f'avg successful steps={avg_steps}')
    print('Saved ' + args.output)
    print('NOTE: This uses the same fixed wall configuration as training; it does NOT test new layouts.')


if __name__ == '__main__':
    main()
