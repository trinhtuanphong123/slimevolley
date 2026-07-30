"""
Simple evaluation example.

run: python eval_ppo.py --render

Evaluate a PPO policy (MLP) against the built-in baseline AI.
Requires a model trained by training_scripts/train_ppo.py (stable-baselines3).
"""

import os
import gymnasium as gym
import numpy as np
import argparse

import slimevolleygym
from stable_baselines3 import PPO


def rollout(env, policy, render_mode=False):
  """ play the trained agent against the built-in baseline policy """
  obs, _ = env.reset()
  terminated = False
  truncated = False
  total_reward = 0

  while not (terminated or truncated):
    action, _states = policy.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if render_mode:
      env.render()

  return total_reward


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Evaluate trained PPO agent.')
  parser.add_argument('--model-path', help='path to stable-baselines3 model.',
                      type=str, default="zoo/ppo/best_model.zip")
  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)

  args = parser.parse_args()
  render_mode = args.render

  assert os.path.exists(args.model_path), \
    args.model_path + " doesn't exist. Train one first: python training_scripts/train_ppo.py"

  env = gym.make("SlimeVolley-v0", render_mode="human" if render_mode else None)

  print("Loading", args.model_path)
  policy = PPO.load(args.model_path, env=env)

  history = []
  for i in range(1000):
    env.reset(seed=i)
    cumulative_score = rollout(env, policy, render_mode)
    print("cumulative score #", i, ":", cumulative_score)
    history.append(cumulative_score)

  print("history dump:", history)
  print("average score", np.mean(history), "standard_deviation", np.std(history))
