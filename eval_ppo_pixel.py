#!/usr/bin/env python3

# Evaluate a PPO CNN agent (stable-baselines3) on the pixel version of the task.
# Visualizes the 84x84x4 preprocessed frames the agent actually sees.
#
# Requires a model trained by training_scripts/train_ppo_pixel.py.

import os
import numpy as np
import argparse
import gymnasium as gym
import slimevolleygym
from slimevolleygym import render_atari, FrameStack
from slimevolleygym._rendering import SimpleImageViewer
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from time import sleep

RENDER_ATARI = True  # render the downsampled 84x84x4 grayscale inputs


def make_env(seed):
    env = gym.make("SlimeVolleyNoFrameskip-v0")
    env = AtariWrapper(env, clip_reward=False)
    env = FrameStack(env, 4)
    env.reset(seed=seed)
    return env


def rollout(env, model, viewer):
    obs = env.reset()
    cumulative_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        if viewer is not None:
            viewer.imshow(render_atari(obs))
            sleep(0.08)
    return cumulative_reward


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate trained PPO CNN agent.')
    parser.add_argument('--model-path', help='path to stable-baselines3 model.',
                        type=str, default="ppo_cnn/best_model.zip")
    parser.add_argument('--seed', help='random seed (integer)', type=int, default=721)
    args = parser.parse_args()

    assert os.path.exists(args.model_path), \
        args.model_path + " doesn't exist. Train one first: python training_scripts/train_ppo_pixel.py"

    env = make_env(args.seed)
    model = PPO.load(args.model_path)

    viewer = SimpleImageViewer(maxwidth=2160) if RENDER_ATARI else None

    rewards = []
    for i in range(1000):
        cumulative_reward = rollout(env, model, viewer)
        print(i, cumulative_reward)
        rewards.append(cumulative_reward)

    print("mean", np.mean(rewards))
    print("stdev", np.std(rewards))

    env.close()
    if viewer is not None:
        viewer.close()
