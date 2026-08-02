#!/usr/bin/env python3

# Trains a convnet PPO agent to play SlimeVolley from pixels
# (SlimeVolleySurvivalNoFrameskip-v0), using the Atari preprocessing wrappers
# bundled with stable-baselines3.
#
# run with:  python train_ppo_pixel.py   (set NUM_ENVS to your CPU core count)
#
# Requires stable-baselines3: pip install slimevolleygym[training]

import os
import gymnasium as gym
import slimevolleygym

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from slimevolleygym import FrameStack  # plain-numpy frame stack (no LazyFrames)

NUM_TIMESTEPS = int(2e9)
NUM_ENVS = 96           # number of parallel workers (CPU cores)
SEED = 101
EVAL_EPISODES = 100
LOGDIR = "ppo_cnn"     # moved to zoo afterwards.


def make_env(env_id, seed):
    # Explicit pre-processing wrappers (replacing AtariWrapper which squashed
    # the 84x168 SlimeVolley image to 84x84): frame skip 4, grayscale, resize,
    # then 4-frame stack. clip_reward is not needed (survival bonus is small).
    def _init():
        env = gym.make(env_id)
        env = MaxAndSkipEnv(env, skip=4)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        env = FrameStack(env, 4)
        env.reset(seed=seed)
        return env
    return _init


def train():
    os.makedirs(LOGDIR, exist_ok=True)
    logger = configure(LOGDIR, ["stdout", "csv"])

    env = VecMonitor(SubprocVecEnv(
        [make_env("SlimeVolleySurvivalNoFrameskip-v0", SEED + 10000 * i)
         for i in range(NUM_ENVS)]))
    eval_env = VecMonitor(SubprocVecEnv(
        [make_env("SlimeVolleyNoFrameskip-v0", SEED + 999000)]))

    # Atari-tuned hyperparameters (from the stable-baselines3 Atari example).
    model = PPO("CnnPolicy", env,
                n_steps=256, batch_size=64, n_epochs=4,
                learning_rate=2.5e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.1, ent_coef=0.01, verbose=2, seed=SEED)

    eval_freq = max(1, 100000 // NUM_ENVS)
    eval_callback = EvalCallback(eval_env, best_model_save_path=LOGDIR,
                                 log_path=LOGDIR, eval_freq=eval_freq,
                                 n_eval_episodes=EVAL_EPISODES,
                                 deterministic=True)

    model.set_logger(logger)
    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    env.close()
    del env
    model.save(os.path.join(LOGDIR, "final_model"))


if __name__ == '__main__':
    train()
