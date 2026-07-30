#!/usr/bin/env python3
# Trains a slime agent from states with multiple parallel workers via SB3's
# SubprocVecEnv (this replaces the old MPI-based train_ppo_mpi.py).
#
# run with:  python train_ppo_vec.py   (set NUM_ENVS to your CPU core count)
#
# Requires stable-baselines3: pip install slimevolleygym[training]

import os
import gymnasium as gym
import slimevolleygym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

NUM_TIMESTEPS = int(2e8)
NUM_ENVS = 96          # set to the number of CPU cores you have
SEED = 831
EVAL_EPISODES = 1000
LOGDIR = "ppo_vec"


def make_env(rank):
    def _init():
        env = gym.make("SlimeVolley-v0")
        env.reset(seed=SEED + rank)
        return env
    set_random_seed(SEED + rank)
    return _init


def train():
    os.makedirs(LOGDIR, exist_ok=True)
    logger = configure(LOGDIR, ["stdout", "csv"])

    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(1000 + i) for i in range(1)]))

    model = PPO("MlpPolicy", env,
                n_steps=4096, batch_size=64, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, verbose=1, seed=SEED)

    # EvalCallback evaluates every eval_freq * n_envs timesteps; divide so the
    # overall evaluation interval is ~200k timesteps regardless of NUM_ENVS.
    eval_freq = max(1, 200000 // NUM_ENVS)
    eval_callback = EvalCallback(eval_env, best_model_save_path=LOGDIR,
                                 log_path=LOGDIR, eval_freq=eval_freq,
                                 n_eval_episodes=EVAL_EPISODES, deterministic=True)

    model.set_logger(logger)
    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    env.close()
    del env
    model.save(os.path.join(LOGDIR, "final_model"))


if __name__ == '__main__':
    train()
