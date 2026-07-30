#!/usr/bin/env python3
"""
Train a PPO agent (stable-baselines3) on SlimeVolley-v0 (state observations).

Standard SB3 training pipeline:
  * environment created with gymnasium.make
  * PPO("MlpPolicy", ...) with SB3 hyperparameters + TensorBoard logging
  * EvalCallback that periodically evaluates and writes the best checkpoint
    to zoo/ppo/
  * model.learn(total_timesteps=...)

This file is configured for a SMOKE TEST (TOTAL_TIMESTEPS = 10_000): it runs
quickly, the policy/value losses should move between rollouts, and
zoo/ppo/best_model.zip should be produced. For real training, raise
TOTAL_TIMESTEPS (a few million steps to reliably beat the built-in baseline)
and N_EVAL_EPISODES (~1000).
"""

import os

import gymnasium

import slimevolleygym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# ---- configuration ---------------------------------------------------------
TOTAL_TIMESTEPS = 10_000        # smoke test; use ~3e6+ for real training
SEED = 721
EVAL_FREQ = 5_000               # evaluate every N env steps (n_envs = 1)
N_EVAL_EPISODES = 10            # raise to ~1000 for a trustworthy evaluation
LOGDIR = "zoo/ppo"              # best model + evaluation logs are written here
TENSORBOARD_LOG = "./ppo_tensorboard"

os.makedirs(LOGDIR, exist_ok=True)

# ---- environments ----------------------------------------------------------
env = Monitor(gymnasium.make("SlimeVolley-v0"))
eval_env = Monitor(gymnasium.make("SlimeVolley-v0"))
env.reset(seed=SEED)
eval_env.reset(seed=SEED)

# ---- model (stable-baselines3 PPO) ----------------------------------------
# Hyperparameters use the SB3 PPO API (NOT the old stable_baselines PPO1):
#   n_steps      rollout length collected per environment
#   batch_size   minibatch size for the optimisation epochs (divides n_steps)
#   n_epochs     optimisation epochs over each rollout
#   gamma / gae_lambda  discount factor / GAE lambda
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log=TENSORBOARD_LOG,
    seed=SEED,
)

# ---- evaluation callback (auto-saves the best model) ----------------------
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=LOGDIR,
    log_path=LOGDIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
)

# ---- train -----------------------------------------------------------------
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
model.save(os.path.join(LOGDIR, "ppo_slimevolley"))

env.close()
eval_env.close()
print(f"Training complete. Artifacts written to {LOGDIR}/ (logs: {TENSORBOARD_LOG}/).")
