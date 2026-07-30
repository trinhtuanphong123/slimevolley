#!/usr/bin/env python3

# Train single CPU PPO on slimevolley (state observations).
# Should solve it (beat the built-in AI on average over 1000 trials) in a few
# hours on a single CPU, within ~3M steps.
#
# Requires stable-baselines3: pip install slimevolleygym[training]

import os
import gymnasium as gym
import slimevolleygym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_FREQ = 250000
EVAL_EPISODES = 1000
LOGDIR = "ppo"  # moved to zoo afterwards.

os.makedirs(LOGDIR, exist_ok=True)
logger = configure(LOGDIR, ["stdout", "csv"])

env = Monitor(gym.make("SlimeVolley-v0"))
eval_env = Monitor(gym.make("SlimeVolley-v0"))
env.reset(seed=SEED)
eval_env.reset(seed=SEED)

# PPO1 hyperparameters mapped to stable-baselines3 PPO:
#   timesteps_per_actorbatch -> n_steps
#   optim_epochs             -> n_epochs
#   optim_stepsize           -> learning_rate
#   optim_batchsize          -> batch_size
#   lam                      -> gae_lambda
#   entcoeff                 -> ent_coef
#   clip_param               -> clip_range
model = PPO("MlpPolicy", env,
            n_steps=4096, batch_size=64, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0, verbose=2, seed=SEED)

eval_callback = EvalCallback(eval_env, best_model_save_path=LOGDIR,
                             log_path=LOGDIR, eval_freq=EVAL_FREQ,
                             n_eval_episodes=EVAL_EPISODES, deterministic=True)

model.set_logger(logger)
model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

model.save(os.path.join(LOGDIR, "final_model"))  # probably never get to this point.

env.close()
