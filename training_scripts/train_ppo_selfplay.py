#!/usr/bin/env python3

# Simple self-play PPO trainer (stable-baselines3).
#
# The opponent inside the environment is replaced by the latest "champion"
# (a previously checkpointed version of the agent). A new champion is promoted
# only when the current agent beats the previous one by BEST_THRESHOLD.
#
# Requires stable-baselines3: pip install slimevolleygym[training]

import os
import gymnasium as gym
import slimevolleygym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from shutil import copyfile

# Settings
SEED = 17
NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e5)
EVAL_EPISODES = int(1e2)
BEST_THRESHOLD = 0.5  # must achieve a mean score above this to replace prev best self

LOGDIR = "ppo_selfplay"


class SlimeVolleySelfPlayEnv(slimevolleygym.SlimeVolleyEnv):
  # the normal single-player env, but the opponent is the best self-play model
  def __init__(self, render_mode=None):
    super(SlimeVolleySelfPlayEnv, self).__init__(render_mode=render_mode)
    self.policy = self          # the opponent policy is 'self'
    self.best_model = None
    self.best_model_filename = None

  def predict(self, obs):  # the opponent's policy
    if self.best_model is None:
      return self.action_space.sample()  # random action until a champion exists
    else:
      action, _ = self.best_model.predict(obs, deterministic=True)
      return action

  def reset(self, *, seed=None, options=None):
    # load the latest champion model if one exists
    modellist = ([f for f in os.listdir(LOGDIR) if f.startswith("history")]
                 if os.path.exists(LOGDIR) else [])
    modellist.sort()
    if len(modellist) > 0:
      filename = os.path.join(LOGDIR, modellist[-1])  # the latest best model
      if filename != self.best_model_filename:
        print("loading model: ", filename)
        self.best_model_filename = filename
        if self.best_model is not None:
          del self.best_model
        self.best_model = PPO.load(filename, env=self)
    return super(SlimeVolleySelfPlayEnv, self).reset(seed=seed, options=options)


class SelfPlayCallback(EvalCallback):
  # only promote a new champion if it beats the prev self by BEST_THRESHOLD;
  # after saving, reset the best score to BEST_THRESHOLD
  def __init__(self, *args, **kwargs):
    super(SelfPlayCallback, self).__init__(*args, **kwargs)
    self.best_mean_reward = BEST_THRESHOLD
    self.generation = 0

  def _on_step(self) -> bool:
    result = super(SelfPlayCallback, self)._on_step()
    if result and self.best_mean_reward > BEST_THRESHOLD:
      self.generation += 1
      print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
      print("SELFPLAY: new best model, bumping up generation to", self.generation)
      source_file = os.path.join(LOGDIR, "best_model.zip")
      backup_file = os.path.join(LOGDIR, "history_" + str(self.generation).zfill(8) + ".zip")
      if os.path.exists(source_file):
        copyfile(source_file, backup_file)
      self.best_mean_reward = BEST_THRESHOLD
    return result


def rollout(env, policy):
  """ play the trained agent against the current champion """
  obs, _ = env.reset()
  terminated = False
  truncated = False
  total_reward = 0
  while not (terminated or truncated):
    action, _states = policy.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
  return total_reward


def train():
  os.makedirs(LOGDIR, exist_ok=True)
  logger = configure(LOGDIR, ["stdout", "csv"])

  env = Monitor(SlimeVolleySelfPlayEnv())
  env.reset(seed=SEED)

  eval_env = Monitor(SlimeVolleySelfPlayEnv())
  eval_env.reset(seed=SEED + 1)

  model = PPO("MlpPolicy", env,
              n_steps=4096, batch_size=64, n_epochs=10,
              learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, verbose=2, seed=SEED)

  eval_callback = SelfPlayCallback(eval_env,
    best_model_save_path=LOGDIR,
    log_path=LOGDIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=False)

  model.set_logger(logger)
  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  model.save(os.path.join(LOGDIR, "final_model"))
  env.close()


if __name__ == "__main__":
  train()
