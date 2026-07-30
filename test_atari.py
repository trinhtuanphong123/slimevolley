"""
Pixel observation environment (Atari-compatible example, with 84x84 resized
4-frame stack). Human-playable demo using stable-baselines3's Atari wrappers.

The agent (right side) defaults to NOOP; take over with the arrow keys.
"""

import numpy as np
import gymnasium as gym
import slimevolleygym
from slimevolleygym import render_atari, FrameStack
from slimevolleygym._rendering import SimpleImageViewer
from stable_baselines3.common.atari_wrappers import AtariWrapper
from pyglet.window import key
from time import sleep


def toAtariAction(action):
  """
  action_table = [[0, 0, 0], # NOOP
                  [1, 0, 0], # LEFT (forward)
                  [1, 0, 1], # UPLEFT (forward jump)
                  [0, 0, 1], # UP (jump)
                  [0, 1, 1], # UPRIGHT (backward jump)
                  [0, 1, 0]] # RIGHT (backward)
  """
  left = action[0]
  right = action[1]
  jump = action[2]
  if left == right:
    left = 0
    right = 0
  if left == 1 and jump == 0:
    return 1
  if left == 1 and jump == 1:
    return 2
  if right == 1 and jump == 0:
    return 5
  if right == 1 and jump == 1:
    return 4
  if jump == 1:
    return 3
  return 0


# simulate typical Atari Env:
if __name__ == "__main__":

  manualAction = [0, 0, 0]  # forward, backward, jump
  manualMode = False

  # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
  def key_press(k, mod):
    global manualMode, manualAction
    if k == key.LEFT:  manualAction[0] = 1
    if k == key.RIGHT: manualAction[1] = 1
    if k == key.UP:    manualAction[2] = 1
    if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

  def key_release(k, mod):
    global manualMode, manualAction
    if k == key.LEFT:  manualAction[0] = 0
    if k == key.RIGHT: manualAction[1] = 0
    if k == key.UP:    manualAction[2] = 0

  viewer = SimpleImageViewer(maxwidth=2160)

  env = gym.make("SlimeVolleyNoFrameskip-v0")
  # standard Atari pre-processing: random no-ops, frame skip 4, 84x84 grayscale warp, then 4-frame stack
  env = AtariWrapper(env, clip_reward=False)
  env = FrameStack(env, 4)
  obs, _ = env.reset(seed=689)

  for t in range(10000):

    if manualMode:  # override with keyboard
      action = toAtariAction(manualAction)
    else:
      action = 0  # NOOP (your agent here)

    obs, reward, terminated, truncated, info = env.step(action)

    if reward > 0 or reward < 0:
      print(t, reward)
      manualMode = False

    viewer.imshow(render_atari(obs))
    sleep(0.08)

    if t == 0:
      viewer.window.on_key_press = key_press
      viewer.window.on_key_release = key_release

    if terminated or truncated:
      obs, _ = env.reset()

  viewer.close()
  env.close()
