"""
Headless smoke test for the SlimeVolley Gymnasium environments.

Run with:  python test_smoke.py
No display required (render_mode=None / rgb_array only). Verifies the
Gymnasium API migration, both observation modes, multi-agent stepping,
determinism, the rendering backends, and the bundled policy models.
"""

import numpy as np
import gymnasium as gym

import slimevolleygym
from slimevolleygym import BaselinePolicy, multiagent_rollout
from slimevolleygym.mlp import makeSlimePolicy, makeSlimePolicyLite


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok -", msg)


def test_state_env():
    print("[state env]")
    env = gym.make("SlimeVolley-v0")
    obs, info = env.reset(seed=0)
    check(obs.shape == (12,) and obs.dtype == np.float32, "reset returns (12,) float32 obs + info")
    out = env.step(env.action_space.sample())
    check(len(out) == 5, "step returns 5-tuple (terminated/truncated split)")
    obs, reward, terminated, truncated, info = out
    check("otherObs" in info, "info carries multi-agent otherObs")

    obs, info = env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample(), env.action_space.sample())
    check(info["otherObs"].shape == (12,), "multi-agent step(action, otherAction) works")

    o1 = gym.make("SlimeVolley-v0").reset(seed=42)[0]
    o2 = gym.make("SlimeVolley-v0").reset(seed=42)[0]
    check(np.allclose(o1, o2), "reset(seed=...) is deterministic")
    check(env.render() is None, "render_mode=None is a headless no-op")
    env.close()


def test_pixel_env():
    print("[pixel env]")
    penv = gym.make("SlimeVolleyPixel-v0")
    pobs, _ = penv.reset(seed=0)
    check(pobs.shape == (84, 168, 3) and pobs.dtype == np.uint8, "pixel obs is (84,168,3) uint8")
    pobs, _, _, _, pi = penv.step(penv.action_space.sample())
    check(pi["otherObs"].shape == (84, 168, 3), "pixel multi-agent otherObs flips horizontally")

    aenv = gym.make("SlimeVolleyNoFrameskip-v0")
    check(isinstance(aenv.action_space, gym.spaces.Discrete), "atari env has Discrete(6) action space")
    aenv.close()


def test_rendering():
    print("[rendering]")
    penv = gym.make("SlimeVolleyPixel-v0", render_mode="rgb_array")
    penv.reset(seed=0); penv.step(penv.action_space.sample())
    arr = penv.render()
    check(isinstance(arr, np.ndarray) and arr.ndim == 3, "pixel rgb_array returns an image")
    penv.close()


def test_policies():
    print("[policies]")
    env = gym.make("SlimeVolley-v0")
    bp = BaselinePolicy()
    obs, _ = env.reset(seed=0)
    check(len(bp.predict(obs)) == 3, "BaselinePolicy.predict returns a 3-action")
    score, steps = multiagent_rollout(env, BaselinePolicy(), BaselinePolicy())
    check(isinstance(score, (int, float)) and steps > 0, "multiagent_rollout runs a full episode")


    ga = makeSlimePolicyLite("zoo/ga_sp/ga.json")
    check(np.asarray(ga.predict(obs)).shape == (3,), "GA JSON model loads and predicts")
    env.close()


def test_sb3():
    print("[stable-baselines3]")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
    except ImportError:
        print("  skip - stable-baselines3 not installed (pip install slimevolleygym[training])")
        return
    env = Monitor(gym.make("SlimeVolley-v0"))
    env.reset(seed=0)
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, n_epochs=1,
                verbose=0, device="cpu")
    model.learn(total_timesteps=256)
    act, _ = model.predict(env.reset(seed=1)[0], deterministic=True)
    check(np.asarray(act).shape == (3,), "SB3 PPO trains and predicts on SlimeVolley-v0")
    env.close()


if __name__ == "__main__":
    test_state_env()
    test_pixel_env()
    test_rendering()
    test_policies()
    test_sb3()
    print("\nALL SMOKE TESTS PASSED")
