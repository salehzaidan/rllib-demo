import os

import gym
from ray.rllib.algorithms.a3c import A3CConfig

CHECKPOINT_PATH = "CartPole/a3c/checkpoint/checkpoint_000010"
ENV_NAME = "CartPole-v1"

algo = (
    A3CConfig()
    .environment(env=ENV_NAME)
    .framework("torch")
    .build()
    .from_checkpoint(os.path.abspath(CHECKPOINT_PATH))
)
env = gym.make(ENV_NAME)

episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = algo.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward

    env.render()
    print(f"action={action} obs={obs} reward={reward} episode_reward={episode_reward}")
