import os

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment("CartPole-v1")
    .framework("torch")
    .build()
)

num_iters = 10
for i in range(num_iters):
    result = algo.train()
    print(pretty_print(result))

    if i == num_iters - 1:
        checkpoint_dir = algo.save(os.path.abspath("CartPole/ppo/checkpoint"))
        print(f"Checkpoint saved in directory {checkpoint_dir}")
