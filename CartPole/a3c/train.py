import os

from ray.rllib.algorithms.a3c import A3CConfig
from ray.tune.logger import pretty_print

algo = (
    A3CConfig()
    .training(gamma=0.95)
    .rollouts(num_rollout_workers=1)
    .environment("CartPole-v1")
    .framework("torch")
    .build()
)

num_iters = 10
for i in range(num_iters):
    result = algo.train()
    print(pretty_print(result))

    if i == num_iters - 1:
        checkpoint_dir = algo.save(os.path.abspath("CartPole/a3c/checkpoint"))
        print(f"Checkpoint saved in directory {checkpoint_dir}")
