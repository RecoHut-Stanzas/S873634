# batch-bandits
Implementation of popular bandit algorithms in batch environments. 

Source code to our paper ["The Impact of Batch Learning in Stochastic Bandits"](https://arxiv.org/abs/2111.02071) accepted at the workshop on the [Ecological Theory of Reinforcement Learning](https://sites.google.com/view/ecorl2021/home?authuser=0), NeurIPS 2021.

## Overview

The repository provides an opportunuty to run simulations or replay logged datasets in _sequential batch_ manner -  sequential interaction with the environment when responses are grouped in batches and observed by the agent only at the end of each batch. Broadly speaking, sequential batch learning is a more generalized way of learning which covers both offline and online settings as special cases bringing together their advantages.


## Framework

Two particularly useful versions of the [multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit#Contextual_bandit) are implemented: Stochastic Multi-Armed Bandit ([MAB](MAB)) and Contextual Multi-Armed Bandit ([CMAB](CMAB)).
The key feature of the project is that both versions support parameter `batch_size` - a certain period of time when the agent interacts with the environment "blindly". Despite the batch setting is a property of the environment, this limitation is considered from a policy perspective. With this, it is assumed that it is not the online agent who works with the batch environment, but the batch policy interacts with the online environment.

The project is built upon [RL-GLue](https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0) framework, which provides an interface to connect agents, environments, and experiment programs. Note, that [MAB/rl_glue.py](MAB/rl_glue.py) and [CMAB/rl_glue.py](CMAB/rl_glue.py) were adapted to make batch interaction possible.

### Implemented algorithms

Version | Algorithm | Comment
------------ | ------------- | ------------- 
MAB | ε - greedy | -
MAB | Thompson Sampling | -
MAB | UCB | -
CMAB | LinTS | see [link](https://gdmarmerola.github.io/ts-for-contextual-bandits/) (and references therein) for more details
CMAB | LinUCB | see [article](https://arxiv.org/abs/1003.0146) for theoretical description
CMAB | Offline evaluator | policy evaluation technique; see [article](https://arxiv.org/abs/1003.5956) for theoretical quarantees

## Project structure
```
.
├── [8.4K]  basics
│   ├── [1.9K]  base_agent.py
│   ├── [1.7K]  base_environment.py
│   ├── [   0]  __init__.py
│   └── [ 812]  random_agent.py
├── [ 31K]  CMAB
│   ├── [   0]  __init__.py
│   ├── [7.3K]  LinTS.py
│   ├── [7.6K]  LinUCB.py
│   ├── [2.5K]  offline_evaluator.py
│   ├── [3.7K]  replay_env.py
│   └── [6.0K]  rl_glue.py
├── [3.0M]  experiments
│   ├── [100K]  CMAB_demo_mushroom_dataset.ipynb
│   ├── [ 96K]  CMAB_demo_simulated_data.ipynb
│   ├── [1.1M]  data
│   │   ├── [365K]  agaricus-lepiota.data
│   │   ├── [6.7K]  agaricus-lepiota.names
│   │   ├── [668K]  mushroom_data_final.pickle
│   │   └── [ 46K]  mushroom_data_preprocessing.ipynb
│   ├── [1.8M]  EcoRL-NeurIPS-2021
│   │   ├── [6.4K]  comparison.py
│   │   ├── [2.7K]  LinTS_dynamic_by_batches.py
│   │   ├── [2.6K]  LinUCB_dynamic_by_batches.py
│   │   ├── [1.7M]  results
│   │   │   ├── [811K]  pictures
│   │   │   │   ├── [ 49K]  LinTS+LinUCB.png
│   │   │   │   ├── [582K]  simulator
│   │   │   │   │   ├── [ 66K]  TS_envs1-3.png
│   │   │   │   │   ├── [ 82K]  TS_envs1-6.png
│   │   │   │   │   ├── [ 63K]  TS_envs4-6.png
│   │   │   │   │   ├── [ 58K]  TS+UCB_envs1-3.png
│   │   │   │   │   ├── [ 88K]  TS+UCB_envs4-6.png
│   │   │   │   │   ├── [ 65K]  UCB_envs1-3.png
│   │   │   │   │   ├── [ 90K]  UCB_envs1-6.png
│   │   │   │   │   └── [ 66K]  UCB_envs4-6.png
│   │   │   │   ├── [ 90K]  TS+UCB_envs1-3.png
│   │   │   │   └── [ 87K]  TS+UCB_envs4-6.png
│   │   │   ├── [480K]  TS
│   │   │   │   └── [476K]  dynamic_by_batches
│   │   │   │       ├── [ 481]  dyn_by_batch_[0.35, 0.18, 0.47, 0.61] batch_regret.pickle
│   │   │   │       ├── [ 78K]  dyn_by_batch_[0.35, 0.18, 0.47, 0.61] online_regret.pickle
│   │   │   │       ├── [ 481]  dyn_by_batch_[0.4, 0.75, 0.57, 0.49] batch_regret.pickle
│   │   │   │       ├── [ 78K]  dyn_by_batch_[0.4, 0.75, 0.57, 0.49] online_regret.pickle
│   │   │   │       ├── [ 481]  dyn_by_batch_[0.7, 0.1] batch_regret.pickle
│   │   │   │       ├── [ 78K]  dyn_by_batch_[0.7, 0.1] online_regret.pickle
│   │   │   │       ├── [ 481]  dyn_by_batch_[0.7, 0.4] batch_regret.pickle
│   │   │   │       ├── [ 78K]  dyn_by_batch_[0.7, 0.4] online_regret.pickle
│   │   │   │       ├── [ 481]  dyn_by_batch_[0.7, 0.5, 0.3, 0.1] batch_regret.pickle
│   │   │   │       ├── [ 78K]  dyn_by_batch_[0.7, 0.5, 0.3, 0.1] online_regret.pickle
│   │   │   │       ├── [ 481]  dyn_by_batch_[0.7, 0.5] batch_regret.pickle
│   │   │   │       └── [ 78K]  dyn_by_batch_[0.7, 0.5] online_regret.pickle
│   │   │   └── [480K]  UCB
│   │   │       └── [476K]  dynamic_by_batches
│   │   │           ├── [ 481]  dyn_by_batch_[0.35, 0.18, 0.47, 0.61] batch_regret.pickle
│   │   │           ├── [ 78K]  dyn_by_batch_[0.35, 0.18, 0.47, 0.61] online_regret.pickle
│   │   │           ├── [ 481]  dyn_by_batch_[0.4, 0.75, 0.57, 0.49] batch_regret.pickle
│   │   │           ├── [ 78K]  dyn_by_batch_[0.4, 0.75, 0.57, 0.49] online_regret.pickle
│   │   │           ├── [ 481]  dyn_by_batch_[0.7, 0.1] batch_regret.pickle
│   │   │           ├── [ 78K]  dyn_by_batch_[0.7, 0.1] online_regret.pickle
│   │   │           ├── [ 481]  dyn_by_batch_[0.7, 0.4] batch_regret.pickle
│   │   │           ├── [ 78K]  dyn_by_batch_[0.7, 0.4] online_regret.pickle
│   │   │           ├── [ 481]  dyn_by_batch_[0.7, 0.5, 0.3, 0.1] batch_regret.pickle
│   │   │           ├── [ 78K]  dyn_by_batch_[0.7, 0.5, 0.3, 0.1] online_regret.pickle
│   │   │           ├── [ 481]  dyn_by_batch_[0.7, 0.5] batch_regret.pickle
│   │   │           └── [ 78K]  dyn_by_batch_[0.7, 0.5] online_regret.pickle
│   │   ├── [2.1K]  TS_dynamic_by_batches.py
│   │   └── [2.1K]  UCB_dynamic_by_batches.py
│   ├── [2.2K]  LinTS_dynamic_by_timesteps.py
│   ├── [2.2K]  LinUCB_dynamic_by_timesteps.py
│   ├── [1.9K]  TS_dynamic_by_timesteps.py
│   └── [1.9K]  UCB_dynamic_by_timesteps.py
├── [ 36K]  images
│   └── [ 32K]  process_flow.svg
├── [1.0K]  LICENSE
├── [ 24K]  MAB
│   ├── [7.7K]  bandit_agents.py
│   ├── [   0]  __init__.py
│   ├── [3.2K]  k_arm_env.py
│   ├── [5.9K]  rl_glue.py
│   └── [3.5K]  wrapper.py
├── [2.3K]  README.md
└── [ 17K]  utilities
    ├── [1.6K]  data_generator.py
    ├── [2.4K]  dataloader.py
    ├── [2.0K]  plot_script.py
    ├── [1.5K]  replay_buffer.py
    ├── [2.1K]  run_experiment.py
    └── [3.0K]  softmax.py

 3.1M used in 15 directories, 74 files
```