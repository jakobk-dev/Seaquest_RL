# Seaquest_RL

---

This repository contains implementations of several DQN variants:

- **DQN** (using Stable-Baseline3 as foundation for variants)
- **Categorical DQN (C51)**
- **NoisyNet DQN**
- **NoisyNet/Categorical DQN** (combined variant using NoisyNet and C51)
- **Dynamic Action Repetition DQN** -> dar
- **Double Q-Learning DQN** -> double
- **Dynamic Frame Skip DQN** -> dfdqn
- **Rainbow DQN** -> rainbow
- **Prioritized Experience Replay DQN** -> per

---


## Project Structure

The project is organized as follows:

```plaintext
├── algorithms/
│   ├── categorical_dqn.py       # C51 implementation
│   ├── noisy_dqn.py             # NoisyNet DQN implementation
│   └── noisy_categorical_dqn.py # Combined variant using NoisyNet and C51
│   └── *.py # All other algorithm python files are in this folder
├── train.py                     # Training script
└── main.py                      # Main entry point
```
## Prerequisites

- **Python**: 3.10.12

## Required Libraries

To get started, install the following libraries:

```bash
pip install gymnasium[atari] stable-baselines3[extra] torch numpy
