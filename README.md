# Seaquest_RL

---

This repository contains implementations of several DQN variants:

- **DQN** (using Stable-Baseline3 as foundation for variants)
- **Categorical DQN (C51)**
- **NoisyNet DQN**
- **NoisyNet/Categorical DQN** (combined variant using NoisyNet and C51)

---


## Project Structure

The project is organized as follows:

```plaintext
├── algorithms/
│   ├── categorical_dqn.py       # C51 implementation
│   ├── noisy_dqn.py             # NoisyNet DQN implementation
│   └── noisy_categorical_dqn.py # Combined variant using NoisyNet and C51
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
└── main.py                      # Main entry point
```
## Prerequisites

- **Python**: 3.7 or higher

## Required Libraries

To get started, install the following libraries:

```bash
pip install gymnasium[atari] stable-baselines3[extra] torch numpy
