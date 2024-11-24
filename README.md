# Seaquest_RL

---

This repository contains implementations of several DQN variants:

- **DQN** (using Stable-Baseline3 as foundation for variants) -> baseline
- **Categorical DQN (C51)** -> categorical
- **NoisyNet DQN** -> noisy
- **NoisyNet/Categorical DQN** (combined variant using NoisyNet and C51) -> noisy_categorical
- **Dynamic Action Repetition DQN** -> dar
- **Double Q-Learning DQN** -> double
- **Dynamic Frame Skip DQN** -> dfdqn
- **Rainbow DQN** -> rainbow
- **Prioritized Experience Replay DQN** -> per

---

To run the code:

`python3 main.py [algorithm]`

The algorithm after main.py would be dar, rainbow, per, dfdqn, and so on. For example this would be running categorical DQN:

`python3 main.py categorical`

## Project Structure

The project is organized as follows:

```plaintext
├── algorithms/
│   ├── categorical_dqn.py                 # C51 implementation
│   ├── double_dqn.py
│   ├── dynamic_action_repetition_dqn.py
│   ├── dynamic_frame_skip_dqn.py 
│   ├── noisy_categorical_dqn.py           # Combined variant using NoisyNet and C51
│   ├── noisy_dqn.py                       # NoisyNet DQN implementation
│   ├── per_dqn.py                         # Prioritized Experience Replay DQN
│   └── rainbow_dqn.py
├── train.py                               # Training script
└── main.py                                # Main entry point
```
## Prerequisites

- **Python**: 3.10.12

## Required Libraries

To get started, install the following libraries:

```bash
pip install gymnasium[atari] stable-baselines3[extra] torch numpy
