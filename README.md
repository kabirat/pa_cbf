# Perception-Aware Control Barrier Function for Safe Navigation Under Uncertain LiDAR Observation

This repository contains the implementation of the **Perception-Aware Control Barrier Function (PA-CBF)** framework presented in the paper:

> **Perception-Aware Control Barrier Function for Safe Navigation Under Uncertain LiDAR Observation**
> Kabirat Olayemi, Mien Van, Sean McLoone, and Rajeem Kutty Thomas

The proposed method combines:

* Beam-wise LiDAR-based Control Barrier Functions (CBFs)
* Uncertainty-aware range-rate estimation from sequential LiDAR measurements
* Finite-horizon predictive safety constraints
* Real-time quadratic-program safety filtering
* Reinforcement Learning (RL) navigation policies

The framework provides probabilistic sampled-data safety guarantees while maintaining computational efficiency suitable for real-time robotic navigation.

---

## Video Demonstration

A demonstration video of representative navigation scenarios is available at:

📹 https://youtu.be/pFp74QQNXdM

---

## Features

* LiDAR-based perception-aware safety filtering
* Robust range-rate estimation under sensor uncertainty
* Predictive PA-CBF with finite-horizon safety constraints
* Convex Quadratic Program (QP) formulation
* Compatible with reinforcement learning navigation policies
* Dynamic obstacle avoidance in uncertain environments

---

## Repository Structure

```text
.
├── envs/               # Navigation environments
├── safety_filter/      # PA-CBF implementation
├── rl/                 # RL policy training and evaluation
├── configs/            # Experiment configurations
├── scripts/            # Training and evaluation scripts
├── results/            # Logs and experimental results
├── videos/             # Demonstration videos
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kabirat/pa_cbf.git
cd <repository>
```

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Evaluate a Trained Policy

```bash
python evaluate.py
```

### Train the RL Policy

```bash
python train.py
```

### Run the PA-CBF Safety Filter

```bash
python run_navigation.py
```

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{olayemi2025pacbf,
  title={Perception-Aware Control Barrier Function for Safe Navigation Under Uncertain LiDAR Observation},
  author={Olayemi, Kabirat and Van, Mien and McLoone, Sean and Thomas, Rajeem Kutty},
  journal={IEEE International Conference on Mechatronics and Automation (ICMA)},
  year={2026}
}
```

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

---

## Contact

**Kabirat Olayemi**
Queen's University Belfast
Email: [kolayemi01@qub.ac.uk](mailto:kolayemi01@qub.ac.uk)
