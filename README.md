

```markdown
# ARTEMIS: Adversarial-Resistant Temporal Embedding Model for Intelligent Security in Blockchain Fraud Detection

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **ARTEMIS** published in *Information Processing & Management* (2025).

## 📋 Overview

ARTEMIS addresses six fundamental challenges in blockchain fraud detection:
- **L1**: Discretization errors from snapshot-based temporal modeling
- **L2**: Information loss in recency-biased memory mechanisms
- **L3**: Limited propagation range failing to detect coordinated attacks
- **L4**: Vulnerability to adversarial manipulation
- **L5**: Catastrophic forgetting across temporal distributions
- **L6**: Inability to adapt rapidly to emerging attack patterns

### Key Features

✨ **Neural ODE Integration** - Continuous-time temporal modeling eliminates O(Δt²) discretization errors  
🧠 **Anomaly-Aware Memory** - Information-prioritized storage retains critical patterns  
🔗 **Multi-Hop Broadcast** - Extended propagation detects coordinated Sybil attacks  
🛡️ **Adversarial Robustness** - Spectral normalization + PGD training provides certified defenses  
📚 **Continual Learning** - Elastic Weight Consolidation prevents catastrophic forgetting  
⚡ **Meta-Learning** - Rapid adaptation to emerging attack patterns with limited examples

## 🏆 Results

| Method | Recall | F1-Score | AUC | FPR |
|--------|--------|----------|-----|-----|
| GAT | 75.12% | 74.19% | 73.58% | 24.91% |
| GraphSAGE | 76.34% | 75.48% | 74.89% | 23.72% |
| TGAT | 79.58% | 78.73% | 78.16% | 20.38% |
| TGN | 80.34% | 79.89% | 79.21% | 19.71% |
| GrabPhisher | 82.15% | 81.92% | 81.34% | 17.83% |
| 2DynEthNet | 86.28% | 85.70% | 84.73% | 14.50% |
| **ARTEMIS (Ours)** | **91.47%** | **90.18%** | **88.92%** | **8.73%** |

**Improvements over best baseline (2DynEthNet):**
- ✅ +5.19% Recall improvement
- ✅ +4.48% F1-Score improvement  
- ✅ 33.5% better adversarial robustness at ε=0.20
- ✅ Statistical significance: p < 0.001, Cohen's d = 1.83

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YourUsername/ARTEMIS.git
cd ARTEMIS
```

2. **Create virtual environment**
```bash
conda create -n artemis python=3.8
conda activate artemis
```

3. **Install dependencies**
```bash
# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric==2.3.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install other dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. **Download ETGraph dataset**
```bash
# Download from XBlock
wget https://xblock.pro/xblock/ETGraph.zip
unzip ETGraph.zip -d data/

# Or use the provided script
python scripts/download_data.py
```

2. **Preprocess data**
```bash
python scripts/preprocess.py --data_dir data/ETGraph --output_dir data/processed
```

Expected directory structure:
```
data/
├── ETGraph/
│   ├── blocks_8000000_8600000/
│   ├── blocks_14250000_14500000/
│   └── phishing_labels.csv
└── processed/
    ├── task_2019_Q1.pt
    ├── task_2019_Q2.pt
    ├── ...
    └── task_2022_Q3.pt
```

## 🔬 Running Experiments

### Training ARTEMIS

**Full model with all components:**
```bash
python train.py \
    --config configs/artemis_full.yaml \
    --data_dir data/processed \
    --output_dir results/artemis \
    --gpu 0
```

**Ablation studies:**
```bash
# Without Neural ODE
python train.py --config configs/ablation/wo_neural_ode.yaml

# Without Anomaly Memory
python train.py --config configs/ablation/wo_anomaly_memory.yaml

# Without Multi-Hop
python train.py --config configs/ablation/wo_multihop.yaml

# Without Adversarial Defense
python train.py --config configs/ablation/wo_adversarial.yaml

# Without EWC
python train.py --config configs/ablation/wo_ewc.yaml

# Without Meta-Learning
python train.py --config configs/ablation/wo_meta.yaml
```

### Training Baselines

```bash
# 2DynEthNet
python train_baseline.py --model 2DynEthNet --data_dir data/processed

# GrabPhisher
python train_baseline.py --model GrabPhisher --data_dir data/processed

# TGN
python train_baseline.py --model TGN --data_dir data/processed

# TGAT
python train_baseline.py --model TGAT --data_dir data/processed

# GraphSAGE
python train_baseline.py --model GraphSAGE --data_dir data/processed

# GAT
python train_baseline.py --model GAT --data_dir data/processed
```

### Evaluation

**Evaluate trained model:**
```bash
python evaluate.py \
    --checkpoint results/artemis/best_model.pth \
    --data_dir data/processed \
    --output_dir results/evaluation
```

**Adversarial robustness testing:**
```bash
python evaluate_robustness.py \
    --checkpoint results/artemis/best_model.pth \
    --epsilon 0.05 0.10 0.15 0.20 \
    --attack_type pgd \
    --output_dir results/robustness
```

**Statistical validation:**
```bash
python statistical_analysis.py \
    --results_dir results/ \
    --output_dir results/statistics
```

## 📁 Repository Structure

```
ARTEMIS/
├── configs/                      # Configuration files
│   ├── artemis_full.yaml        # Full ARTEMIS configuration
│   └── ablation/                # Ablation study configs
│       ├── wo_neural_ode.yaml
│       ├── wo_anomaly_memory.yaml
│       ├── wo_multihop.yaml
│       ├── wo_adversarial.yaml
│       ├── wo_ewc.yaml
│       └── wo_meta.yaml
│
├── data/                        # Data directory (not tracked)
│   ├── ETGraph/                 # Raw dataset
│   └── processed/               # Preprocessed data
│
├── models/                      # Model implementations
│   ├── artemis.py              # Main ARTEMIS model
│   ├── neural_ode.py           # Neural ODE module (L1)
│   ├── anomaly_memory.py       # Anomaly-aware memory (L2)
│   ├── multihop.py             # Multi-hop broadcast (L3)
│   ├── adversarial_defense.py  # Adversarial defenses (L4)
│   ├── ewc.py                  # Elastic Weight Consolidation (L5)
│   └── meta_learning.py        # Meta-learning module (L6)
│
├── baselines/                   # Baseline implementations
│   ├── 2dynethnet.py
│   ├── grabphisher.py
│   ├── tgn.py
│   ├── tgat.py
│   ├── graphsage.py
│   └── gat.py
│
├── scripts/                     # Utility scripts
│   ├── download_data.py        # Dataset download
│   ├── preprocess.py           # Data preprocessing
│   └── visualize_results.py    # Result visualization
│
├── utils/                       # Helper functions
│   ├── data_loader.py          # Data loading utilities
│   ├── metrics.py              # Evaluation metrics
│   ├── logger.py               # Logging utilities
│   └── visualization.py        # Plotting functions
│
├── train.py                     # Main training script
├── train_baseline.py           # Baseline training
├── evaluate.py                 # Evaluation script
├── evaluate_robustness.py      # Adversarial evaluation
├── statistical_analysis.py     # Statistical validation
│
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
├── README.md                   # This file
├── LICENSE                     # License file
└── .gitignore                  # Git ignore rules
```

## ⚙️ Configuration

Key hyperparameters in `configs/artemis_full.yaml`:

```yaml
model:
  hidden_dim: 256              # Node embedding size
  ode_solver: 'dopri5'         # Adaptive Runge-Kutta solver
  memory_capacity: 1000        # Anomaly storage size
  broadcast_hops: 2            # Multi-hop propagation range
  attention_heads: 4           # Attention heads per hop

adversarial:
  pgd_epsilon: 0.1            # Perturbation budget
  pgd_alpha: 0.01             # Attack step size
  pgd_iterations: 5           # Attack steps

continual:
  ewc_lambda: 0.5             # EWC regularization strength

meta:
  meta_lr: 0.001              # Inner loop learning rate
  adversarial_ratio: 0.3      # Fraction of adversarial tasks

training:
  optimizer: 'AdamW'
  learning_rate: 0.001
  weight_decay: 0.01
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10
```

## 📊 Reproducing Paper Results

### Main Results (Table 3)

```bash
# Train all models
bash scripts/train_all_models.sh

# Evaluate all models
bash scripts/evaluate_all_models.sh

# Generate comparison table
python scripts/generate_table3.py --results_dir results/
```

### Ablation Study (Table 5)

```bash
# Run all ablation experiments
bash scripts/run_ablations.sh

# Generate ablation table
python scripts/generate_table5.py --results_dir results/ablations/
```

### Adversarial Robustness (Table 6, Figure 3)

```bash
# Run adversarial evaluation
bash scripts/evaluate_adversarial.sh

# Generate robustness table and figure
python scripts/generate_robustness_results.py --results_dir results/adversarial/
```

### Statistical Validation (Figure 4)

```bash
# Perform statistical tests
python scripts/statistical_validation.py \
    --artemis_results results/artemis/ \
    --baseline_results results/2dynethnet/ \
    --output_dir results/statistics/
```

## 🎯 Using Pre-trained Models

Download pre-trained weights:
```bash
# Download from Google Drive / Zenodo
wget https://zenodo.org/record/XXXXX/artemis_pretrained.pth

# Or use our script
python scripts/download_pretrained.py
```

Inference with pre-trained model:
```bash
python inference.py \
    --checkpoint pretrained/artemis_pretrained.pth \
    --input data/test_transactions.csv \
    --output predictions.csv
```

## 📈 Performance Monitoring

Monitor training with TensorBoard:
```bash
tensorboard --logdir results/tensorboard --port 6006
```

Key metrics logged:
- Training/validation loss
- Recall, Precision, F1-Score, AUC
- False Positive Rate
- Adversarial robustness score
- Component-wise contributions

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size
python train.py --batch_size 16

# Or enable gradient checkpointing
python train.py --gradient_checkpointing
```

**Slow ODE Integration:**
```bash
# Use faster solver
python train.py --ode_solver rk4

# Reduce tolerance
python train.py --ode_rtol 1e-3 --ode_atol 1e-4
```

**Dataset Download Issues:**
```bash
# Manual download from XBlock
# URL: https://xblock.pro/#/dataset/68
# Place files in data/ETGraph/
```

## 📝 Citation

If you use ARTEMIS in your research, please cite our paper:

```bibtex
@article{eyezoo2025artemis,
  title={ARTEMIS: Adversarial-Resistant Temporal Embedding Model for Intelligent Security in Blockchain Fraud Detection},
  author={Eyezo'o, Benjamin Fabien and Xia, Qi and Gao, Jianbin and Xia, Hu and Victor, Kombou and Richard, Befoum Stephane and Ntuala, Grace Mupoyi and Mulenga, Rossini Mukupa and Jonathan, Anto Leoba},
  journal={Information Processing \& Management},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.ipm.2025.XXXXX}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

**Eyezo'o Benjamin Fabien** - *Lead Author* - [fabienebfy@gmail.com](mailto:fabienebfy@gmail.com)

**Qi Xia** - *Corresponding Author* - [xiaqi@uestc.edu.cn](mailto:xiaqi@uestc.edu.cn)

**University of Electronic Science and Technology of China (UESTC)**
- School of Computer Science and Engineering (School of Cyber Security)
- Chengdu, Sichuan 611731, China

**State Key Laboratory of Blockchain and Data Security**
- Zhejiang University
- Hangzhou, Zhejiang 310027, China

## 🔗 Links

- 📄 [Paper](link-to-paper)
- 💾 [Dataset (ETGraph)](https://xblock.pro/#/dataset/68)
- 🎓 [UESTC BlockchainLab](http://www.blockchainlab.cn/)
- 📧 [Contact](mailto:xiaqi@uestc.edu.cn)

## 🙏 Acknowledgments

This work was supported by:
- National Natural Science Foundation of China
- State Key Laboratory of Blockchain and Data Security
- University of Electronic Science and Technology of China (UESTC)

We thank the authors of 2DynEthNet, GrabPhisher, TGN, TGAT, GraphSAGE, and GAT for their open-source implementations that facilitated our comparative evaluation.

## 📊 Updates

- **2025-01**: Initial release with paper publication
- **2025-01**: Pre-trained models available
- **2025-01**: ETGraph dataset preprocessing scripts released

---

**⭐ If you find ARTEMIS useful, please star this repository!**
```

---

## Additional Files to Include

### requirements.txt
```txt
torch>=2.0.1
torch-geometric>=2.3.1
torch-scatter>=2.1.1
torch-sparse>=0.6.17
torch-cluster>=1.6.1
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.13.0
torchdiffeq>=0.2.3
networkx>=3.1
```

### .gitignore
```gitignore
# Data
data/ETGraph/
data/processed/
*.csv
*.pt
*.pth

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# Results
results/
logs/
checkpoints/
tensorboard/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

