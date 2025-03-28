# HROM - Hybrid Rotary Optimized Model  
*A Conversational AI Architecture with Enhanced Position Awareness*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)  
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Overview
HROM is a transformer-based language model specifically designed for dialogue systems, combining rotary positional embeddings with optimized architectural choices for efficient conversation processing. The model achieves strong performance while maintaining safety and computational efficiency.

## Key Features

### Core Innovations
- **Rotary Position Encoding**  
  Leverages rotational matrices for dynamic position awareness in attention mechanisms

- **Hybrid Activation**  
  SwiGLU (Sigmoid-Weighted Gated Linear Unit) feed-forward networks

- **Conversation Structure**  
  Special handling for multi-turn dialogues with user/assistant role tokens

- **Safety First**  
  Integrated content filtering and generation safeguards

## Model Architecture

### Structural Overview
| Component                | Specification                          |
|--------------------------|----------------------------------------|
| Layers                   | 6 transformer blocks                   |
| Attention Heads          | 8 per layer                            |
| Hidden Dimension         | 512                                    |
| Feed-Forward Dimension   | 2048                                   |
| Context Window           | 1024 tokens                            |
| Vocabulary Size          | 32,000 BPE tokens                      |

### Key Technical Components
1. **Positional Encoding**  
   Rotary embeddings that preserve relative positional information through vector rotations

2. **Attention Mechanism**  
   Multi-head attention with combined causal/padding masks

3. **Activation Strategy**  
   SwiGLU non-linearity in feed-forward networks

4. **Safety Systems**  
   Real-time content filtering and generation constraints

## Getting Started

### Installation
1. Install PyTorch 2.0+  
2. Install supporting packages: `tokenizers` and `datasets`  
3. Clone repository

### Basic Usage
1. **Initialization**  
   Load pretrained tokenizer and model weights

2. **Text Generation**  
   Process user input through the safety system and generate responses

3. **Training**  
   Configure dataset paths and hyperparameters in training scripts

## Training Configuration

### Optimization Setup
- **Batch Size**: 32 sequences
- **Learning Rate**: 3e-4
- **Epochs**: 50
- **Regularization**:  
  - 0.1 dropout rate  
  - Gradient clipping at 1.0

### Dataset Handling
- Processes multi-turn conversations from DailyDialog
- Supports up to 6 dialogue turns per sample
- Dynamic padding and memory-efficient batching

## Safety Systems

### Content Protection
- Blocklist filtering for harmful phrases
- Generation termination protocol
- Interactive safety checks during response creation

## Performance
- Efficient CUDA utilization via mixed-precision training
- Checkpoint management system for long-running jobs
- Memory-optimized attention masking

## License
Apache License 2.0 - See LICENSE file for details. Commercial use requires prior authorization.
