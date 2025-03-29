# HROM - Hybrid Rotary Optimized Model  
*A Conversational AI Architecture with Enhanced Position Awareness*

[![PyTorch](https://img.shields.io/badge/PTorch-2.0+-red.svg)](https://pytorch.org)  
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Overview
HROM is a transformer-based language model specifically designed for dialogue systems, combining rotary positional embeddings with optimized architectural choices for efficient conversation processing. The model achieves strong performance while maintaining safety and computational efficiency.

## Key Features

### Core Innovations
- **Rotary Position Encoding**  
  Leverages rotational matrices for dynamic position awareness in attention mechanisms using vector rotations

- **Hybrid Activation**  
  SwiGLU (Sigmoid-Weighted Gated Linear Unit) feed-forward networks with chunk-based processing

- **Conversation Structure**  
  Special handling for multi-turn dialogues with dedicated user/assistant role tokens (`<user>` and `<assistant>`)

- **Safety First**  
  Integrated content filtering with blocklist matching and generation-time safeguards

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
| Special Tokens           | 6 (including role markers)             |

### Key Technical Components
1. **Positional Encoding**  
   Rotary embeddings that preserve relative positional information through vector rotations using learned frequency bands

2. **Attention Mechanism**  
   Combined causal/padding masks with memory-efficient implementation

3. **Activation Strategy**  
   SwiGLU non-linearity with chunked processing in feed-forward networks

4. **Safety Systems**  
   Real-time content filtering with phrase blocklists and generation termination protocols

## Getting Started

### Installation
1. Install PyTorch 2.0+  
2. Install supporting packages: 
   ```bash
   pip install tokenizers datasets
   ```
3. Clone repository

### Basic Usage
1. **Initialization**  
   ```python
   tokenizer = Tokenizer.from_file("tokenizer/hrom_tokenizer.json")
   model = HROM().load_state_dict(torch.load("checkpoints/model.pt"))
   safety = SafetyManager(model, tokenizer)
   ```

2. **Safe Text Generation**  
   ```python
   response = safety.generate_safely("User prompt here")
   ```

3. **Training**  
   Configure dataset paths and hyperparameters in the training script:
   ```bash
   python train.py
   ```

## Training Configuration

### Optimization Setup
- **Batch Size**: 32 sequences
- **Learning Rate**: 1e-4 (reduced for stability)
- **Epochs**: 100 
- **Gradient Accumulation**: 4 steps
- **Regularization**:  
  - 0.1 dropout rate  
  - Gradient clipping at 1.0

### Dataset Handling
- Processes multi-turn conversations from DailyDialog
- Supports up to 6 dialogue turns per sample with role tagging
- Dynamic padding and memory-efficient batching
- Special token integration (`<s>`, `</s>` markers)

## Safety Systems

### Content Protection
- Blocklist filtering with configurable prohibited phrases
- Generation-time toxicity checks
- Interactive safety checks during response creation
- Automatic termination on unsafe content detection

## Performance
- **Mixed Precision Training**: CUDA-optimized via automatic gradient scaling
- **Checkpoint Management**: Automatic rotation of model snapshots
- **Memory Optimization**: Gradient accumulation for larger effective batch sizes
- **Efficient Attention**: Combined causal/padding mask implementation

## Implementation Details

### Novel Components
- **Rotary Position Implementation**  
  `RotaryEmbedding` class with frequency band learning

- **Safety Manager**  
  Real-time generation monitoring with:
  - Phrase blocklists
  - Token-level content scanning
  - Early termination protocol

- **Training Infrastructure**  
  - Automatic mixed precision
  - Gradient accumulation
  - Checkpoint rotation system

## License
Apache License 2.0 - See LICENSE file for details. Commercial use requires prior authorization.

--- 
Key changes from previous version:
- Added gradient accumulation support
- Implemented more sophisticated safety systems
- Reduced learning rate and increased training duration
- Improved tokenizer handling with role-specific tokens
- Added checkpoint management system
- Enhanced dataset processing with turn limitations
- Implemented mixed-precision training infrastructure
```
