
# Project Title

A brief description of what this project does and who it's for

# LLM Training Pipeline

A comprehensive GPT-style Language Model training pipeline built with PyTorch. This project provides a flexible, memory-efficient framework for training transformer-based language models from scratch or fine-tuning existing architectures.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Key Components & Classes](#key-components--classes)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Format](#data-format)
- [Training](#training)
- [Monitoring & Logging](#monitoring--logging)

---

## 🎯 Overview

This LLM training pipeline is designed to train GPT-style transformer models efficiently. It supports:
- Multiple model sizes (90M, 124M, 355M parameters, etc.)
- Streaming data loading for large datasets
- Mixed precision training with AMP
- Checkpoint management and resume capabilities
- MLflow/DagsHub experiment tracking
- Causal language modeling

The pipeline processes diverse training data including books, code (Python, Java, C++, JavaScript), StackExchange Q&A, and web text.

---

## 🛠️ Tech Stack

**Core Dependencies:**
- **PyTorch**: Deep learning framework for model training and inference
- **tiktoken**: GPT-2 tokenizer from OpenAI
- **MLflow**: Experiment tracking and model registry
- **DagsHub**: Version control and experiment tracking integration
- **NumPy/Pandas**: Data manipulation (optional)

**Python Environment:**
- Python 3.8+
- CUDA-enabled GPU recommended (CPU training supported but slower)
- Virtual environment: `lpvenv/`

```
torch
tiktoken
dagshub
mlflow
```

---

## ✨ Features

- **Multiple Model Sizes**: Predefined configurations for 90M, 124M, 355M, and 1558M parameter models
- **Memory-Efficient Streaming**: Load data on-the-fly from disk without loading entire datasets into memory
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) with `torch.amp` for faster training and lower memory usage
- **Checkpoint Management**: Automatic saving of best model, latest checkpoint, and final model
- **Flexible Data Loading**: Support for multiple data sources and formats
- **Experiment Tracking**: Integrated MLflow and DagsHub for experiment monitoring
- **Learning Rate Scheduling**: Cosine annealing learning rate scheduler
- **Gradient Clipping**: Stable training with gradient norm clipping
- **Causal Language Modeling**: Autoregressive text generation with causal masking

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AvatanshuGupta/LLM_training_pipeline.git
```

### 2. Create and Activate Virtual Environment
```bash
# Using Python venv
python -m venv lpvenv
lpvenv\Scripts\activate  # Windows
source lpvenv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 📁 Project Structure

```
Llm_pipeline/
├── train.py                          # Main training entry point
├── requirements.txt                   # Python dependencies
├── checkpoints/                       # Saved model checkpoints
│   ├── best_model.pth                # Best validation model
│   ├── final_model.pth               # Final trained model
│   └── latest.pth                    # Latest checkpoint
├── data/                             # Training data directory
│   ├── books/                        # Book texts
│   │   ├── books_final.txt
│   │   ├── books_final_part1.txt
│   │   └── books_final_part2.txt
│   ├── code/                         # Sourcecode (multiple languages)
│   │   ├── code_clean_part*.txt      # Python (16 parts)
│   │   ├── cpp_clean_part*.txt       # C++ (8 parts)
│   │   ├── java_clean_part*.txt      # Java (4 parts)
│   │   └── js_clean_part*.txt        # JavaScript (3 parts)
│   ├── stackexchange/                # Q&A data
│   │   ├── stackexchange_clean*.txt  # StackExchange (16 parts)
│   │   └── arxiv_clean.txt           # ArXiv papers
│   └── webtext/                      # Web content
│       └── webtext_clean_part*.txt   # Web text (12 parts)
└── src/                              # Source code
    ├── __init__.py
    ├── train.py                      # [Deprecated, use ../train.py]
    ├── dataloader.py                 # Data loading utilities
    ├── instructFT.py                 # Instruction fine-tuning (future)
    ├── utils.py                      # Utility functions
    ├── components/                   # Model components
    │   ├── __init__.py
    │   ├── architecture.py           # GPTModel class
    │   ├── core.py                   # Transformer building blocks
    │   └── data.py                   # Dataset classes
    ├── configs/                      # Configuration files
    │   └── gpt_configs.py            # Model configurations
    └── modelFunction/                # Training & evaluation
        ├── evalAndTrain.py           # Training loop
        ├── loss.py                   # Loss functions
        └── run.py                    # [Legacy]
```

---

## 🏗️ Architecture

### Transformer Block Diagram

```
Input Tokens
    ↓
[Token Embedding + Positional Embedding]
    ↓
[Dropout]
    ↓
┌─────────────────────────────────────────────────┐
│         TransformerBlock (N layers)              │
│  ┌──────────────────────────────────────────┐   │
│  │  Multi-Head Self-Attention               │   │
│  │  - Causal Masking                        │   │
│  │  - Scaled Dot-Product Attention         │   │
│  └──────────────────────────────────────────┘   │
│                   ↓                              │
│  ┌──────────────────────────────────────────┐   │
│  │  Layer Normalization + Residual          │   │
│  └──────────────────────────────────────────┘   │
│                   ↓                              │
│  ┌──────────────────────────────────────────┐   │
│  │  Feed-Forward Network                    │   │
│  │  - Linear(emb_dim, 4*emb_dim)           │   │
│  │  - GELU Activation                      │   │
│  │  - Linear(4*emb_dim, emb_dim)          │   │
│  └──────────────────────────────────────────┘   │
│                   ↓                              │
│  ┌──────────────────────────────────────────┐   │
│  │  Layer Normalization + Residual          │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
    ↓
[Final Layer Normalization]
    ↓
[Output Linear Head (vocab_size)]
    ↓
Logits (predictions for next token)
```

---

## 🔧 Key Components & Classes

### 1. **GPTModel** (`src/components/architecture.py`)
Main transformer-based language model class.

**Constructor Parameters:**
```python
cfg : dict
    Configuration dictionary with keys:
    - vocab_size: int - Vocabulary size (default: 50257)
    - emb_dim: int - Embedding dimension (e.g., 640, 768)
    - context_length: int - Max sequence length (default: 1024)
    - n_layers: int - Number of transformer blocks (e.g., 10, 12)
    - n_heads: int - Number of attention heads (e.g., 10, 12)
    - drop_rate: float - Dropout probability (e.g., 0.1)
    - qkv_bias: bool - Whether to use bias in QKV projections
```

**Methods:**
- `forward(in_idx)`: Generate logits for input token indices
  - **Input**: `in_idx` (batch_size, seq_length)
  - **Output**: logits (batch_size, seq_length, vocab_size)

---

### 2. **MultiHeadAttention** (`src/components/core.py`)
Implements scaled dot-product attention with causal masking.

**Constructor Parameters:**
```python
d_in : int - Input dimension
d_out : int - Output dimension (must be divisible by num_heads)
context_length : int - Maximum sequence length for masking
dropout : float - Dropout rate
num_heads : int - Number of attention heads
qkv_biasing : bool - Whether to use bias in linear projections
```

**Key Features:**
- Causal masking (prevents attending to future tokens)
- Multi-head attention for capturing different representations
- Scaled dot-product attention mechanism

---

### 3. **TransformerBlock** (`src/components/core.py`)
A complete transformer encoder block with attention + feed-forward.

**Constructor Parameters:**
```python
cfg : dict - Configuration dictionary (same as GPTModel)
```

**Sub-components:**
- Multi-Head Attention layer
- Feed-Forward Network (FFN)
- Layer Normalization
- Residual connections with dropout

---

### 4. **GELU** (`src/components/core.py`)
Gaussian Error Linear Unit activation function (approximation).

**Formula:**
```
GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

---

### 5. **FeedForward** (`src/components/core.py`)
Position-wise feed-forward network.

**Architecture:**
```
Linear(emb_dim → 4*emb_dim) → GELU → Linear(4*emb_dim → emb_dim)
```

---

### 6. **LayerNorm** (`src/components/core.py`)
Layer normalization with learnable scale and shift parameters.

**Formula:**
```
norm_x = (x - mean) / √(var + eps)
output = scale * norm_x + shift
```

---

### 7. **GptDataset** (`src/components/data.py`)
In-memory dataset for small datasets (loads entire text into RAM).

**Constructor Parameters:**
```python
text : str - Raw text data
tokenizer : tiktoken.Encoding - Tokenizer instance
max_length : int - Max token sequence length
stride : int - Step size for creating overlapping sequences
```

---

### 8. **StreamingGptDataset** (`src/components/data.py`)
Memory-efficient streaming dataset (loads data on-the-fly from files).

**Constructor Parameters:**
```python
file_paths : list[str] - List of file paths to read from
tokenizer : tiktoken.Encoding - Tokenizer instance
max_length : int - Max token sequence length
stride : int - Step size for overlapping sequences
chunk_size : int - Size of text chunks to read (default: 8192)
```

**Key Features:**
- Streams data from disk instead of loading everything into memory
- Ideal for large datasets (100+ GB)
- Supports shuffling across files

---

### 9. **Data Loading Functions** (`src/dataloader.py`)

#### `create_dataloaders()`
Factory function to create either streaming or in-memory dataloaders.

**Parameters:**
```python
text : str or None - Raw text (required if use_streaming=False)
tokenizer : tiktoken.Encoding - Tokenizer instance (required)
batch_size : int - Batch size (default: 4)
max_length : int - Max sequence length (default: 256)
stride : int - Stride for overlapping samples (default: 128)
shuffle : bool - Shuffle data (default: True)
drop_last : bool - Drop last incomplete batch (default: True)
num_workers : int - DataLoader workers (default: 0)
use_streaming : bool - Use streaming dataset (default: False)
file_paths : list[str] or None - File paths for streaming (required if use_streaming=True)
chunk_size : int - Chunk size for streaming (default: 8192)
```

**Returns:**
- `DataLoader` - PyTorch DataLoader for training

---

### 10. **Training Functions** (`src/modelFunction/evalAndTrain.py`)

#### `model_train_simple()`
Main training loop with validation and checkpointing.

**Parameters:**
```python
model : nn.Module - GPTModel instance
train_loader : DataLoader - Training data loader
val_loader : DataLoader or None - Validation data loader
optimizer : torch.optim.Optimizer - Optimizer (AdamW)
device : torch.device - Device to train on (cuda or cpu)
num_epochs : int - Number of training epochs
eval_freq : int - Evaluate every N steps
eval_iter : int - Number of iterations for evaluation
```

**Returns:**
```python
train_losses : list[float] - Training loss per checkpoint
val_losses : list[float] - Validation loss per checkpoint
track_tokens_seen : list[int] - Cumulative tokens seen
```

**Features:**
- Mixed precision training (AMP) for GPU efficiency
- Automatic gradient clipping
- Learning rate scheduling (Cosine Annealing)
- Checkpoint saving (best model, latest, final)
- MLflow experiment tracking (if available)

---

#### `evaluate_model()`
Evaluates model performance on train/val sets.

**Parameters:**
```python
model : nn.Module - Model to evaluate
train_loader : DataLoader - Training data
val_loader : DataLoader - Validation data
device : torch.device - Device for evaluation
eval_iter : int - Iterations for evaluation
```

**Returns:**
```python
train_loss : float - Average training loss
val_loss : float - Average validation loss
```

---

### 11. **Loss Functions** (`src/modelFunction/loss.py`)

#### `calc_loss_batch()`
Computes cross-entropy loss for a single batch.

**Returns:**
- `loss` : Tensor - Cross-entropy loss value

#### `cal_loss_loader()`
Computes average loss over entire dataloader.

**Parameters:**
```python
data_loader : DataLoader - Data to evaluate
model : nn.Module - Model
device : torch.device - Device
num_batches : int - Number of batches to evaluate
```

**Returns:**
- `avg_loss` : float - Average loss

---

### 12. **Utility Functions** (`src/utils.py`)

#### `save_checkpoint()`
Saves model checkpoint with optimizer state.

**Saves:**
- Model weights
- Optimizer state
- Training metadata (epoch, step, tokens seen, best loss)

#### `load_checkpoint()`
Loads checkpoint and resumes training from saved state.

**Returns:**
- `global_step` : int - Training step
- `start_epoch` : int - Starting epoch
- `tokens_seen` : int - Tokens processed
- `best_val_loss` : float - Best validation loss

---

## ⚙️ Configuration

### Model Configurations (`src/configs/gpt_configs.py`)

**GPT-90M (Small)**
```python
{
    "vocab_size": 50257,      # GPT-2 vocabulary
    "context_length": 1024,   # Max sequence length
    "emb_dim": 640,           # Hidden dimension
    "n_heads": 10,            # Attention heads
    "n_layers": 10,           # Transformer layers
    "drop_rate": 0.1,         # Dropout
    "qkv_bias": False         # Query-Key-Value bias
}
```

**GPT-124M (Base - GPT-2 Small)**
```python
{
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

**Additional Models Available:**
- GPT-355M (Medium)
- GPT-774M (Large)
- GPT-1558M (XL)

---

## 🚀 Usage

### Quick Start

#### 1. Prepare Data
Place your `.txt` files in subdirectories under `data/`:
```
data/
├── books/
├── code/
├── stackexchange/
└── webtext/
```

#### 2. Run Training
```bash
python train.py
```

### Training Script Overview

The main training script (`train.py`) performs:

```python
# 1. Load configuration
config = GPT_CONFIG_90M

# 2. Initialize model
model = GPTModel(config)

# 3. Create streaming dataloaders
train_loader = create_dataloaders(
    tokenizer=tokenizer,
    batch_size=8,
    max_length=256,
    stride=128,
    use_streaming=True,
    file_paths=train_files
)

# 4. Train model
train_losses, val_losses, tokens_seen = model_train_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=10,
    eval_freq=100,
    eval_iter=10
)
```

### Custom Configuration

Modify `train.py` to customize training:

```python
# Change model size
config = GPT_CONFIG_124M  # or GPT_CONFIG_90M

# Adjust hyperparameters
num_epochs = 10
batch_size = 8
learning_rate = 5e-4
max_length = 256
stride = 128
eval_freq = 100
eval_iter = 10
```

---

## 📊 Data Format

### Expected Data Structure

All training data should be `.txt` files placed in `data/` subdirectories:

**Books:**
- `data/books/books_final.txt`
- `data/books/books_final_part1.txt`, `part2.txt`

**Source Code:**
- `data/code/code_clean_part*.txt` (Python - 16 files)
- `data/code/cpp_clean_part*.txt` (C++ - 8 files)
- `data/code/java_clean_part*.txt` (Java - 4 files)
- `data/code/js_clean_part*.txt` (JavaScript - 3 files)

**Q&A & Papers:**
- `data/stackexchange/stackexchange_clean*.txt` (16 files)
- `data/stackexchange/arxiv_clean.txt`

**Web Text:**
- `data/webtext/webtext_clean_part*.txt` (12 files)

### Data Requirements

- **Encoding**: UTF-8
- **Format**: Plain text (.txt)
- **Size**: No limit (streaming loader handles 100+ GB)
- **Content**: Natural language or code

### Example Adding Custom Data

```bash
# Create custom data directory
mkdir data/custom

# Add your .txt files
cp my_corpus.txt data/custom/

# Training automatically picks up new files
python train.py
```

The `get_train_val_files()` function in `train.py` automatically discovers all `.txt` files and splits them 90/10 for training/validation.

---

## 🎓 Training

### Training Configuration

Default training parameters in `train.py`:

```python
num_epochs = 10              # Full passes through dataset
batch_size = 8               # Samples per gradient update
learning_rate = 5e-4         # Initial learning rate
eval_freq = 100              # Evaluate every 100 steps
eval_iter = 10               # Use 10 iterations for evaluation
max_length = 256             # Max tokens per sequence
stride = 128                 # Overlapping sequences
```

### Training Loop Features

1. **Mixed Precision Training**: Automatically uses `torch.amp.autocast()` on GPU
2. **Gradient Clipping**: Clips gradients with max norm = 1.0
3. **Learning Rate Schedule**: Cosine annealing from `lr` to near 0
4. **Checkpoint Management**:
   - Saves best model based on validation loss
   - Saves latest checkpoint every eval cycle
   - Saves final model at end of training
5. **Resume Training**: Automatically loads latest checkpoint if available

### Device Selection

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- **GPU (CUDA)**: ~8-10x faster training
- **CPU**: Slower but works for experimentation

### Monitoring Training

Training outputs key metrics:

```
[*] Using device: cuda
[*] Model: GPT-90M (640D, 10 layers, 10 heads)
[*] Total files found: 81
[✓] Train files: 72
[✓] Val files: 9
[✓] Model loaded: 87,910,912 parameters
...
[✓] TRAINING COMPLETED!
Final train loss: 3.2541
Final val loss: 3.5123
Total tokens processed: 2,450,000
Saved models:
  - Best model: checkpoints/best_model.pth
  - Final model: checkpoints/final_model.pth
  - Latest checkpoint: checkpoints/latest.pth
```

---

## 📈 Monitoring & Logging

### MLflow/DagsHub Integration

The training script integrates with MLflow for experiment tracking:

```python
dagshub.init(repo_owner="avatanshugupta", repo_name="LLM_training_pipeline", mlflow=True)
mlflow.set_experiment("llm-training")
```

**Tracked Metrics:**
- Training loss per checkpoint
- Validation loss per checkpoint
- Learning rate
- Tokens seen
- Training duration

**View Results:**
1. MLflow Dashboard: `mlflow ui`
2. DagsHub Repository: https://dagshub.com/avatanshugupta/LLM_training_pipeline

### Checkpoint Files

**Location:** `checkpoints/`

- `best_model.pth`: Best validation performance
- `final_model.pth`: Model at training completion
- `latest.pth`: Latest checkpoint (resume point)

**Checkpoint Contents:**
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # For AMP
    'global_step': global_step,
    'epoch': epoch,
    'tokens_seen': tokens_seen,
    'best_val_loss': best_val_loss,
    'config': config
}
```

### Resuming Training

Training automatically resumes from the latest checkpoint:

```bash
python train.py  # Automatically loads checkpoints/latest.pth
```

---

## 🔍 Example Workflow

### 1. Data Preparation
```bash
# Organize data in data/ directory
data/
├── books/
├── code/
├── stackexchange/
└── webtext/
```

### 2. Configure Training
```python
# Edit train.py
config = GPT_CONFIG_90M
num_epochs = 10
batch_size = 8
learning_rate = 5e-4
```

### 3. Start Training
```bash
python train.py
```

### 4. Monitor Progress
```bash
# In another terminal
mlflow ui
# Visit http://localhost:5000
```

### 5. Load Checkpoint
```python
from src.components.architecture import GPTModel
from src.configs.gpt_configs import GPT_CONFIG_90M

model = GPTModel(GPT_CONFIG_90M)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---



## 👤 Author

Avatanshu Gupta

## 🤝 Contributing

Contributions welcome! Please follow existing code style and add tests for new features.

---

## ❓ FAQ

**Q: How do I use only GPU training?**
A: No changes needed—the script automatically detects CUDA and uses GPU if available.

**Q: Can I train on CPU?**
A: Yes, but it will be significantly slower (5-10x). GPU training is recommended.

**Q: What's the maximum model size I can train?**
A: Depends on your GPU memory. GPT-124M requires ~7-8 GB VRAM per batch.

**Q: How do I add more training data?**
A: Simply add `.txt` files to `data/` subdirectories. The script automatically discovers them.

**Q: Can I resume from a checkpoint?**
A: Yes, automatically! The script loads from `checkpoints/latest.pth` if it exists.

---
