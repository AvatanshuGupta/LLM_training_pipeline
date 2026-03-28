import torch
import tiktoken
import random
from pathlib import Path
from src.components.architecture import GPTModel
from src.configs.gpt_configs import GPT_CONFIG_90M
from src.dataloader import create_dataloaders
from src.modelFunction.evalAndTrain import model_train_simple



def get_train_val_files(data_root="data", split_ratio=0.9):
    all_files = []

    for path in Path(data_root).glob("**/*.txt"):
        all_files.append(str(path))

    if len(all_files) == 0:
        raise ValueError(f"No .txt files found in {data_root}")

    print(f"[*] Total files found: {len(all_files)}")

    random.shuffle(all_files)

    split_idx = int(len(all_files) * split_ratio)

    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"[✓] Train files: {len(train_files)}")
    print(f"[✓] Val files: {len(val_files)}")

    return train_files, val_files


def main():
    # Configuration
    config = GPT_CONFIG_90M

    # Training hyperparameters
    num_epochs = 10
    batch_size = 8
    learning_rate = 5e-4
    eval_freq = 100
    eval_iter = 10
    max_length = 256
    stride = 128

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Using device: {device}")
    print(f"[*] Model: GPT-90M ({config['emb_dim']}D, {config['n_layers']} layers, {config['n_heads']} heads)")

    # Load tokenizer
    print("\n[*] Loading tokenizer (GPT-2)...")
    tokenizer = tiktoken.get_encoding("gpt2")

    # Get data folder paths
    print("\n[*] Identifying data folders...")
    train_files, val_files = get_train_val_files("data")

    # Check if we have validation data
    if not val_files:
        print("\n[!] WARNING: No validation data found!")
        print("    Consider creating data/val/ directory or splitting data manually")

    # Create streaming dataloaders (memory efficient)
    print("\n[*] Creating STREAMING dataloaders...")
    print("    [✓] Data loads on-the-fly from disk (memory efficient)")

    train_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        use_streaming=True,
        file_paths=train_files
    )

    val_loader = None
    if val_files:
        val_loader = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            use_streaming=True,
            file_paths=val_files
        )

    print(f"[✓] Dataloaders ready:")
    print(f"    Train: {len(train_files)} source(s)")
    print(f"    Val: {len(val_files)} source(s)" if val_files else "    Val: NONE (skipping validation)")

    # Initialize model
    print("\n[*] Initializing GPT-90M model...")
    model = GPTModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] Model loaded: {total_params:,} parameters")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    # Training configuration summary
    print("\n[*] Training Configuration:")
    print(f"    Epochs: {num_epochs}")
    print(f"    Batch size: {batch_size}")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Max sequence length: {max_length}")
    print(f"    Eval frequency: every {eval_freq} steps")
    print(f"    Checkpoints saved to: checkpoints/")
    print()

    # Start training
    train_losses, val_losses, track_tokens_seen = model_train_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter
    )

    # Training complete
    print("\n" + "="*50)
    print("[✓] TRAINING COMPLETED!")
    print("="*50)
    if train_losses and val_losses:
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
    if track_tokens_seen:
        print(f"Total tokens processed: {track_tokens_seen[-1]:,}")
    else:
        print("No evaluation checkpoints reached (dataset too small or eval_freq too large)")
    print(f"\nSaved models:")
    print(f"  - Best model: checkpoints/best_model.pth")
    print(f"  - Final model: checkpoints/final_model.pth")
    print(f"  - Latest checkpoint: checkpoints/latest.pth")


if __name__ == "__main__":
    main()
    