import os
import torch
import tiktoken
from pathlib import Path
from src.components.architecture import GPTModel
from src.configs.gpt_configs import GPT_CONFIG_90M
from src.dataloader import create_dataloaders
from src.modelFunction.evalAndTrain import model_train_simple


def get_data_folders(data_root="data"):
    """
    Get all data folders. Expects structure with data sources.
    For train/val split with streaming, separate into:
    - data/train/ and data/val/ (recommended)
    OR
    - Uses all data folders and splits files internally
    """
    data_sources = ["webtext", "books", "code", "stackexchange"]

    # Check if separate train/val directories exist
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")

    if os.path.exists(train_root) and os.path.exists(val_root):
        # Separate train/val structure
        print("[*] Found separate train/val directories")
        train_folders = [os.path.join(train_root, src) for src in data_sources
                        if os.path.exists(os.path.join(train_root, src))]
        val_folders = [os.path.join(val_root, src) for src in data_sources
                       if os.path.exists(os.path.join(val_root, src))]
        return train_folders, val_folders

    # Single data directory - split files per source
    print("[*] Using single data directory with per-source split")
    train_folders = []
    val_folders = []

    print("    Scanning data folders...")
    for source in data_sources:
        source_path = os.path.join(data_root, source)
        if os.path.exists(source_path):
            files = sorted(list(Path(source_path).glob("**/*.txt")))
            num_files = len(files)
            print(f"      {source}: {num_files} files")

            if num_files > 0:
                # Split 90/10
                split_idx = max(1, int(num_files * 0.9))
                train_files = files[:split_idx]
                val_files = files[split_idx:]

                if len(train_files) > 0:
                    train_folders.append(source_path)
                if len(val_files) > 0:
                    val_folders.append(source_path)

    print(f"\n[✓] Data folders identified:")
    print(f"    Train sources: {len(train_folders)}")
    print(f"    Val sources: {len(val_folders)}")

    if not train_folders:
        raise ValueError(f"No training data found in {data_root}")

    return train_folders, val_folders


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
    train_folders, val_folders = get_data_folders(data_root="data")

    # Check if we have validation data
    if not val_folders:
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
        data_folder=train_folders
    )

    val_loader = None
    if val_folders:
        val_loader = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            use_streaming=True,
            data_folder=val_folders
        )

    print(f"[✓] Dataloaders ready:")
    print(f"    Train: {len(train_folders)} source(s)")
    print(f"    Val: {len(val_folders)} source(s)" if val_folders else "    Val: NONE (skipping validation)")

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
    print(f"  - Final model: final_model.pth")
    print(f"  - Latest checkpoint: checkpoints/latest.pth")


if __name__ == "__main__":
    main()
    