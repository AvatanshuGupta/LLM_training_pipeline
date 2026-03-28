import torch
import os
from datetime import datetime


def text_to_token_ids(text,tokenizer):
    tokenized_text=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
    text_tensor=torch.tensor(tokenized_text).unsqueeze(0)
    return text_tensor

def token_ids_to_text(ids,tokenizer):
    ids=ids.squeeze(0)
    return tokenizer.decode(ids.tolist())


CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, step, epoch, tokens_seen, best_val_loss):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"ckpt_step{step}_epoch{epoch}_{timestamp}.pth"
    filepath = os.path.join(CHECKPOINT_DIR, filename)

    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss
    }

    # Save timestamped checkpoint
    torch.save(checkpoint_data, filepath)

    # Save latest checkpoint (overwrite)
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pth")
    torch.save(checkpoint_data, latest_path)

    cleanup_checkpoints()

    print(f" Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer):
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pth")

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resumed from step {checkpoint['step']}")
        return (
            checkpoint["step"],
            checkpoint["epoch"],
            checkpoint.get("tokens_seen", 0),
            checkpoint.get("best_val_loss", float("inf"))
        )

    return 0, 0, 0, float("inf")

def cleanup_checkpoints(max_keep=5):
    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_") and f.endswith(".pth")],
        key=lambda x: os.path.getctime(os.path.join(CHECKPOINT_DIR, x)),
        reverse=True
    )

    for f in files[max_keep:]:
        try:
            os.remove(os.path.join(CHECKPOINT_DIR, f))
        except Exception as e:
            print(f"Could not delete {f}: {e}")