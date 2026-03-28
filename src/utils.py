import torch
import os


def text_to_token_ids(text,tokenizer):
    tokenized_text=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
    text_tensor=torch.tensor(tokenized_text).unsqueeze(0)
    return text_tensor

def token_ids_to_text(ids,tokenizer):
    ids=ids.squeeze(0)
    return tokenizer.decode(ids.tolist())



CHECKPOINT_PATH = "checkpoint.pth"

def save_checkpoint(model, optimizer, step, epoch, tokens_seen, best_val_loss):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss
    }, CHECKPOINT_PATH)


def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
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