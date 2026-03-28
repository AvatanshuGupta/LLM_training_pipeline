import torch
from .loss import cal_loss_loader,calc_loss_batch
from .run import generate_and_print_sample
import mlflow
import dagshub
from src.utils import save_checkpoint,load_checkpoint
from torch.amp import autocast, GradScaler
import os


# Initialize DagsHub tracking with error handling
MLFLOW_AVAILABLE = False
try:
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "avatanshugupta")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "LLM_training_pipeline")
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment("llm-training")
    MLFLOW_AVAILABLE = True
    print("[✓] MLflow/DagsHub initialized successfully")
except Exception as e:
    print(f"[!] Warning: MLflow/DagsHub unavailable: {e}")
    print("    Training will continue without experiment tracking")


def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    with torch.no_grad():
        train_loss = cal_loss_loader(train_loader, model, device, eval_iter)
        # Handle missing validation loader
        if val_loader is None:
            val_loss = float("nan")
        else:
            val_loss = cal_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss
    

def model_train_simple(model,train_loader,val_loader,optimizer,device,
                       num_epochs,eval_freq,eval_iter):

    scaler = GradScaler(enabled=(device.type == "cuda"))

    train_losses,val_losses,track_tokens_seen=[],[],[]
    tokens_seen,global_step=0,-1
    best_val_loss = float("inf")

    # Load checkpoint
    global_step, start_epoch, tokens_seen, best_val_loss = load_checkpoint(model, optimizer)
    model.to(device)

    mlflow_context = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow_context = mlflow.start_run()
            mlflow_context.__enter__()
            mlflow.log_param("epochs", num_epochs)
            mlflow.log_param("eval_freq", eval_freq)
        except Exception as e:
            print(f"[!] MLflow run failed: {e}")
            MLFLOW_AVAILABLE = False

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()

            for input_batch,target_batch in train_loader:
                optimizer.zero_grad()

                with autocast(device_type=device.type):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)

                if torch.isnan(loss):
                    print("NaN detected, skipping step")
                    continue

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                tokens_seen+=input_batch.numel()
                global_step+=1

                # MLflow logging
                if MLFLOW_AVAILABLE and global_step % 50 == 0:
                    try:
                        mlflow.log_metric("train_loss_step", loss.item(), step=global_step)
                    except Exception as e:
                        print(f"[!] MLflow metric logging failed: {e}")

                if global_step%eval_freq==0:
                    train_loss,val_loss=evaluate_model(
                        model,train_loader,val_loader,device,eval_iter
                    )

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # Save best model (skip if val_loss is NaN)
                    is_best = val_loss < best_val_loss and not (val_loss != val_loss)
                    if is_best:
                        best_val_loss = val_loss
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save(model.state_dict(), os.path.join("checkpoints", "best_model.pth"))
                        if MLFLOW_AVAILABLE:
                            try:
                                mlflow.log_artifact(os.path.join("checkpoints", "best_model.pth"))
                            except Exception as e:
                                print(f"[!] MLflow artifact logging failed: {e}")

                    # Log metrics to MLflow
                    if MLFLOW_AVAILABLE:
                        try:
                            mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=global_step)
                            mlflow.log_metric("train_loss", train_loss, step=global_step)
                            mlflow.log_metric("val_loss", val_loss, step=global_step)
                        except Exception as e:
                            print(f"[!] MLflow metrics failed: {e}")

                    # Save checkpoint
                    save_checkpoint(model, optimizer, global_step, epoch, tokens_seen, best_val_loss)

        # Save final model
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("checkpoints", "final_model.pth"))
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(os.path.join("checkpoints", "final_model.pth"))
            except Exception as e:
                print(f"[!] MLflow final artifact logging failed: {e}")

    finally:
        if mlflow_context is not None:
            try:
                mlflow_context.__exit__(None, None, None)
            except Exception as e:
                print(f"[!] MLflow context exit failed: {e}")

    return train_losses,val_losses,track_tokens_seen