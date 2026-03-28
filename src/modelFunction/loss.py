import torch


def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch,target_batch=input_batch.to(device),target_batch.to(device)
    logits=model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())

def cal_loss_loader(data_loader,model,device,num_batches=None):
    total_loss=0
    batch_count=0

    # Handle both regular Dataset and IterableDataset
    try:
        loader_len = len(data_loader)
        if loader_len == 0:
            return float("nan")
        if num_batches is None:
            num_batches = loader_len
        else:
            num_batches = min(num_batches, loader_len)
    except TypeError:
        # IterableDataset - no len() support
        if num_batches is None:
            num_batches = 10

    for i,(input_batch,target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        loss=calc_loss_batch(input_batch,target_batch,model,device)
        total_loss+=loss.item()
        batch_count+=1

    if batch_count == 0:
        return float("nan")
    return total_loss/batch_count

