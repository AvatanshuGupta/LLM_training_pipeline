from torch.utils.data import DataLoader
from src.components.data import GptDataset, StreamingGptDataset


def create_dataloaders(text=None, tokenizer=None, batch_size=4, max_length=256, stride=128,
                       shuffle=True, drop_last=True, num_workers=0, use_streaming=False,
                       data_folder=None):
    """
    Create dataloaders with option for streaming or in-memory loading.

    Args:
        text: Text to load (for non-streaming)
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_length: Max sequence length
        stride: Stride for creating samples
        shuffle: Shuffle data
        drop_last: Drop last incomplete batch
        num_workers: Number of workers
        use_streaming: Use streaming dataset (loads from files)
        data_folder: Folder path for streaming datasets
    """
    # Validate required parameters
    if tokenizer is None:
        raise ValueError("tokenizer argument is required")

    if use_streaming:
        if not data_folder:
            raise ValueError("use_streaming=True requires data_folder argument")
        # Streaming dataset (memory efficient)
        dataset = StreamingGptDataset(
            folder_paths=data_folder,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            chunk_size=8192
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=0  # IterableDataset doesn't support num_workers > 0
        )
    else:
        if text is None:
            raise ValueError("use_streaming=False requires text argument")
        # Traditional in-memory dataset
        dataset = GptDataset(text=text, tokenizer=tokenizer, max_length=max_length, stride=stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

    return dataloader