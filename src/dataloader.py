from torch.utils.data import DataLoader
from src.components.data import GptDataset, StreamingGptDataset


def create_dataloaders(
    text=None,
    tokenizer=None,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    use_streaming=False,
    file_paths=None,   # renamed
    chunk_size=8192    # configurable
):
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
        file_paths: List of file paths for streaming dataset
        chunk_size: Size of text chunks to read
    """

    # Validate tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer argument is required")

    # ========================
    # STREAMING MODE
    # ========================
    if use_streaming:

        # Validate file_paths
        if not file_paths:
            raise ValueError("use_streaming=True requires file_paths (list of files)")

        if not isinstance(file_paths, (list, tuple)):
            raise TypeError("file_paths must be a list of file paths")

        # Create streaming dataset
        dataset = StreamingGptDataset(
            file_paths=file_paths,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
            chunk_size=chunk_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=0  # ⚠️ keep 0 for IterableDataset
        )

    # ========================
    # IN-MEMORY MODE
    # ========================
    else:
        if text is None:
            raise ValueError("use_streaming=False requires text argument")

        dataset = GptDataset(
            text=text,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

    return dataloader