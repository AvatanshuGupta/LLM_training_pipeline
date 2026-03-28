import torch
import random
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
from src.instructFT import format_input
from pathlib import Path


class GptDataset(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        self.input_ids=[]
        self.target_ids=[]

        tokens=tokenizer.encode(text,allowed_special={"<|endoftext|>"})

        for i in range(0,len(tokens)-max_length,stride):
            input_id=tokens[i:i+max_length]
            target_id=tokens[i+1:i+max_length+1]

            # Store as tensors directly (faster than creating on every __getitem__)
            self.input_ids.append(torch.tensor(input_id, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_id, dtype=torch.long))

        # Validate dataset not empty
        if len(self.input_ids) == 0:
            raise ValueError(f"Text too short ({len(text):,} chars) for max_length={max_length}. "
                           f"Provide at least {max_length + stride:,} characters.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        # Return pre-created tensors (faster)
        return self.input_ids[index], self.target_ids[index]


class StreamingGptDataset(IterableDataset):
    """Streams data from files without loading everything into RAM."""
    def __init__(self, file_paths, tokenizer, max_length=256, stride=128, chunk_size=32768):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.chunk_size = chunk_size  # Characters to process at once

        # Validate file existence
        for fp in self.file_paths:
            if not Path(fp).exists():
                raise FileNotFoundError(f"File not found: {fp}")

    def _load_txt_files(self):
        for file in self.file_paths:
            yield file

    def _process_file_chunks(self, filepath):
        """Generator that yields tokenized chunks from a file with context preservation."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                buffer = ""
                overlap_size = self.chunk_size // 4  # 25% overlap for context

                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        # Process remaining buffer
                        if buffer:
                            tokens = self.tokenizer.encode(buffer, allowed_special={"<|endoftext|>"})
                            # Yield all valid sequences from final buffer
                            for i in range(0, max(1, len(tokens) - self.max_length + 1), self.stride):
                                if len(tokens[i:i+self.max_length+1]) > self.max_length:
                                    yield tokens[i:i+self.max_length+1]
                        break

                    buffer += chunk + "<|endoftext|>"

                    # Process when buffer is large enough
                    if len(buffer) >= self.chunk_size:
                        # Tokenize the main part, keep overlap
                        split_point = len(buffer) - min(overlap_size, len(buffer) // 2)
                        main_part = buffer[:split_point]
                        overlap_part = buffer[split_point:]

                        tokens = self.tokenizer.encode(
                            buffer + "<|endoftext|>",
                            allowed_special={"<|endoftext|>"}
                        )

                        # Yield sliding windows
                        for i in range(0, max(1, len(tokens) - self.max_length + 1), self.stride):
                            if len(tokens[i:i+self.max_length+1]) > self.max_length:
                                yield tokens[i:i+self.max_length+1]

                        # Prepare buffer for next iteration with overlap
                        buffer = overlap_part

        except Exception as e:
            print(f"[!] Error processing {filepath}: {e}")
            # Log but continue - don't block training

    def __iter__(self):
        files = list(self._load_txt_files())
        random.shuffle(files)

        for filepath in files:
            for tokens in self._process_file_chunks(filepath):
                if len(tokens) > self.max_length:
                    input_ids = torch.tensor(tokens[:self.max_length], dtype=torch.long)
                    target_ids = torch.tensor(tokens[1:self.max_length+1], dtype=torch.long)
                    yield input_ids, target_ids
    


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
    
    