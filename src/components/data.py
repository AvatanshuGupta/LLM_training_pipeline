import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.utils import format_input

class GptDataset(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        self.input_ids=[]
        self.target_ids=[]

        tokens=tokenizer.encode(text,allowed_special={"<|endoftext|>"})

        for i in range(0,len(tokens)-max_length,stride):
            input_id=tokens[i:i+max_length]
            target_id=tokens[i+1:i+max_length+1]

            self.input_ids.append(torch.tensor(input_id))
            self.target_ids.append(torch.tensor(target_id))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]
    


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
    
    