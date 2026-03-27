from torch.utils.data import DataLoader
from src.components.data import GptDataset

def create_dataloaders(text,tokenizer,batch_size=4,max_length=256,stride=128,
                       shuffle=True,drop_last=True,num_workers=0):
        dataset=GptDataset(text=text,tokenizer=tokenizer,max_length=max_length,stride=stride)

        dataloader=DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              drop_last=drop_last,
                              num_workers=num_workers)
        
        return dataloader