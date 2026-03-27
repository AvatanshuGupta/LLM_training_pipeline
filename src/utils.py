import torch


def text_to_token_ids(text,tokenizer):
    tokenized_text=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
    text_tensor=torch.tensor(tokenized_text).unsqueeze(0)
    return text_tensor

def token_ids_to_text(ids,tokenizer):
    ids=ids.squeeze(0)
    return tokenizer.decode(ids.tolist())
