import torch


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text



def text_to_token_ids(text,tokenizer):
    tokenized_text=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
    text_tensor=torch.tensor(tokenized_text).unsqueeze(0)
    return text_tensor

def token_ids_to_text(ids,tokenizer):
    ids=ids.squeeze(0)
    return tokenizer.decode(ids.tolist())


def generate(model,idx,max_next_tokens,context_size,temperature=0.0,top_k=None,eos_id=None):
    for _ in range(max_next_tokens):
        idx_cond=idx[:,-context_size:]
        with torch.no_grad():
            logits=model(idx_cond)
        logits=logits[:,-1,:]

        if top_k is not None:
            top_logits,top_pos=torch.topk(logits,top_k)
            min_value=top_logits[:,-1]
            logits=torch.where(
                condition=logits < min_value,
                input=torch.tensor(float("-inf")).to(logits.device),
                other=logits
            )

        if temperature > 0.0 :
            logits=logits/temperature
            probas=torch.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probas,num_samples=1)
        
        else :
            idx_next=torch.argmax(logits,dim=-1,keepdim=True)
        
        if eos_id is not None and (idx_next == eos_id).any():
            break

        idx=torch.cat((idx,idx_next),dim=1)
    
    return idx



