import torch
import torch.nn.functional as F


def get_entropy(text, args, tokenizer, model):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(args.DEVICE)  # input_ids + mask
        logits = model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()
