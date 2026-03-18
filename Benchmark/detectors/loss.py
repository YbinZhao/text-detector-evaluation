import logging
import torch

def get_ll(text, args, tokenizer, model):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(args.DEVICE)
        labels = tokenized['input_ids']
        if labels.nelement() == 0:
            logging.error(f"Empty input: {text}")
            return 0
        else:
            return -model(**tokenized, labels=labels).loss.item()


def get_lls(texts, args, tokenizer, model):
    return [get_ll(text, args, tokenizer, model) for text in texts]