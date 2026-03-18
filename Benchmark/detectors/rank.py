import torch


def get_rank(text, args, tokenizer, model, log=False):
    with torch.no_grad():
        if text == "":
            return None
        else:
            tokenized = tokenizer(text, return_tensors="pt").to(args.DEVICE)
            logits = model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]

            matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

            assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"

            ranks = ranks.float() + 1
            if log:
                ranks = torch.log(ranks)

            return ranks.float().mean().item()


def get_ranks(texts, args, tokenizer, model, log=True):
    return [get_rank(text, args, tokenizer, model, log=log) for text in texts]
