import random
from collections import defaultdict


def word_count(text: str) -> int:
    return len(text.split())


def sample(
    data,
    text_key,
    n_samples=4000,
    bin_size=20,
    max_words=360,
    seed=2023
):
    """
    Length-stratified sampling without replacement.
    Original dataset indices are preserved as `orig_index`.
    """

    random.seed(seed)

    # build bins: 0-20, 20-40, ..., 340-360, overflow
    bins = list(range(0, max_words, bin_size)) + [10**9]
    buckets = defaultdict(list)

    # assign each item to a bin
    for orig_index, item in enumerate(data):
        wc = word_count(item[text_key])
        for b in bins:
            if wc < b + bin_size:
                rec = dict(item)
                rec["orig_index"] = orig_index
                rec["word_count"] = wc
                rec["len_bin"] = b
                buckets[b].append(rec)
                break

    num_bins = len(buckets)
    base = n_samples // num_bins
    remainder = n_samples % num_bins

    sampled = []
    surplus_bins = []

    # first round: per-bin sampling
    for i, b in enumerate(sorted(buckets.keys())):
        quota = base + (1 if i < remainder else 0)
        pool = buckets[b]

        if len(pool) >= quota:
            picked = random.sample(pool, quota)
            sampled.extend(picked)
            buckets[b] = [x for x in pool if x not in picked]
            if buckets[b]:
                surplus_bins.append(b)
        else:
            sampled.extend(pool)
            buckets[b] = []

    # fill remaining slots
    while len(sampled) < n_samples and surplus_bins:
        b = random.choice(surplus_bins)
        if not buckets[b]:
            surplus_bins.remove(b)
            continue
        sampled.append(buckets[b].pop())
        if not buckets[b]:
            surplus_bins.remove(b)

    if len(sampled) < n_samples:
        raise RuntimeError(
            f"Not enough samples: need {n_samples}, got {len(sampled)}"
        )

    return sampled[:n_samples]
