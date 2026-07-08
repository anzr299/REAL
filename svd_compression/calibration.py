"""Build a calibration batch with the model's OWN tokenizer.

Domains: 'c4' (general text), 'math' (NuminaMath cn_k12/olympiads solutions), 'code' (the-stack-smol).
build_calibration() reproduces the paper Table-1 recipe (c4/math/code ~8/68/24 by tokens) or a single
domain. Calibration drives BOTH the REAP saliency (which experts are weak) and the activation-aware fit
(what activations the low-rank approximation is optimised for), so it should match your deployment domain.
"""
import os

import torch
import torch.nn.functional as functional
from datasets import load_dataset


def print_sequence_stats(batch, tokenizer):
    sequence_lengths = batch["attention_mask"].sum(1).float()
    num_pad_tokens = (batch["input_ids"] == tokenizer.pad_token_id).sum().item()
    print(f'batch {tuple(batch["input_ids"].shape)} seqlen avg={sequence_lengths.mean():.1f} '
          f'min={int(sequence_lengths.min())} max={int(sequence_lengths.max())} pad={num_pad_tokens}', flush=True)


def create_batch(tokenizer, domain, batch_size, sequence_length, seed=42, c4_location=None):
    """Tokenize `batch_size` sequences of length `sequence_length` from one `domain`.

    Copied from https://github.com/SamsungSAILMontreal/ream/blob/main/data/calibration_data.py (function
    `create_batch`; variables renamed and a `c4_location` arg added, logic otherwise unchanged).
    """
    min_sequence_length = int(0.9 * sequence_length)
    if domain == "c4":
        c4_source = c4_location if (c4_location and os.path.exists(c4_location)) else "allenai/c4"
        dataset = (load_dataset(c4_source, "en", split="validation", streaming=True)
                   if c4_source == "allenai/c4" else load_dataset(c4_source, split="validation", streaming=True))
        dataset = dataset.shuffle(seed=seed)
        batch = {"input_ids": [], "attention_mask": []}
        for example in dataset:
            if len(example["text"]) < min_sequence_length:
                continue
            tokenized = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True,
                                  max_length=sequence_length)
            padded_ids = functional.pad(tokenized["input_ids"], (0, sequence_length - tokenized["input_ids"].shape[1]),
                                        value=tokenizer.pad_token_id)
            padded_mask = functional.pad(tokenized["attention_mask"],
                                         (0, sequence_length - tokenized["attention_mask"].shape[1]), value=0)
            batch["input_ids"].append(padded_ids[0])
            batch["attention_mask"].append(padded_mask[0])
            if len(batch["input_ids"]) >= batch_size * 3:
                break
    elif domain == "math":
        dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train")
        dataset = dataset.filter(lambda row: row["source"] in ["cn_k12", "olympiads"]
                                 and len(row["solution"]) > min_sequence_length)
        dataset = dataset.shuffle(seed=seed)
        batch = tokenizer(dataset[:batch_size * 100]["solution"], return_tensors="pt",
                          padding=True, truncation=True, max_length=sequence_length)
    elif domain == "code":
        dataset = load_dataset("bigcode/the-stack-smol", split="train")
        dataset = dataset.filter(lambda row: len(row["content"]) > min_sequence_length)
        dataset = dataset.shuffle(seed=seed)
        batch = tokenizer(dataset[:batch_size * 3]["content"], return_tensors="pt",
                          padding=True, truncation=True, max_length=sequence_length)
    else:
        raise NotImplementedError(domain)

    # drop sequences that are mostly padding
    filtered = {"input_ids": [], "attention_mask": []}
    for index in range(len(batch["input_ids"])):
        if (batch["input_ids"][index] == tokenizer.pad_token_id).sum() > (sequence_length - min_sequence_length):
            continue
        filtered["input_ids"].append(batch["input_ids"][index])
        filtered["attention_mask"].append(batch["attention_mask"][index])
        if len(filtered["input_ids"]) >= batch_size:
            break
    filtered = {key: torch.stack(value, 0) for key, value in filtered.items()}
    print_sequence_stats(filtered, tokenizer)
    return filtered


# Paper Table-1 recipe: per-domain (num_sequences, max_tokens); ~8/68/24 by token count.
PAPER_RECIPE = [("c4", 512, 128), ("math", 1024, 512), ("code", 512, 512)]


def build_calibration(tokenizer, recipe=None, seed=42, single_domain=None, num_sequences=1024, sequence_length=512):
    """Build a calibration batch. Either a single domain (single_domain='math') or a multi-domain recipe
    (default: the paper 8/68/24 c4/math/code mix). Pads all domains to a common sequence length."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if single_domain is not None:
        return create_batch(tokenizer, single_domain, num_sequences, sequence_length, seed=seed)
    recipe = recipe or PAPER_RECIPE
    all_input_ids, all_attention_masks = [], []
    for domain, domain_num_sequences, domain_sequence_length in recipe:
        print(f"[calib] building '{domain}' {domain_num_sequences}x{domain_sequence_length}", flush=True)
        domain_batch = create_batch(tokenizer, domain, domain_num_sequences, domain_sequence_length, seed=seed)
        all_input_ids.append(domain_batch["input_ids"])
        all_attention_masks.append(domain_batch["attention_mask"])
    max_length = max(tensor.shape[1] for tensor in all_input_ids)

    def pad_to_max_length(tensor, pad_value):
        return tensor if tensor.shape[1] == max_length \
            else functional.pad(tensor, (0, max_length - tensor.shape[1]), value=pad_value)

    return {
        "input_ids": torch.cat([pad_to_max_length(ids, tokenizer.pad_token_id) for ids in all_input_ids], 0),
        "attention_mask": torch.cat([pad_to_max_length(mask, 0) for mask in all_attention_masks], 0),
    }
