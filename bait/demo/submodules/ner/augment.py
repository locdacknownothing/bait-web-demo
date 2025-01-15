import numpy as np
import pandas as pd
from adept_augmentations import EntitySwapAugmenter
from datasets import Dataset, concatenate_datasets


def augment(
    conll_dataset: Dataset, num_augmentations: int = 4, keep_original: bool = False
) -> Dataset:
    augmenter = EntitySwapAugmenter(conll_dataset)
    augmented_dataset = augmenter.augment(N=num_augmentations)

    if keep_original:
        augmented_dataset = concatenate_datasets([conll_dataset, augmented_dataset])

    augmented_dataset = deduplicate(augmented_dataset)
    return augmented_dataset


def deduplicate(dataset: Dataset) -> Dataset:
    dataset = dataset.map(get_hash)
    uniques = set(dataset.unique("hash"))
    dataset_filter = dataset.filter(check_uniques, fn_kwargs={"uniques": uniques})
    return dataset_filter


def get_hash(sample):
    """Get hash of tokens field."""
    tokens = sample["tokens"]
    if not isinstance(tokens, tuple):
        tokens = tuple(tokens)

    return {"hash": hash(tokens)}  # can use any hashing function here


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False
