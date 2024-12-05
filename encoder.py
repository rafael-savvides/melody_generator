import json
from config import DATASETS


def get_encoding(dataset):
    if not dataset in DATASETS:
        raise NotImplementedError(f"Unknown dataset={dataset}.")
    return make_encoding(DATASETS[dataset]["notes"])


def make_encoding(tokens: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Make encoder and decoder

    Args:
        tokens: Iterable of tokens.

    Returns:
        dict where v[token] gives a token's integer encoding.
        dict where v[token_encoded] gives a token's decoding.
    """
    encoding = {token: i for i, token in enumerate(tokens)}
    decoding = {v: k for k, v in encoding.items()}
    return encoding, decoding


def save_encoding(encoding, file):
    """Save encoding dictionary to json"""
    with open(file, "w") as f:
        json.dump(encoding, f)


def load_encoding(file):
    """Load encoding and decoding from json"""
    with open(file) as f:
        encoding = json.load(f)
    decoding = {v: k for k, v in encoding.items()}
    return encoding, decoding


if __name__ == "__main__":
    from pathlib import Path

    for dataset in DATASETS.keys():
        encoding, decoding = get_encoding(dataset)
        save_encoding(encoding, Path(DATASETS["encoding_path"]))
