from config import NUM_PITCHES, TOKENS
import json


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


encoding, decoding = make_encoding(
    [str(i) for i in range(NUM_PITCHES)] + list(TOKENS.values())
)

if __name__ == "__main__":
    from pathlib import Path
    from config import PATH_TO_ENCODING

    path_to_encoding = Path(PATH_TO_ENCODING)
    save_encoding(encoding, path_to_encoding)
