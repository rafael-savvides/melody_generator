import os
import torch
from models import MelodyLSTM
from pathlib import Path
from prepare_data import encoding
import numpy as np
import music21 as m21
from prepare_data import HOLD, REST

decoding = {v: k for k, v in encoding.items()}


def generate_melody(
    model: MelodyLSTM,
    initial_sequence: list[str],
    num_notes: int,
    sequence_length: int,
    temperature: float = 1.0,
    random_seed: int = None,
) -> list[str]:
    """Generate a melody

    Args:
        model: A MelodyLSTM model.
        initial_sequence: A list of note tokens to start the melody.
        num_notes: Number of tokens to generate.
        sequence_length: The number of tokens to use as context in the model.
        temperature: Temperature parameter for sampling. Defaults to 1.0.
        random_seed: Random seed. Defaults to None.

    Returns:
        a melody that starts with `initial_sequence` and continues for `num_notes` tokens
    """
    np.random.seed(random_seed)
    melody = list(initial_sequence)
    for i in range(num_notes):
        inputs = melody[-sequence_length:]
        scores = np.exp(model(inputs)[-1].detach().numpy())  # exp(log(softmax(.)))
        next_item = sample_with_temperature(scores.ravel(), t=temperature)
        melody.append(next_item)
    return melody


def sample_with_temperature(scores: np.ndarray, t: float = 1.0) -> int:
    """Sample with temperature

    Sample from a discrete probability distribution with some randomness, given by a temperature.

    Args:
        scores: Scores (like softmax) for C classes as an array of shape (C,). Scores approximate a discrete probability distribution over C classes.
        t: Temperature. Defaults to 1.0.

    Returns:
        an integer in [0, C-1]
    """
    prob = scores ** (1.0 / t)
    prob = prob / sum(prob)  # TODO Maybe make more numerically stable, logsumexp.
    return np.random.choice(range(len(scores)), p=prob).item()


def time_series_to_midi(
    sequence: list[str],
    step_duration: float,
    filename: str | Path = None,
    hold_token=HOLD,
    rest_token=REST,
):
    """Convert a time series melody to midi

    Args:
        sequence: list of strings. A melody as notes or rests or hold tokens at fixed time steps.
        filename: Path to save midi file. Defaults to None.

    Returns:
        music21 stream
    """
    stream = m21.stream.Stream()

    step = 1
    for e in sequence:
        if e == hold_token:
            step += 1
        else:
            length = step_duration * step
            if e == rest_token:
                note = m21.note.Rest(quarterLength=length)
            else:
                note = m21.note.Note(pitch=int(e), quarterLength=length)
            stream.append(note)
            step = 1

    if filename is not None:
        stream.write(fmt="midi", fp=filename)
    return stream


def load_model(model_file) -> tuple[MelodyLSTM, dict]:
    # TODO Should this be a class method in MelodyLSTM?
    model_dict = torch.load(model_file, weights_only=False)
    config, state_dict = model_dict["config"], model_dict["state_dict"]

    model = MelodyLSTM(
        num_unique_tokens=config["num_unique_tokens"],
        embedding_size=config["embedding_size"],
        hidden_size=config["hidden_size"],
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


if __name__ == "__main__":
    from datetime import datetime

    # TODO Add cmd args

    MODEL_FILE = os.getenv("MODEL_FILE", None)
    if MODEL_FILE is None:
        # Most recently modified file in models/.
        MODEL_FILE = max(Path("models").glob("*.*"), key=lambda f: f.stat().st_mtime)
    model, config = load_model(MODEL_FILE)
    print(f"Loaded {MODEL_FILE}.")

    path_to_generated = Path("generated")
    path_to_generated.mkdir(exist_ok=True, parents=True)

    STEP_DURATION = 0.25
    NUM_STEPS = 300
    TEMPERATURE = 1
    # TODO Check why in midi this sequence is incorrect.
    sequence = ["41", "H", "H", "H", "41", "40", "H", "H"]
    print(
        f"Generating {NUM_STEPS} steps of duration {STEP_DURATION}*quarter_note with initial sequence '{' '.join(sequence)}'"
    )
    melody = generate_melody(
        model=model,
        initial_sequence=[encoding[e] for e in sequence],
        num_notes=NUM_STEPS,
        sequence_length=config["sequence_length"],
        temperature=TEMPERATURE,
        random_seed=None,
    )
    stream = time_series_to_midi(
        [decoding[e] for e in melody], step_duration=STEP_DURATION
    )
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    output_file = path_to_generated / f"melody_{timestamp}.mid"
    stream.write("midi", output_file)
    print(f"Saved to {output_file}.")