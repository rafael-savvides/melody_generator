import os
import torch
from models import MelodyLSTM
from pathlib import Path
import numpy as np
import music21 as m21
from config import TOKENS
import argparse
from torch.nn.functional import softmax


def generate_melody(
    model: MelodyLSTM,
    initial_sequence: list[str],
    num_notes: int,
    sequence_length: int,
    temperature: float = 1.0,
    random_seed: int = None,
    allowed_notes: list[int] = None,
) -> list[str]:
    """Generate a melody

    Args:
        model: A MelodyLSTM model.
        initial_sequence: A list of note tokens to start the melody.
        num_notes: Number of tokens to generate.
        sequence_length: The number of tokens to use as context in the model.
        temperature: Temperature parameter for sampling. Defaults to 1.0.
        random_seed: Random seed. Defaults to None.
        allowed_notes: Notes that can be sampled. For example, to not sample notes not seen during training.

    Returns:
        a melody that starts with `initial_sequence` and continues for `num_notes` tokens
    """
    rng = torch.default_generator
    if random_seed is not None:
        rng = rng.manual_seed(random_seed)
    melody = list(initial_sequence)
    for _ in range(num_notes):
        inputs = melody[-sequence_length:]
        output = model(inputs)[-1].detach().ravel()  # log_softmax(.)
        if allowed_notes is not None:
            out = torch.full_like(output, -torch.inf)
            out[allowed_notes] = output[allowed_notes]
            output = out
        next_item = sample_with_temperature(
            output.ravel(), t=temperature, rng=rng, logp=True
        )
        melody.append(next_item)
    return melody


def sample_with_temperature(
    scores: torch.tensor,
    t: float = 1.0,
    rng: torch.Generator = torch.default_generator,
    logp: bool = True,
) -> int:
    """Sample with temperature

    Sample from a discrete probability distribution with some randomness, given by a temperature.

    Args:
        scores: Softmax scores for C classes as a tensor of shape (C,).
        t: Temperature. Defaults to 1.0.
        rng: torch random number Generator.
        logp: If True, scores are log-softmax.

    Returns:
        an integer in [0, C-1]
    """
    logscores = np.log(scores) if not logp else scores
    prob = softmax(logscores / t, dim=0)
    return torch.multinomial(prob, 1, generator=rng).item()


def time_series_to_midi(
    sequence: list[str],
    step_duration: float,
    filename: str | Path = None,
    hold_token=TOKENS["hold"],
    rest_token=TOKENS["rest"],
    end_token=TOKENS["end"],
):
    """Convert a time series melody to midi

    Args:
        sequence: list of strings. A melody as notes or rests or hold tokens at fixed time steps.
        filename: Path to save midi file. Defaults to None.

    Returns:
        music21 stream
    """

    def make_note(pitch, length):
        return (
            m21.note.Rest(quarterLength=length)
            if pitch is None
            else m21.note.Note(midi=pitch, quarterLength=length)
        )

    stream = m21.stream.Stream()
    step = 1
    pitch_prev = None
    for i, e in enumerate(sequence):
        if e == hold_token:
            step += 1
            continue
        if e == end_token:
            break
        if i > 0:
            stream.append(make_note(pitch_prev, step_duration * step))
        step = 1
        pitch_prev = None if e == rest_token else int(e)
    stream.append(make_note(pitch_prev, step_duration * step))

    if filename is not None:
        stream.write(fmt="midi", fp=filename)
    return stream


def load_model(model_file: str | Path, model_class: object) -> tuple[object, dict]:
    model_dict = torch.load(model_file, weights_only=False)
    try:
        hparams, state_dict = model_dict["hparams"], model_dict["model_state_dict"]
    except KeyError:
        # Old version.
        hparams, state_dict = model_dict["config"], model_dict["state_dict"]

    model = model_class(
        output_size=hparams["output_size"],
        embedding_size=hparams["embedding_size"],
        hidden_size=hparams["hidden_size"],
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, hparams


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--initial_sequence",
        type=str,
        default="60",
        help="Initial sequence as a space-delimited sequence of MIDI note numbers and rest/hold tokens. Example: '60 62 64.'. See encoding for non-number tokens.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter. Higher values produce more random outcomes.",
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=200, help="Number of steps to generate."
    )
    parser.add_argument(
        "-d",
        "--step_duration",
        type=float,
        default=0.25,
        help="Duration of each step as a fraction of a quarter note.",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for the generation.",
    )
    return parser


if __name__ == "__main__":
    from datetime import datetime
    from config import PATH_TO_MODELS
    from encoder import encoding, decoding

    parser = make_argparser()
    args = parser.parse_args()
    INITIAL_SEQUENCE = args.initial_sequence
    STEP_DURATION = args.step_duration
    STEPS = args.steps
    TEMPERATURE = args.temperature
    RANDOM_SEED = args.random_seed
    allowed_notes = "36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 E R".split(
        " "
    )

    MODEL_FILE = os.getenv("MODEL_FILE", None)
    if MODEL_FILE is None:
        # Most recently modified .pth file in models/.
        MODEL_FILE = max(
            Path(PATH_TO_MODELS).glob("*.pth"),
            key=lambda f: f.stat().st_mtime,
        )
    model, hparams = load_model(MODEL_FILE, MelodyLSTM)
    print(f"Loaded {MODEL_FILE}.")

    path_to_generated = Path("generated")
    path_to_generated.mkdir(exist_ok=True, parents=True)

    print(
        f"Generating {STEPS} steps of duration {STEP_DURATION}*quarter_note with initial sequence '{INITIAL_SEQUENCE}'"
    )
    melody = generate_melody(
        model=model,
        initial_sequence=[encoding[e] for e in INITIAL_SEQUENCE.split(" ")],
        num_notes=STEPS,
        sequence_length=hparams["sequence_length"],
        temperature=TEMPERATURE,
        random_seed=RANDOM_SEED,
        allowed_notes=[encoding[e] for e in allowed_notes],
    )
    stream = time_series_to_midi(
        [decoding[e] for e in melody], step_duration=STEP_DURATION
    )

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_file = path_to_generated / f"melody_{timestamp}.mid"
    stream.write("midi", output_file)
    print(f"Saved to {output_file}.")
