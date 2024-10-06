from pathlib import Path
import music21 as m21


REST = "R"
HOLD = "H"
START = "S"
END = "E"


def read_midi_to_time_series(
    file: Path | str, rest=REST, hold=HOLD, step=0.25
) -> list[int | tuple | str, float]:
    """Read midi file to a fixed-step time series representation

    Example: [52, 32, hold, hold, rest, 22, hold, ...] where all events are equidistant in time.

    Args:
        file: path to midi file
        rest: symbol for a rest
        hold: symbol for holding the previous note
        step: Step size for time series (sampling step). Fraction of quarter note. Durations smaller than step are quantized to step. Defaults to 0.25.

    Returns:
        list of tokens (midi note numbers, rests, or holds)
    """
    song = m21.converter.parse(file)
    instruments = m21.instrument.partitionByInstrument(song).parts
    instrument = instruments[0]  # Use first instrument.
    notes = []
    for event in instrument.recurse():
        if isinstance(event, m21.note.Note):
            note = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            note = rest
        elif isinstance(event, m21.chord.Chord):
            # note = tuple(n.pitch.midi for n in event.notes) # Save all notes as tuple.
            note = event.notes[0].pitch.midi  # Save first note in chord.
        else:
            continue
        duration = event.duration.quarterLength
        num_steps = max(1, int(duration / step))
        notes.append(note)
        notes.extend([hold] * (num_steps - 1))
    return notes


if __name__ == "__main__":
    from tqdm import tqdm

    path_to_raw = Path("/Users/savv/datasets/maestro-v3.0.0")
    path_to_processed = Path("dataset")
    path_to_processed.mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(path_to_raw.glob("**/*.midi"))):
        sequence = read_midi_to_time_series(file)
        with open(path_to_processed / (file.name + ".txt"), "w") as f:
            f.write(" ".join(str(e) for e in sequence))
