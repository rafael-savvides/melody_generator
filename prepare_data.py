from pathlib import Path
import music21 as m21


REST = "R"
HOLD = "H"
START = "S"
END = "E"


def read_midi_to_events(file: Path | str, rest=REST) -> list[int | tuple | str, float]:
    """Get note list from midi file

    Args:
        file: path to midi file

    Returns:
        list of (note, duration) tuples where:

        - note: midi note number, tuple of midi note numbers, or rest.
        - duration: fraction of quarter note
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
        notes.append((note, event.duration.quarterLength))
    return notes


def encode_song(note_list: list[int | str], step=0.25) -> str:
    """Encode a list of (note, duration) events to a time series representation

    Args:
        note_list: list of (note, duration) events
        step: Step size for time series (sampling step). Fraction of quarter note. Defaults to 0.25.

    Returns:
        string of events as a time series
    """
    encoded = [START]
    for note, duration in note_list:
        num_steps = max(1, int(duration / step))
        encoded.append(note)
        encoded.extend([HOLD] * (num_steps - 1))
    encoded.append(END)
    return " ".join(str(n) for n in encoded)


if __name__ == "__main__":
    from tqdm import tqdm

    path_to_raw = Path("/Users/savv/datasets/maestro-v3.0.0")
    path_to_processed = Path("dataset")
    path_to_processed.mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(path_to_raw.glob("**/*.midi"))):
        event_list = read_midi_to_events(file)
        encoded_song = encode_song(event_list)
        with open(path_to_processed / (file.name + ".txt"), "w") as f:
            f.write(encoded_song)
