from pathlib import Path
import music21 as m21

REST = "R"
HOLD = "H"


def read_midi_to_time_series(
    file: Path | str, rest=REST, hold=HOLD, step=0.25, target_key=m21.pitch.Pitch("C")
) -> list[int | tuple | str, float]:
    """Read midi file to a fixed-step time series representation

    Example: [52, 32, hold, hold, rest, 22, hold, ...] where all events are equidistant in time.

    Args:
        file: path to midi file
        rest: symbol for a rest
        hold: symbol for holding the previous note
        step: Step size for time series (sampling step). Fraction of quarter note. Durations smaller than step are quantized to step. Defaults to 0.25.
        target_key: Key to transpose to. Defaults to C major / A minor.

    Returns:
        list of tokens (midi note numbers, rests, or holds)
    """
    song = m21.converter.parse(file)
    if target_key is not None:
        song = transpose_song(song, target_key=target_key)
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


def read_midi_to_event_sequence(
    file: Path | str, target_key=m21.pitch.Pitch("C")
) -> list[tuple[int, float, float]]:
    """Read midi file to an event sequence in C

    An event is of the form (pitch, duration, offset). The offset is the time interval since the last note ended.
    Durations and offsets are in fractions of a quarter note.

    Args:
        file: Path to midi file
        target_key: Key to transpose to. Defaults to C major / A minor.

    Returns:
        list of events
    """
    song = m21.converter.parse(file)
    if target_key is not None:
        song = transpose_song(song, target_key=target_key)
    instruments = m21.instrument.partitionByInstrument(song).parts
    instrument = instruments[0]  # Use first instrument.
    notes = []
    offset = 0
    for event in instrument.recurse():
        # Note: To have duration in seconds, we would need to track tempo changes.
        duration = event.duration.quarterLength
        if isinstance(event, m21.note.Note):
            pitch = event.pitch.midi
        elif isinstance(event, m21.chord.Chord):
            # note = tuple(n.pitch.midi for n in event.notes) # Save all notes as tuple.
            pitch = event.notes[0].pitch.midi  # Save first note in chord.
        elif isinstance(event, m21.note.Rest):
            offset += duration
            continue
        else:
            continue
        notes.append((pitch, duration, offset))
        offset = 0
    return notes


def transpose_song(song: m21.stream, target_key=m21.pitch.Pitch("C")) -> m21.stream:
    """Transpose a music21 stream to a target key

    Args:
        song: music21 stream
        target_key: Target key. Defaults to C major / A minor.

    Returns:
        transposed song (music21 stream)
    """
    key = song.analyze("key")
    if key.mode == "major":
        return song.transpose(m21.interval.Interval(key.tonic, target_key))
    elif key.mode == "minor":
        return song.transpose(
            m21.interval.Interval(key.tonic, target_key.transpose(-3))
        )


def write_time_series(sequence, file):
    with open(file, "w") as f:
        f.write(" ".join(str(e) for e in sequence))


def write_event_sequence(sequence, file):
    with open(file, "w") as f:
        for item in sequence:
            f.write(",".join(str(e) for e in item))
            f.write("\n")


if __name__ == "__main__":
    from tqdm import tqdm

    representation = "event_sequence"

    path_to_raw = Path("/Users/savv/datasets/maestro-v3.0.0")
    path_to_processed = Path(f"dataset_{representation}")
    path_to_processed.mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(path_to_raw.glob("**/*.midi"))):
        if representation == "time_series":
            sequence = read_midi_to_time_series(file)
            write_time_series(sequence, path_to_processed / (file.name + ".txt"))
        elif representation == "event_sequence":
            sequence = read_midi_to_event_sequence(file)
            write_event_sequence(sequence, path_to_processed / (file.name + ".txt"))
