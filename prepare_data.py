from pathlib import Path
import music21 as m21
import csv
from fractions import Fraction

NUM_PITCHES = 128
STEP_SIZE = 0.25
REST = "R"
HOLD = "H"
END = "E"


def read_midi_to_time_series(
    file: Path | str,
    rest=REST,
    hold=HOLD,
    step=STEP_SIZE,
    target_key=m21.pitch.Pitch("C"),
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
            # TODO Deal with chords differently.
        else:
            # TODO Try using end token.
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
            # TODO Using first/bassiest note creates large pitch leaps.
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


def read_time_series(file: str | Path) -> list[str]:
    """Read txt file containing space-delimited sequences of pitch numbers or rest tokens or hold tokens"""
    with open(file) as f:
        return f.read().split(" ")


def read_event_sequence(file: str | Path) -> list:
    """Read txt file containing newline-delimited sequences of (pitch, duration, offset)"""
    with open(file) as f:
        data = []
        for row in csv.reader(f):
            pitch, duration, offset = row
            if duration.find("/"):
                duration = Fraction(duration)
            if offset.find("/"):
                offset = Fraction(offset)
            data.append((int(pitch), float(duration), float(offset)))
    return data


def make_integer_encoding(
    num_int: int = 0, non_int_tokens: list[str] = tuple()
) -> dict[str, int]:
    """Make integer encoder

    Maps str(int) to int e.g. "22" to 22 up to `num_int` after which it encodes `non_int_tokens`.

    Args:
        num_int: Number of pitches. Defaults to 0.
        non_int_tokens: Number of non-pitch tokens. Defaults to tuple().

    Returns:
        dict where v[token] gives a token's integer encoding.
    """
    encoding = {str(i): i for i in range(num_int)}
    for token in non_int_tokens:
        encoding[token] = len(encoding)
    return encoding


encoding = make_integer_encoding(NUM_PITCHES, non_int_tokens=[REST, HOLD, END])

if __name__ == "__main__":
    from tqdm import tqdm

    representation = "time_series"

    # 3696777 notes in 1276 files
    path_to_raw = Path("/Users/savv/datasets/maestro-v3.0.0")
    path_to_processed = Path(f"data/{representation}")
    path_to_processed.mkdir(parents=True, exist_ok=True)

    if representation == "time_series":
        for file in tqdm(list(path_to_raw.glob("**/*.midi"))):
            sequence = read_midi_to_time_series(file)
            write_time_series(sequence, path_to_processed / (file.name + ".txt"))
    elif representation == "event_sequence":
        for file in tqdm(list(path_to_raw.glob("**/*.midi"))):
            sequence = read_midi_to_event_sequence(file)
            write_event_sequence(sequence, path_to_processed / (file.name + ".txt"))
