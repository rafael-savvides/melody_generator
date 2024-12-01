from pathlib import Path
import music21 as m21
import csv
from fractions import Fraction
from tqdm import tqdm
import json
from config import STEP_SIZE, TOKENS


def prepare_data(dataset: str, path_to_raw: str | Path, path_to_processed: str | Path):
    """Process and save a dataset

    Args:
        dataset: Dataset name.
        path_to_raw: Path from which to read raw data.
        path_to_processed: Path in which to save processed data.
    """
    if dataset == "maestro-v3.0.0-time_series":
        process_midis(path_to_raw, path_to_processed, representation="time_series")
    elif dataset == "jsb_chorales":
        bachs = process_jsb_chorales(path_to_raw)
        write_time_series(bachs, file=path_to_processed)


def process_midis(
    path_to_raw: Path | str,
    path_to_processed: Path | str,
    representation="time_series",
    progress: bool = True,
):
    """Process midi files to txt files

    Read midi files recursively in path_to_raw, and save them as txt files to path_to_processed.

    Args:
        path_to_raw: Directory with midi files.
        path_to_processed: Directory to save processed txt files.
        representation: Either "time_series" or "event_sequence". Defaults to "time_series".
        progress: If True, show a progress bar.
    """
    path_to_raw, path_to_processed = Path(path_to_raw), Path(path_to_processed)
    path_to_processed.mkdir(parents=True, exist_ok=True)
    if representation == "time_series":
        read_midi, write_txt = (
            read_midi_to_time_series,
            lambda seq, file: write_time_series([seq], file),
        )
    elif representation == "event_sequence":
        read_midi, write_txt = read_midi_to_event_sequence, write_event_sequence
    else:
        raise ValueError(f"Unknown representation ({representation}).")
    print(
        f"Reading .midi files recursively in {path_to_raw} and saving .txt files to {path_to_processed}."
    )
    for file in tqdm(list(path_to_raw.glob("**/*.midi")), disable=not progress):
        sequence = read_midi(file)
        write_txt(sequence, path_to_processed / (file.name + ".txt"))


def read_midi_to_time_series(
    file: Path | str,
    rest=TOKENS["rest"],
    hold=TOKENS["hold"],
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
            # TODO Deal with chords differently. Could save tuple, then write to txt with commas: 62 62,67
        else:
            # TODO Try using end token. Could use \n as end token when writing to txt.
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


def read_time_series(
    file: str | Path,
    song_delim: str = "\n",
    beat_delim: str = " ",
    note_delim: str = ",",
) -> list[list[str | tuple]]:
    """Read songs in a time-series representation from a text file

    - The file contains songs. Songs are separated by `song_delim`.
    - A song is a sequence of beats. Beats are separated by `beat_delim`.
    - A beat is a string token: a pitch, a tuple of pitches (separated by `note_delim`) or a non-pitch token (like rest or hold).

    Examples:

    - "62 64 65" -> ["62", "64", "65"]
    - "62 64 65\n40 40,44 50" -> ["62", "64", "65"], ["40", ("40", "44"), "50"]

    Args:
        file: Path to a text file.
        song_delim: Song deliminator.
        beat_delim: Beat deliminator.
        note_delim: Note deliminator.

    Returns:
        songs in a time series format
    """
    with open(file) as f:
        raw: str = f.read()
    return [
        [
            beat if beat.find(note_delim) == -1 else tuple(beat.split(note_delim))
            for beat in song.split(beat_delim)
        ]
        for song in raw.split(song_delim)
        if song != ""
    ]


def write_time_series(
    songs: list[list[str | tuple]],
    file: str | Path,
    song_delim: str = "\n",
    beat_delim: str = " ",
    note_delim: str = ",",
):
    """Write songs in a time-series representation to a text file

    Args:
        songs: list of songs. See read_time_series().
        file: Path to a text file.
        song_delim: Song deliminator.
        beat_delim: Beat deliminator.
        note_delim: Note deliminator.
    """
    with open(file, "w") as f:
        for song in songs:
            beats = [
                note_delim.join(beat) if isinstance(beat, tuple) else str(beat)
                for beat in song
            ]
            f.write(beat_delim.join(beats))
            f.write(song_delim)


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


def write_event_sequence(
    sequence, file, field_delim: str = ",", beat_delim: str = "\n"
):
    with open(file, "w") as f:
        for item in sequence:
            f.write(field_delim.join(str(e) for e in item))
            f.write(beat_delim)


def process_jsb_chorales(
    path_to_raw, rest_token=TOKENS["rest"], end_token=TOKENS["end"]
):
    # TODO May need to transpose to C?
    bachs = read_jsb_chorales(path_to_raw, return_split=False)
    bachs = [
        make_monophonic(song, num_voices=4, end_token=end_token, rest_token=rest_token)
        for song in bachs
    ]
    return bachs


def read_jsb_chorales(
    path, return_split=True
) -> dict[str, list[list[list[int]]]] | list[list[int]]:
    """Read Bach chorales dataset

    Args:
        path: path to jsb-chorales-16th.json
        return_split: Return. Defaults to True.

    Returns:
        If return split is True:
            dict with "train", "valid", "test", each a list of songs.
        else:
            list of songs ("train", "valid", "test" combined)
    """
    with open(path) as f:
        data: dict[str, list[list]] = json.load(f)
    for split, songs in data.items():
        # The raw data has some float values, like 63.0.
        songs = [[list(map(int, beat)) for beat in song] for song in songs]
        data[split] = songs
    if return_split:
        return data
    else:
        return [beat for song in data.values() for beat in song]


def make_monophonic(
    song: list[list[int | str]],
    num_voices=4,
    end_token: str = None,
    rest_token: str = None,
) -> list[int | str]:
    """Flatten a multi-voice song into a note sequence

    Essentially a np.reshape, but with handling of different-sized elements.

    Args:
        song: A song as a list of beats, each a list of up to `num_voices` notes (int or str)
        num_voices: Number of voices. Defaults to 4.
        end_token: If not None, `end_token` is appended at the end of each voice. Defaults to None.
        rest_token: If not None, `rest_token` is appended whenever a voice is missing (i.e. when len(beat) < num_voices). Defaults to None.

    Returns:
        list of notes
    """
    mono = []
    for i in range(num_voices):
        for beat in song:
            try:
                mono.append(beat[i])
            except IndexError:
                if rest_token is not None:
                    mono.append(rest_token)
        if end_token is not None:
            mono.append(end_token)
    return mono


if __name__ == "__main__":
    from config import PATH_TO_DATA, DATASETS

    dataset = "jsb_chorales"

    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset ({dataset}). Allowed values: {list(DATASETS)}."
        )

    prepare_data(
        dataset=dataset,
        path_to_raw=Path(PATH_TO_DATA) / DATASETS[dataset]["raw"],
        path_to_processed=Path(PATH_TO_DATA) / DATASETS[dataset]["processed"],
    )
