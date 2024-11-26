from generate import time_series_to_midi
import music21 as m21


def test_writing_to_midi():
    def ts_to_midi(s):
        return time_series_to_midi(
            s,
            step_duration=0.25,
            hold_token="H",
            rest_token="R",
            end_token="E",
        )

    def stream2list(stream):
        return [
            (
                e.pitch.midi if isinstance(e, m21.note.Note) else None,
                e.duration.quarterLength,
            )
            for e in stream
        ]

    stream = ts_to_midi(["41", "H", "H", "H", "41", "40", "H", "H", "69", "2", "H"])
    assert stream2list(stream) == [
        (41, 1.0),
        (41, 0.25),
        (40, 0.75),
        (69, 0.25),
        (2, 0.5),
    ]
    stream = ts_to_midi(["41", "H", "H", "R", "41", "40", "H", "H", "69", "2", "H"])
    assert stream2list(stream) == [
        (41, 0.75),
        (None, 0.25),
        (41, 0.25),
        (40, 0.75),
        (69, 0.25),
        (2, 0.5),
    ]
