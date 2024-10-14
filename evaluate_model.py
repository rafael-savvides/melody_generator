import os
from pathlib import Path
import torch
from models import MelodyLSTM


def generate_melody(model, initial_sequence, num_notes, sequence_length):
    melody = list(initial_sequence)
    for i in range(num_notes):
        inputs = melody[-sequence_length:]
        scores = model(inputs)[-1]
        next_item = torch.argmax(scores).item()
        melody.append(next_item)
    return melody


if __name__ == "__main__":
    from models import config

    MODEL_FILE = os.getenv("MODEL_FILE", None)
    if MODEL_FILE is None:
        MODEL_FILE = list(Path("models").glob("*.*"))[0]

    if MODEL_FILE.name.find("time_series"):
        model = MelodyLSTM(
            num_unique_tokens=config["num_unique_tokens"],
            embedding_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
        )
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
        example_input = [36, 129, 129, 66, 130]
        scores = model(example_input)[-1]

    elif MODEL_FILE.find("event_sequence"):
        pass
        # example_input = torch.tensor(
        #     [[36, 0.25, 0], [24, 0.25, 3.0], [69, 0.25, 0.25], [65, 0.25, 0], [65, 0.25, 0]]
        # )
        # model(example_input)[-1]
