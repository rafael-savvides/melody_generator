import os
from pathlib import Path
import torch
from models import MelodyLSTM


@torch.no_grad()
def generate_melody(model, initial_sequence, num_notes, sequence_length):
    melody = list(initial_sequence)
    for i in range(num_notes):
        inputs = melody[-sequence_length:]
        scores = model(inputs)[-1]
        next_item = torch.argmax(scores).item()
        melody.append(next_item)
    return melody


if __name__ == "__main__":
    MODEL_FILE = os.getenv("MODEL_FILE", None)
    if MODEL_FILE is None:
        # Most recently modified file in models/.
        MODEL_FILE = max(Path("models").glob("*.*"), key=lambda f: f.stat().st_mtime)

    model_dict = torch.load(MODEL_FILE)  # state_dict and config
    config, state_dict = model_dict["config"], model_dict["state_dict"]
    if MODEL_FILE.name.find("time_series"):
        model = MelodyLSTM(
            output_size=config["output_size"],
            embedding_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
        )
        model.load_state_dict(state_dict)
        model.eval()
        example_input = [36, 129, 129, 66, 130]
        scores = model(example_input)[-1]

    elif MODEL_FILE.find("event_sequence"):
        pass
        # example_input = torch.tensor(
        #     [[36, 0.25, 0], [24, 0.25, 3.0], [69, 0.25, 0.25], [65, 0.25, 0], [65, 0.25, 0]]
        # )
        # model(example_input)[-1]
