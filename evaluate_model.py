import os
import torch
from train_model import MelodyLSTM

if __name__ == "__main__":
    from train_model import config

    MODEL_FILE = os.getenv("MODEL_FILE", None)
    if MODEL_FILE is None:
        MODEL_FILE = "model_2024-10-14T12:17:53_event_sequence.pth"
    model = MelodyLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
    )
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    example_input = torch.tensor(
        [[36, 0.25, 0], [24, 0.25, 3.0], [69, 0.25, 0.25], [65, 0.25, 0], [65, 0.25, 0]]
    )
    model(example_input)[-1]
