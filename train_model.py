import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch import optim
from prepare_data import read_event_sequence
import numpy as np

# TODO Move to json/yaml file?
config = {
    "input_size": 3,
    "hidden_size": 64,
    "output_size": 3,
    "sequence_length": 16,
}


# TODO Move model definition to own file?
class MelodyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = 1

        # (seq_len, input_size) -> (seq_len, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size)
        # (seq_len, hidden_size) -> (seq_len, output_size)
        self.fc = nn.Linear(hidden_size, output_size)
        # Uses `seq_len * hidden_size` features to predict `output_size` targets.

    def forward(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        """Forward pass

        Args:
            x: torch tensor of shape (seq_len, input_size) and dtype float32.
            If x is a list or numpy array it is cast into a torch tensor.

        Returns:
            torch tensor of shape (seq_len, output_size)
        """
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = torch.tensor(np.atleast_2d(x), dtype=torch.float32)
        seq_len = len(x)
        # lstm's default h0 (hidden state) and c0 (cell state) are zeroes.
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out.view(seq_len, -1))
        # TODO One-hot encode pitches. Or use nn.Embedding?
        # TODO Enforce positive pitch, duration and offset (e.g. exp or softplus?)
        return out


def train_model(
    model: MelodyLSTM,
    train_loader,
    loss_fn,
    optimizer: torch.optim.optimizer.Optimizer,
    num_epochs: int = 10,
    progress=True,
):
    """Train model

    Args:
        model: _description_
        train_loader: _description_
        loss_fn: _description_
        optimizer: _description_
        num_epochs: _description_. Defaults to 10.
        progress: _description_. Defaults to True.

    Returns:
        model
    """
    model.train()
    i = 0
    for epoch in range(num_epochs):
        if progress:
            print(f"Epoch {epoch}")
        for inputs, target in train_loader:
            if progress and (i % 1000) == 0:
                print(f"i={i}")

            model.zero_grad()

            inputs = torch.tensor(inputs, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            output = model(inputs)
            loss = loss_fn(output[-1], target)
            loss.backward()
            optimizer.step()
            i = i + 1
            # if i > 10000:
            #     break
    return model


def scale_pitch(x: list | np.ndarray, pitch_range=128) -> np.ndarray:
    """Scale pitch

    Args:
        x: List of (pitch, duration, offset) tuples.
        pitch_range: Maximum pitch. Defaults to 128.

    Returns:
        x with its `pitch` elements divided by `pitch_range`.
        x is cast into a 2D numpy array.
        If x is a single (pitch, duration, offset) tuple, it becomes a (1, 3) array.
    """
    return np.atleast_2d(x) / np.array([pitch_range, 1.0, 1.0])


def generate_melody(model, initial_sequence, num_notes, sequence_length):
    melody = initial_sequence
    for i in range(num_notes):
        inputs = melody[-sequence_length:]
        next_item = model(inputs)[-1]
        melody.append(next_item)
    return melody


class EventSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self, path: str | Path, sequence_length: int, transform: callable = None
    ):
        """Event sequence dataset

        Args:
            path: Directory of files with comma-delimited tuples of midi pitch,duration,offset e.g. 36,0.25,0. Each line is an event.
            sequence_length: Sequence length to use as input for predicting the next item in the sequence.
            transform: A function applied to both sequence and next_item.
        """
        self.path = path
        self.files = Path(path).glob("*.txt")
        self.sequence_length = sequence_length
        self.transform = transform
        self.data = []
        for file in self.files:
            self.data.extend(read_event_sequence(file))
        # Alternative to reading in memory: read files and flatten to create a list of (file_idx, seq_idx). Then index this list in __getitem__.
        # TODO Should EOF be considered? Now the ending of one file predicts the beginning of another. Shuffling between epochs helps a bit.

    def __getitem__(self, idx):
        """Get item

        Args:
            idx: Index.

        Returns:
            sequence: list of `self.sequence_length` tokens
            next_item: next token to predict
        """
        sequence = self.data[idx : idx + self.sequence_length]
        next_item = self.data[idx + self.sequence_length]
        if self.transform is not None:
            sequence, next_item = self.transform(sequence), self.transform(next_item)
        return sequence, next_item

    def __len__(self):
        return len(self.data) - self.sequence_length


if __name__ == "__main__":
    from datetime import datetime

    learning_rate = 0.1

    dataset = "event_sequence"  # 3696777 notes in 1276 files
    path_to_dataset_txt = Path(f"data/{dataset}")
    data = EventSequenceDataset(
        path=path_to_dataset_txt,
        sequence_length=config["sequence_length"],
        transform=scale_pitch,
    )
    data_loader = DataLoader(data, shuffle=True)

    model = MelodyLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
    )
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_model(
        model=model,
        train_loader=data_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=1,
    )

    model_file = f"model_{datetime.now().isoformat(timespec='seconds')}_{dataset}.pth"
    torch.save(model.state_dict(), model_file)  # TODO Save with config?
