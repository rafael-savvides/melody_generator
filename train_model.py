import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch import optim
from prepare_data import read_event_sequence, read_time_series
import numpy as np
from models import MelodyLSTM, MelodyLSTMPlus


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


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self, path: str | Path, sequence_length: int, transform: callable = None
    ):
        """Time series dataset of notes

        Args:
            path: Directory of files with space-delimited tokens of midi pitches or a rest symbol or a hold symbol.
            sequence_length: Sequence length to use as input for predicting the next item in the sequence.
            transform: A function applied to both sequence and next_item.
        """
        self.path = path
        self.files = Path(path).glob("*.txt")
        self.sequence_length = sequence_length
        self.transform = transform
        self.data = []
        for file in self.files:
            self.data.extend(read_time_series(file))

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
    from models import config

    learning_rate = 0.1

    dataset = "time_series"
    path_to_dataset_txt = Path(f"data/{dataset}")
    if dataset == "event_sequence":
        data = EventSequenceDataset(
            path=path_to_dataset_txt,
            sequence_length=config["sequence_length"],
            transform=scale_pitch,
        )
        data_loader = DataLoader(data, shuffle=True)

        model = MelodyLSTMPlus(
            pitch_range=config["num_unique_tokens"] - 2,
            embedding_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
        )
        loss_fn = nn.MSELoss()  # TODO Change loss to NLLLoss + MSELoss.
    elif dataset == "time_series":
        data = TimeSeriesDataset(
            path=path_to_dataset_txt,
            sequence_length=config["sequence_length"],
        )
        data_loader = DataLoader(data, shuffle=True)

        model = MelodyLSTM(
            num_unique_tokens=config["num_unique_tokens"],
            embedding_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
        )
        loss_fn = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_model(
        model=model,
        train_loader=data_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=1,
    )

    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    model_file = f"model_{dataset}_{timestamp}.pth"
    torch.save(model.state_dict(), model_file)  # TODO Save with config?