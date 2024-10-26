import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from torch import optim
from prepare_data import read_event_sequence, read_time_series, encoding
import numpy as np
from models import MelodyLSTM, MelodyLSTMPlus
from torch.utils.tensorboard import SummaryWriter

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train(
    model: MelodyLSTM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 1,
    device=torch.device("cpu"),
    writer: SummaryWriter = None,
    progress: bool = True,
):
    """Train MelodyLSTM model

    Args:
        model: MelodyLSTM model.
        train_loader: _description_
        validation_loader: _description_
        loss_fn: _description_
        optimizer: _description_
        num_epochs: _description_. Defaults to 1.
        progress: _description_. Defaults to True.

    Returns:
        model
    """
    # TODO save intermediate models
    model.train()
    for epoch in range(1, num_epochs + 1):
        if progress:
            print(f"Epoch {epoch}")
        loss_tr = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            writer=writer,
            device=device,
            epoch=epoch,
            progress=progress,
        )

        loss_va = validate_epoch(model, validation_loader, device=device)
        if progress:
            print(f"loss_va = {loss_va:.4f}")
        if writer is not None:
            writer.add_scalar("Loss/Train_epoch", loss_tr, epoch)
            writer.add_scalar("Loss/Validation_epoch", loss_va, epoch)
    return model


def train_epoch(
    model: MelodyLSTM,
    train_loader: DataLoader,
    loss_fn: callable,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter = None,
    device=torch.device("cpu"),
    epoch: int = 1,
    progress: bool = True,
):
    loss_sum = 0
    model.train()
    for i, (inputs, target) in enumerate(train_loader, start=1):
        model.zero_grad()

        # TODO Make it work with batches.
        inputs = torch.tensor(inputs).to(device)
        target = torch.tensor(target).to(device)

        output = model(inputs)
        loss = loss_fn(output[-1].reshape(1, -1), target)
        loss_sum += loss.detach().item()
        loss_avg = loss_sum / i
        if (i % 1000) == 0:
            if progress:
                print(f"i={i}. " f"loss = {loss_avg:.4E}. ")
            if writer is not None:
                writer.add_scalar(
                    "Loss/Train", loss_avg, (epoch - 1) * len(train_loader) + i
                )
        loss.backward()
        optimizer.step()
    return loss_avg


def validate_epoch(
    model: MelodyLSTM,
    validation_loader: DataLoader,
    device=torch.device("cpu"),
):
    model.eval()
    loss_running = 0
    with torch.no_grad():
        for i, (inputs, target) in enumerate(validation_loader, start=1):
            # TODO Make it work with batches.
            inputs = torch.tensor(inputs).to(device)
            target = torch.tensor(target).to(device)
            output = model(inputs)
            loss = loss_fn(output[-1].reshape(1, -1), target)
            loss_running += loss.item()
    return loss_running / i


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
        self,
        path: str | Path,
        sequence_length: int,
        transform: callable = None,
        size: int = None,
    ):
        """Time series dataset of notes

        Args:
            path: Directory of files with space-delimited tokens of midi pitches or a rest symbol or a hold symbol.
            sequence_length: Sequence length to use as input for predicting the next item in the sequence.
            transform: A function applied to both sequence and next_item.
            size: Number of files to use. If None, uses all files.
        """
        self.path = path
        self.files = list(Path(path).glob("*.txt"))
        if size is not None:
            self.files = np.random.choice(self.files, size=size, replace=False)
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
        idx1 = idx + self.sequence_length
        sequence = self.data[idx:idx1]
        next_item = self.data[idx1]
        if self.transform is not None:
            sequence, next_item = self.transform(sequence), self.transform([next_item])
        return sequence, next_item

    def __len__(self):
        return len(self.data) - self.sequence_length


if __name__ == "__main__":
    from datetime import datetime
    from models import config

    learning_rate = 0.01
    num_epochs = 20
    size = 10  # Number of files to use in the data folder.
    # TODO Write params to tensorboard.

    data_name = "time_series"
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_name = f"melodylstm_{data_name}_{timestamp}"

    path_to_models = Path("models")
    path_to_models.mkdir(parents=True, exist_ok=True)

    path_to_dataset_txt = Path(f"data/{data_name}")
    # TODO Include raw data path (maestro) as config param. Current `dataset` param is representation/processed.

    print(model_name)
    if data_name == "event_sequence":
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
        # TODO Change loss to NLLLoss + MSELoss. Can use ignore_index.
        loss_fn = nn.MSELoss()
    elif data_name == "time_series":
        data = TimeSeriesDataset(
            path=path_to_dataset_txt,
            sequence_length=config["sequence_length"],
            transform=lambda seq: [encoding[e] for e in seq],
            size=size,
        )
        seed_split = 42
        pct_tr = 0.8
        generator = torch.Generator().manual_seed(seed_split)
        data_tr, data_va = random_split(data, (pct_tr, 1 - pct_tr), generator=generator)
        # TODO use batch_size=16.
        train_loader = DataLoader(data_tr, batch_size=1, shuffle=True)
        validation_loader = DataLoader(data_va, batch_size=1, shuffle=False)
        print(
            f"Data: {len(data)} sequences from {len(data.files)} files "
            f"(tr+va = {len(data_tr)}+{len(data_va)}). "
            f"Sequence length = {data.sequence_length}. "
            f"Batch size = {train_loader.batch_size}. "
        )

        model = MelodyLSTM(
            num_unique_tokens=config["num_unique_tokens"],
            embedding_size=config["embedding_size"],
            hidden_size=config["hidden_size"],
        ).to(device)
        loss_fn = nn.NLLLoss()  # Input: log probabilities

    print(
        f"Training model... \n"
        f"{str(config)} \n"
        f"learning_rate={learning_rate}, num_epochs={num_epochs}"
    )
    writer = SummaryWriter(f"runs/{model_name}")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        writer=writer,
        num_epochs=num_epochs,
        device=device,
    )

    model_file = path_to_models / f"{model_name}.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "encoding": encoding,
            "timestamp": timestamp,
        },
        model_file,
    )
    print(f"Saved to {model_file}.")
