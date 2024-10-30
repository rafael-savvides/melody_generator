import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from torch import optim
from prepare_data import read_event_sequence, read_time_series, encoding
import numpy as np
from models import MelodyLSTM
from torch.utils.tensorboard import SummaryWriter


def train(
    model: MelodyLSTM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 1,
    device: torch.device = torch.device("cpu"),
    writer: SummaryWriter = None,
    hparams: dict = None,
    progress: bool = True,
    file: str | Path = None,
) -> tuple[float, float]:
    """Train MelodyLSTM model

    Args:
        model: MelodyLSTM model.
        train_loader: _description_
        validation_loader: _description_
        loss_fn: _description_
        optimizer: _description_
        num_epochs: _description_. Defaults to 1.
        progress: _description_. Defaults to True.
        file: Path in which to save model checkpoints.
        hparams: Dictionary of hyperparameters to log with writer and save to a checkpoint.

    Returns:
        train loss, validation loss
    """
    if writer is not None:
        writer.add_hparams(hparams, metric_dict={}, run_name="hparams")
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
        if file is not None:
            save_checkpoint(
                file=file,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                hparams=hparams,
            )
            print(f"Saved checkpoint to {file}.")
        if progress:
            print(f"loss_va = {loss_va:.4f}")
        if writer is not None:
            writer.add_scalar("Loss/Train_epoch", loss_tr, epoch)
            writer.add_scalar("Loss/Validation_epoch", loss_va, epoch)
    return loss_tr, loss_va


def train_epoch(
    model: MelodyLSTM,
    train_loader: DataLoader,
    loss_fn: callable,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter = None,
    device: torch.device = torch.device("cpu"),
    epoch: int = 1,
    progress: bool = True,
):
    loss_sum = 0
    num_instances = 0
    progress_step = 1000  # Print and log every `progress_step` batch.
    model.train()
    for batch, (inputs, target) in enumerate(train_loader, start=1):
        model.zero_grad()

        inputs = inputs.to(device)  # (batch_size, sequence_length)
        target = target.to(device)  # (batch_size,)
        output = model(inputs)  # (batch_size, sequence_length, num_unique_tokens)

        batch_size = len(inputs)
        num_instances += batch_size
        loss_batch = loss_fn(output[:, -1], target)  # Compare last item (next token).
        loss_sum += loss_batch.detach() * batch_size
        loss_avg = loss_sum / num_instances

        if batch == 1 or (batch % progress_step) == 0:
            if progress:
                print(
                    f"batch {batch}. "
                    f"loss_batch = {loss_batch.item():.4E}. "
                    f"loss = {loss_avg.item():.4E}. "
                )
            if writer is not None:
                step = (epoch - 1) * len(train_loader.dataset) + num_instances
                writer.add_scalar("Loss/Train", loss_avg.item(), step)
        loss_batch.backward()
        optimizer.step()
    return loss_avg.item()


def validate_epoch(
    model: MelodyLSTM,
    validation_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
):
    model.eval()
    loss_sum = 0
    num_instances = 0
    with torch.no_grad():
        for _, (inputs, target) in enumerate(validation_loader, start=1):
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)

            batch_size, sequence_length = inputs.shape
            num_instances += batch_size
            loss_batch = loss_fn(output[:, -1], target)
            loss_sum += loss_batch * batch_size
    return loss_sum.item() / num_instances


def save_checkpoint(file, model, optimizer, epoch, hparams):
    torch.save(
        {
            "name": file,
            "epoch": epoch,
            "hparams": hparams,
            "model_state_dict": model.state_dict(),  # TODO What exactly is in state_dict?
            "optimizer_state_dict": optimizer.state_dict(),
        },
        file,
    )


def load_checkpoint(file):
    d = torch.load(file)
    return d


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
        num_files: int = None,
    ):
        """Time series dataset of notes

        Args:
            path: Directory of files with space-delimited tokens of midi pitches or a rest symbol or a hold symbol.
            sequence_length: Sequence length to use as input for predicting the next item in the sequence.
            transform: A function (list[str] -> iterable of any) applied to sequences.
            num_files: Number of files to use from path. If None, uses all files.
        """
        self.sequence_length = sequence_length
        self.transform = transform
        self.path = path
        self.files = list(Path(path).glob("*.txt"))
        if num_files is not None:
            self.files = np.random.choice(self.files, size=num_files, replace=False)
        data = []  # list[str]
        for file in self.files:
            data.extend(read_time_series(file))
        if self.transform is not None:
            data = self.transform(data)
        self.data = data

    def __getitem__(self, idx):
        """Get item

        Args:
            idx: Index.

        Returns:
            sequence: iterable of `self.sequence_length` tokens (type determined by self.transform).
            If self.transform is None, `sequence` is a list of strings.
            next_item: next token to predict
        """
        idx1 = idx + self.sequence_length
        return self.data[idx:idx1], self.data[idx1]

    def __len__(self):
        return len(self.data) - self.sequence_length


def get_data(
    name: str,
    path: str | Path,
    sequence_length: int,
    encoding: dict = None,
    num_files: int = None,
) -> torch.utils.data.Dataset:
    """Get dataset by name

    Args:
        name: Name of the dataset.
        path: Path to the data.
        sequence_length: Sequence length.
        encoding: Encoding. Defaults to None.
        num_files: Number of files to load from path. Defaults to None.

    Returns:
        torch Dataset
    """
    if name == "maestro-v3.0.0-time_series":
        data = TimeSeriesDataset(
            path=path,
            sequence_length=sequence_length,
            transform=lambda seq: torch.tensor([encoding[e] for e in seq]),
            num_files=num_files,
        )
    else:
        raise ValueError(f"Unknown data_name ({data_name}).")
    return data


def make_data_loaders(
    data: torch.utils.data.Dataset,
    batch_size: int,
    pct_tr: float,
    seed_split: int = None,
    seed_loader: int = None,
) -> tuple[DataLoader, DataLoader]:
    """Make train and validation data loaders

    Args:
        data: Torch dataset.
        batch_size: Batch size.
        pct_tr: Percent of data for training set.
        seed_split: Random seed for splitting to train-validation. Defaults to None.
        seed_loader: Random seed for train loader. Defaults to None.

    Returns:
        tuple of torch DataLoaders
    """
    data_tr, data_va = random_split(
        data,
        lengths=(pct_tr, 1 - pct_tr),
        generator=torch.Generator().manual_seed(seed_split),
    )
    train_loader = DataLoader(
        data_tr,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed_loader),
    )
    validation_loader = DataLoader(data_va, batch_size=1, shuffle=False)
    return train_loader, validation_loader


if __name__ == "__main__":
    from datetime import datetime
    from config import (
        LEARNING_RATE,
        BATCH_SIZE,
        NUM_EPOCHS,
        NUM_FILES,
        SEED_SPLIT,
        SEED_LOADER,
        PCT_TR,
        SEQUENCE_LENGTH,
        NUM_UNIQUE_TOKENS,
        EMBEDDING_SIZE,
        HIDDEN_SIZE,
        DEVICE,
        PATH_TO_MODELS,
    )

    data_name = "maestro-v3.0.0-time_series"
    path_to_txt_data = Path(f"data/{data_name}")

    t_start = datetime.now()
    timestamp = t_start.strftime("%Y-%m-%dT%H-%M-%S")
    model_name = f"melodylstm_{data_name}_{timestamp}"

    path_to_models = Path(PATH_TO_MODELS)
    path_to_models.mkdir(parents=True, exist_ok=True)
    model_file = path_to_models / f"{model_name}.pth"

    data = get_data(
        name=data_name,
        path=path_to_txt_data,
        sequence_length=SEQUENCE_LENGTH,
        encoding=encoding,
        num_files=NUM_FILES,
    )
    train_loader, validation_loader = make_data_loaders(
        data,
        batch_size=BATCH_SIZE,
        pct_tr=PCT_TR,
        seed_split=SEED_SPLIT,
        seed_loader=SEED_LOADER,
    )

    model = MelodyLSTM(
        num_unique_tokens=NUM_UNIQUE_TOKENS,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
    ).to(DEVICE)
    # TODO Check str device works.
    loss_fn = nn.NLLLoss()  # Input: log probabilities
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    hparams = {
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_files": NUM_FILES,
        "num_sequences": len(data),
        "data_name": data_name,
        "num_unique_tokens": NUM_UNIQUE_TOKENS,
        "embedding_size": EMBEDDING_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "seed_split": SEED_SPLIT,
        "seed_loader": SEED_LOADER,
        # TODO Where to save the encoding? Maybe save path to encoding? Or just save it to hparams?
    }
    # TODO Log to file.
    print("Training model...")
    print(
        f"Model: {model_name}\n"
        f"Data: {len(data)} sequences from {len(data.files)} files "
        f"(tr+va = {len(train_loader.dataset)}+{len(validation_loader.dataset)}). "
    )
    print(f"Hyperparameters: {str(hparams)}")
    loss_tr, loss_va = train(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        writer=SummaryWriter(f"runs/{model_name}", flush_secs=30),
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        hparams=hparams,
        file=model_file,
    )

    t_end = datetime.now()
    print(f"Done. ({t_end:%F %T}, {(t_end - t_start).seconds} sec)")
