import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from torch import optim
from prepare_data import read_event_sequence, read_time_series
import numpy as np
from models import MelodyLSTM
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse


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
    encoding: dict = None,
    progress: bool = True,
    file: str | Path = None,
    checkpoint: str | Path = None,
) -> tuple[float, float]:
    """Train MelodyLSTM model

    Args:
        model: MelodyLSTM model.
        train_loader: Torch DataLoader for training data.
        validation_loader: Torch DataLoader for validation data.
        loss_fn: Loss function.
        optimizer: Torch optimizer.
        num_epochs: Number of epochs. Defaults to 1.
        hparams: Dictionary of hyperparameters to log with writer and save to a checkpoint.
        encoding: Dictionary that maps MIDI notes as strings to integers fed into the model.
        progress: If True, prints losses per several batches and per epoch. Defaults to True.
        file: Path in which to save model checkpoints.
        checkpoint: Path from which to load a model checkpoint.

    Returns:
        train loss, validation loss
    """
    if writer is not None:
        writer.add_hparams(hparams, metric_dict={}, run_name="hparams")
    if checkpoint is not None:
        load_checkpoint(checkpoint, model, optimizer)
        log(f"Loaded model and optimizer checkpoint from {checkpoint}.")
    for epoch in range(1, num_epochs + 1):
        model.train()
        if progress:
            log(f"Epoch {epoch}")
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

        loss_va = validate_epoch(model, loss_fn, validation_loader, device=device)
        if file is not None:
            save_checkpoint(
                file=file,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                hparams=hparams,
                encoding=encoding,
            )
            log(f"Saved checkpoint to {file}.")
        if progress:
            log(f"loss_va = {loss_va:.4f}")
            log(f"loss_tr = {loss_tr:.4f}")
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
) -> float:
    """Train one epoch of a torch model

    Args:
        model: Torch model.
        train_loader: torch DataLoader for training data.
        loss_fn: Loss function.
        optimizer: Torch optimizer.
        writer: Tensorboard writer.. Defaults to None.
        device: Torch device. Defaults to torch.device("cpu").
        epoch: Current epoch.. Defaults to 1.
        progress: If true, print loss every 1000 batches. Defaults to True.

    Returns:
        average loss on the training data
    """
    loss_sum = 0
    num_instances = 0
    progress_step = 1000  # Print and log every `progress_step` batch.
    model.train()
    for batch, (inputs, target) in enumerate(train_loader, start=1):
        model.zero_grad()

        inputs = inputs.to(device)  # (batch_size, sequence_length)
        target = target.to(device)  # (batch_size,)
        output = model(inputs)  # (batch_size, sequence_length, output_size)

        batch_size = len(inputs)
        num_instances += batch_size
        loss_batch = loss_fn(output[:, -1], target)  # Compare last item (next token).
        loss_sum += loss_batch.detach() * batch_size
        loss_avg = loss_sum / num_instances

        loss_batch.backward()
        if batch == 1 or (batch % progress_step) == 0:
            if progress:
                log(
                    f"batch {batch}. "
                    f"loss_batch = {loss_batch.item():.4E}. "
                    f"loss = {loss_avg.item():.4E}. "
                )
            if writer is not None:
                step = (epoch - 1) * len(train_loader.dataset) + num_instances
                grad_norm = check_gradient_norm(model)
                writer.add_scalar("Gradient Norm", grad_norm, step)
                writer.add_scalar("Loss/Train", loss_avg.item(), step)
        optimizer.step()
    return loss_avg.item()


@torch.no_grad()
def validate_epoch(
    model: MelodyLSTM,
    loss_fn: callable,
    validation_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute the average validation loss

    Args:
        model: Torch model (nn.Module).
        loss_fn: Loss function.
        validation_loader: Torch DataLoader for validation data.
        device: Torch device. Defaults to torch.device("cpu").

    Returns:
        average loss on the validation data
    """
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


def check_gradient_norm(model: nn.Module) -> float:
    """Check the L2 norm of the gradient of the (learnable) parameters of a model"""
    return (
        sum(
            [
                param.grad.data.norm(2).item() ** 2
                for param in model.parameters()
                if param.grad is not None
            ]
        )
        ** 0.5
    )


def count_model_parameters(model: nn.Module):
    """Count number of learnable model parameters in a PyTorch model"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def save_checkpoint(file, model, optimizer, epoch, hparams, encoding):
    torch.save(
        {
            "name": file,
            "epoch": epoch,
            "hparams": hparams,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "encoding": encoding,
        },
        file,
    )


def load_checkpoint(
    file: str | Path, model: nn.Module = None, optimizer: torch.optim.Optimizer = None
):
    d = torch.load(file)
    if model is not None:
        model.load_state_dict(d["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(d["optimizer_state_dict"])
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
    """Time series dataset

    An item is a tuple (sequence, next_item).
    """

    def __init__(
        self,
        path: str | Path,
        sequence_length: int,
        transform: callable = None,
        size: int = None,
    ):
        """Time series dataset of notes

        Args:
            path: Path to a text file or a directory of text files. The text file is a sequence of tokens, see :func:`read_time_series`.
            sequence_length: Sequence length to use as input for predicting the next item in the sequence.
            transform: A function (list[str] -> iterable of any) applied to sequences.
            size: Number of data items, i.e., dataset length (number of tokens - sequence length). If None, uses all data in `path`. Else reads the first `size` sequence items from files in `path` (files are read in Path.glob()'s order).
        """
        self.sequence_length = sequence_length
        self.transform = transform
        self.path = Path(path)
        if self.path.is_file():
            self.files = [self.path]
        else:
            self.files = list(self.path.glob("*.txt"))
        data: list[str] = []
        for file in self.files:
            songs = read_time_series(file)
            for song in songs:
                data.extend(song)
            if size is not None:
                if len(data) - self.sequence_length >= size:
                    data = data[: size + self.sequence_length]
                    break
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
    size: int = None,
) -> torch.utils.data.Dataset:
    """Get dataset by name

    Args:
        name: Name of the dataset.
        path: Path to the data.
        sequence_length: Sequence length.
        encoding: Encoding. Defaults to None.
        size: Data size. Defaults to None.

    Returns:
        torch Dataset
    """
    if name == "maestro-v3.0.0-time_series":
        data = TimeSeriesDataset(
            path=path,
            sequence_length=sequence_length,
            transform=lambda seq: torch.tensor([encoding[e] for e in seq]),
            size=size,
        )
    elif name == "jsb_chorales":
        data = TimeSeriesDataset(
            path=path,
            sequence_length=sequence_length,
            transform=lambda seq: torch.tensor([encoding[e] for e in seq]),
            size=size,
        )
    else:
        raise ValueError(f"Unknown data_name ({name}).")
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
    batch_size_validation = 10000
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
    validation_loader = DataLoader(
        data_va, batch_size=batch_size_validation, shuffle=False
    )
    return train_loader, validation_loader


def make_logger(file=None):
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    if file is not None:
        formatter = logging.Formatter(
            "{asctime}. {message}", style="{", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(filename=file, encoding="utf8")
        logger.addHandler(file_handler)
        file_handler.setFormatter(formatter)
    return logger


def log(message):
    try:
        logger.info(message)
    except:
        print(message)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type="str",
        default=None,
        help="Model checkpoint to load. If None, train from scratch.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type="str",
        default="jsb_chorales",
        help=f"Dataset to train on. One of: {', '.join(DATASETS.keys())}.",
    )
    return parser


if __name__ == "__main__":
    from datetime import datetime
    from encoder import get_encoding
    from config import (
        LEARNING_RATE,
        BATCH_SIZE,
        NUM_EPOCHS,
        DATA_SIZE,
        SEED_SPLIT,
        SEED_LOADER,
        PCT_TR,
        SEQUENCE_LENGTH,
        OUTPUT_SIZE,
        EMBEDDING_SIZE,
        HIDDEN_SIZE,
        DEVICE,
        PATH_TO_MODELS,
        PATH_TO_DATA,
        PATH_TO_LOGS,
        DATASETS,
        OPTIMIZER,
        OPTIMIZER_PARAMS,
        DROPOUT,
    )

    parser = make_parser()
    args = parser.parse_args()

    dataset = args.dataset
    checkpoint = args.checkpoint

    t_start = datetime.now()
    timestamp = t_start.strftime("%Y-%m-%dT%H-%M-%S")
    model_name = f"melodylstm_{dataset}_{timestamp}"

    path_to_models = Path(PATH_TO_MODELS)
    path_to_models.mkdir(parents=True, exist_ok=True)
    model_file = path_to_models / f"{model_name}.pth"

    Path(PATH_TO_LOGS).mkdir(parents=True, exist_ok=True)
    logger = make_logger(Path(PATH_TO_LOGS) / f"{model_name}.txt")

    encoding, decoding = get_encoding(dataset)
    OUTPUT_SIZE = len(encoding) if OUTPUT_SIZE is None else OUTPUT_SIZE
    data = get_data(
        name=dataset,
        path=Path(PATH_TO_DATA) / DATASETS[dataset]["processed"],
        sequence_length=SEQUENCE_LENGTH,
        encoding=encoding,
        size=DATA_SIZE,
    )
    train_loader, validation_loader = make_data_loaders(
        data,
        batch_size=BATCH_SIZE,
        pct_tr=PCT_TR,
        seed_split=SEED_SPLIT,
        seed_loader=SEED_LOADER,
    )
    # TODO Fix initialization seed (also for optimizer?)
    model = MelodyLSTM(
        output_size=OUTPUT_SIZE,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)
    loss_fn = nn.NLLLoss()  # Input: log probabilities

    if OPTIMIZER == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, **OPTIMIZER_PARAMS)
    elif OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, **OPTIMIZER_PARAMS)
    else:
        raise ValueError(f"Unknown OPTIMIZER={OPTIMIZER}.")

    hparams = {
        "dataset": dataset,
        "num_sequences": len(data),
        "embedding_size": EMBEDDING_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "data_size": DATA_SIZE,
        "output_size": OUTPUT_SIZE,
        "seed_split": SEED_SPLIT,
        "seed_loader": SEED_LOADER,
        "device": DEVICE,
        "optimizer": OPTIMIZER,
        "optimizer_params": str(OPTIMIZER_PARAMS),
        "dropout": DROPOUT,
    }

    log(f"Training model {model_name} ({count_model_parameters(model)} parameters)")
    log(
        f"Data: {len(data)} sequences from {len(data.files)} files "
        f"(tr+va = {len(train_loader.dataset)}+{len(validation_loader.dataset)}). "
    )
    log(f"Hyperparameters: {str(hparams)}")
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
        encoding=encoding,
        file=model_file,
        checkpoint=checkpoint,
    )

    t_end = datetime.now()
    log(f"Done. ({t_end:%F %T}, {(t_end - t_start).seconds} sec)")
