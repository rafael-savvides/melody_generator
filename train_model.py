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
            # TODO Should every epoch be saved, or should I overwrite? If every epoch, then how to name them? _epochs=epoch? Should there be a folder with all epochs? Save only if validation loss improves?
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


def save_checkpoint(file, model, optimizer, epoch, hparams):
    torch.save(
        {
            "name": file,  # TODO How to name?
            "epoch": epoch,
            "hparams": hparams,
            "model_state_dict": model.state_dict(),  # TODO What exactly is in state_dict?
            "optimizer_state_dict": optimizer.state_dict(),
        },
        file,
    )


def load_checkpoint(file):
    # TODO What to return?
    d = torch.load(file)
    return d


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
        for batch, (inputs, target) in enumerate(validation_loader, start=1):
            # TODO Can the loader be loaded on the device?
            # TODO Use pin_memory in the loader.
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)

            batch_size, sequence_length = inputs.shape
            num_instances += batch_size
            loss_batch = loss_fn(output[:, -1], target)
            loss_sum += loss_batch * batch_size
    return loss_sum.item() / num_instances


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


if __name__ == "__main__":
    from datetime import datetime
    from models import config

    learning_rate = 0.01
    num_epochs = 50
    num_files = 1  # Number of files to use in the data folder.

    data_name = "time_series"
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_name = f"melodylstm_{data_name}_{timestamp}"

    path_to_models = Path("models")
    path_to_models.mkdir(parents=True, exist_ok=True)
    model_file = path_to_models / f"{model_name}.pth"

    path_to_dataset_txt = Path(f"data/{data_name}")
    # TODO Include raw data name (maestro) in processed data path and as a config param.

    print(model_name)
    if data_name == "event_sequence":
        raise NotImplementedError
        # data = EventSequenceDataset(
        #     path=path_to_dataset_txt,
        #     sequence_length=config["sequence_length"],
        #     transform=scale_pitch,
        # )
        # data_loader = DataLoader(data, shuffle=True)  # TODO Update data loader.

        # model = MelodyLSTMPlus(
        #     pitch_range=config["num_unique_tokens"] - 2,
        #     embedding_size=config["embedding_size"],
        #     hidden_size=config["hidden_size"],
        # )
        # # TODO Change loss to NLLLoss + MSELoss. Can use ignore_index.
        # loss_fn = nn.MSELoss()
    elif data_name == "time_series":
        data = TimeSeriesDataset(
            path=path_to_dataset_txt,
            sequence_length=config["sequence_length"],
            transform=lambda seq: torch.tensor([encoding[e] for e in seq]),
            num_files=num_files,
        )
        seed_split = 42
        pct_tr = 0.8
        generator = torch.Generator().manual_seed(seed_split)
        data_tr, data_va = random_split(data, (pct_tr, 1 - pct_tr), generator=generator)
        train_loader = DataLoader(data_tr, batch_size=16, shuffle=True)
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

    hparams = config | {
        "lr": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "num_files": num_files,
        "num_sequences": len(data),
        "data_name": data_name,
    }

    loss_tr, loss_va = train(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        writer=writer,
        num_epochs=num_epochs,
        device=device,
        hparams=hparams,
        file=model_file,
    )
    with writer as w:
        w.add_hparams(
            hparams,
            metric_dict={"loss_tr": loss_tr, "loss_va": loss_va},
            run_name=f"hparams_{timestamp}",
        )

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
