import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import csv
from fractions import Fraction
from torch import optim


class MelodyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
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

    def forward(self, x):
        """Forward pass

        Args:
            x: torch tensor of shape (seq_len, input_size)

        Returns:
            torch tensor of shape (seq_len, output_size)
        """
        # lstm's default h0 (hidden state) and c0 (cell state) are zeroes.
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out.view(len(x), -1))
        # TODO One-hot encode pitches. Or use nn.Embedding?
        # TODO Enforce positive pitch, duration and offset (e.g. exp or softplus?)
        return out


def train_model(
    model: MelodyLSTM, train_loader, loss_fn, optimizer, num_epochs=10, progress=True
):
    model.train()
    i = 0
    for epoch in range(num_epochs):
        if progress:
            print(f"Epoch {epoch}")
        for inputs, target in train_loader:
            # TODO Cant do multiple epochs with a generator. Use custom torch Dataset. Not sure how to make it return pieces of a file with each __get__item.
            if progress and (i % 100) == 0:
                print(f"i={i}")

            model.zero_grad()

            inputs = scale_pitch(torch.tensor(inputs, dtype=torch.float32))
            target = scale_pitch(torch.tensor(target, dtype=torch.float32))

            output = model(inputs)
            loss = loss_fn(output[-1], target)
            loss.backward()
            optimizer.step()
            i = i + 1
    return model


def scale_pitch(x, pitch_range=128):
    """Scale pitch

    Args:
        x: Tuple of (pitch, duration, offset)
        pitch_range: Maximum pitch. Defaults to 128.

    Returns:
        x with its first element divided by pitch_range
    """
    return x / torch.tensor([pitch_range, 1.0, 1.0])


def make_data_loader(files: list[str | Path], read_fn: callable, sequence_length: int):
    """Make lazy data loader for sequence data

    Args:
        files: list of files containing sequences
        read_fn: function that reads a file into a list
        sequence_length: Length of input sequence to use as context.

    Yields:
        inputs: list of `sequence_length` tokens
        output: next token to predict
    """
    for file in files:
        sequence = read_fn(file)
        # TODO Handle case where len(sequence) < sequence length?
        for i in range(len(sequence) - sequence_length):
            inputs = sequence[i : i + sequence_length]
            output = sequence[i + sequence_length]
            yield inputs, output


class EventSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, sequence_length: int):
        """Event sequence dataset

        Args:
            path: Directory of files with comma-delimited tuples of midi pitch,duration,offset e.g. 36,0.25,0. Each line is an event.
            sequence_length: Sequence length to use as input for predicting the next item in the sequence.
        """
        self.path = path
        self.files = Path(path).glob("*.txt")
        self.sequence_length = sequence_length
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
        return sequence, next_item

    def __len__(self):
        return len(self.data) - self.sequence_length


def read_time_series(file: str | Path) -> list:
    """Read txt file containing space-delimited sequences of pitch numbers or rest tokens or hold tokens"""
    with open(file) as f:
        return f.read().split(" ")


def read_event_sequence(file: str | Path) -> list:
    """Read txt file containing newline-delimited sequences of (pitch, duration, offset)"""
    with open(file) as f:
        data = []
        for row in csv.reader(f):
            pitch, duration, offset = row
            if duration.find("/"):
                duration = Fraction(duration)
            if offset.find("/"):
                offset = Fraction(offset)
            data.append((int(pitch), float(duration), float(offset)))
    return data


if __name__ == "__main__":
    from datetime import datetime

    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, SEQUENCE_LENGTH = 3, 64, 3, 16
    learning_rate = 0.1

    dataset = "event_sequence"  # 3696777 notes in 1276 files
    path_to_dataset_txt = Path(f"data/{dataset}")
    data = EventSequenceDataset(
        path=path_to_dataset_txt, sequence_length=SEQUENCE_LENGTH
    )
    data_loader = DataLoader(data, shuffle=True)

    model = MelodyLSTM(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE
    )
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    model_ = train_model(
        model=model,
        train_loader=data_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=1,
    )

    model_file = f"model_{datetime.now().isoformat(timespec='seconds')}_{dataset}.pkl"
    torch.save(model_, model_file)
