import torch
import torch.nn as nn
import numpy as np
from torch.functional import softplus, log_softmax

config = {
    "input_size": 3,
    "hidden_size": 64,
    "output_size": 3,
    "sequence_length": 16,
}


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


class MelodyLSTM2(nn.Module):
    def __init__(self, pitch_range: int, embedding_size: int, hidden_size: int):
        """_summary_

        Args:
            input_size: Number of unique tokens (vocabulary size) minus two (duration and offset).
            embedding_size: Embedding dimension.
            hidden_size: Hidden dimension.
        """
        super().__init__()
        self.input_size = pitch_range + 2
        self.pitch_range = pitch_range
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # (seq_len, input_size) -> (seq_len, embedding_size)
        self.pitch_embedding = nn.Embedding(pitch_range, embedding_size)
        # (seq_len, embedding_size) -> (seq_len, hidden_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        # (seq_len, hidden_size) -> (seq_len, input_size)
        self.hidden2pitch = nn.Linear(hidden_size, softplus)
        self.hidden2duration = nn.Linear(hidden_size, 1)
        self.hidden2offset = nn.Linear(hidden_size, 1)

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
        embeds = self.pitch_embedding(x)
        lstm_out, _ = self.lstm(embeds)
        pitch = self.hidden2pitch(lstm_out.view(seq_len, -1))
        pitch_scores = log_softmax(pitch, dim=1)  # Log-probabilities.
        duration = softplus(self.hidden2duration(lstm_out.view(seq_len, -1)))
        offset = softplus(self.hidden2offset(lstm_out.view(seq_len, -1)))
        # TODO Edit loss_fn
        return pitch_scores, duration, offset
