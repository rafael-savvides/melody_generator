import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softplus, log_softmax


class MelodyLSTM(nn.Module):
    __version__ = "0.0.2"

    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0,
    ):
        """LSTM model for melody generation

        A song is a sequence of notes. A note is represented as a pitch or rest or hold token.

        Args:
            output_size: Number of unique tokens (vocabulary size).
            embedding_size: Pitch embedding dimension.
            hidden_size: Hidden LSTM dimension.
            dropout: Dropout rate at LSTM output.
        """
        super().__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # The embedding is useful because the pitch space is sparse.
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size)
        self.embedding = nn.Embedding(output_size, embedding_size)
        # (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, hidden_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, note_seq: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        """Forward pass

        Args:
            note_seq: torch tensor of shape (batch_size, seq_len) and dtype float32.
            If note_seq is a list or numpy array it is cast into a torch tensor.

        Returns:
            Log-probabilities as torch tensor of shape (batch_size, seq_len, output_size)

        Notes:
            To input a single note sequence as a list, it should be a nested list:
            - note_seq = [1,2,3] -> (batch_size, seq_len) = (3, 1)
            - note_seq = [[1,2,3]] -> (batch_size, seq_len) = (1, 3)
        """
        if isinstance(note_seq, np.ndarray) or isinstance(note_seq, list):
            batch_size = len(note_seq)
            note_seq = torch.tensor(note_seq, dtype=torch.int)
            note_seq = note_seq.reshape(batch_size, -1)
        note_seq = torch.atleast_2d(note_seq)
        embeds = self.embedding(note_seq)
        lstm_out, _ = self.lstm(embeds)
        lstm_out_dropout = self.dropout(lstm_out)
        out = self.fc(lstm_out_dropout)
        scores = log_softmax(out, dim=2)
        return scores


class MelodyLSTMEvents(nn.Module):
    def __init__(self, pitch_range: int, embedding_size: int, hidden_size: int):
        """LSTM model for melody generation

        - A song is a sequence of notes. A note is represented as a (pitch, duration, offset) tuple.
        - Pitch is treated as a classification problem with `pitch_range` classes.
        - Duration and offset are treated as a regression problem.

        Args:
            pitch_range: Number of unique tokens (vocabulary size).
            embedding_size: Pitch embedding dimension.
            hidden_size: Hidden LSTM dimension.
        """
        super().__init__()
        self.input_size = pitch_range + 2
        self.pitch_range = pitch_range
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # The embedding is required because the pitch space is too sparse.
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
        # TODO Should x contain (pitch, duration, offset)?
        # Should duration and offset be embedded together with pitch?
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = torch.tensor(np.atleast_2d(x), dtype=torch.float32)
        seq_len = len(x)
        embeds = self.pitch_embedding(x)
        lstm_out, _ = self.lstm(embeds)
        pitch = self.hidden2pitch(lstm_out.view(seq_len, -1))
        duration = softplus(self.hidden2duration(lstm_out.view(seq_len, -1)))
        offset = softplus(self.hidden2offset(lstm_out.view(seq_len, -1)))
        pitch_scores = log_softmax(pitch, dim=1)  # Log-probabilities.
        # TODO Use loss: NLLLoss + MSELoss. Can use ignore_index.
        return pitch_scores, duration, offset
