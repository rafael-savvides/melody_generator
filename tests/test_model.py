from models import MelodyLSTM
import torch


def test_dims():
    batch_size = 1
    inputs = torch.tensor([1, 5, 7]).reshape(-1, batch_size)
    seq_len = len(inputs)
    model = MelodyLSTM(num_unique_tokens=10, embedding_size=5, hidden_size=12)
    assert model(inputs).shape == (seq_len, batch_size, model.num_unique_tokens)


def test_dims_list_input():
    batch_size = 1
    inputs = [1, 5, 7]
    seq_len = len(inputs)
    model = MelodyLSTM(num_unique_tokens=10, embedding_size=5, hidden_size=12)
    assert model(inputs).shape == (seq_len, batch_size, model.num_unique_tokens)
