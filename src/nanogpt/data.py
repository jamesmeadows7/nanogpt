import torch


# character-level tokeniser: maps each unique character to an integer
class CharTokeniser:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        # take a string, output list of integers
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        # take a list of integers, output a string
        return "".join(self.itos[i] for i in ids)


# split data into train and validation sets
def train_val_split(
    data: torch.Tensor, train_frac: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(train_frac * len(data))
    return data[:n], data[n:]


# sample a random batch of inputs and targets
def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
