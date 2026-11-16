import torch

from nanogpt.data import CharTokeniser, get_batch, train_val_split
from nanogpt.model import GPT, GPTConfig

# training hyperparameters
batch_size = 64  # independent sequences processed in parallel
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "mps" if torch.mps.is_available() else "cpu"
eval_iters = 200


# average loss over eval_iters batches for each split
@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, torch.Tensor]) -> dict[str, float]:
    out = {}
    model.eval()
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, model.config.block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    torch.manual_seed(1337)

    with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokeniser = CharTokeniser(text)
    config = GPTConfig(vocab_size=tokeniser.vocab_size)

    # train and test splits
    data = torch.tensor(tokeniser.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(data)

    model = GPT(config)
    model = model.to(device)

    # create PyTorch optimizer
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        # every eval_interval steps evaluate loss on train and val
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, {"train": train_data, "val": val_data})
            print(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch(train_data, config.block_size, batch_size, device)

        # evaluate the loss
        _, loss = model(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    # generate from model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokeniser.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))


if __name__ == "__main__":
    main()
