# nanogpt

A small character-level GPT trained on the Tiny Shakespeare dataset. Built following Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) video. The model is a decoder-only Transformer, as introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

## Usage

```
uv sync
uv run python -m nanogpt.train
```

This trains the model and then prints a 1000-character sample of Shakespearean prose.

## Structure

```
nanogpt/
├── data/
│   └── tinyshakespeare.txt   # training data
├── src/nanogpt/
│   ├── model.py              # GPT config and model (attention, feed-forward, blocks, GPT)
│   ├── data.py               # character tokeniser, train/val split, batching
│   └── train.py              # training hyperparameters and training loop
└── scripts/
    ├── bigram.py             # bigram model baseline
    └── self_attention.py     # self-attention maths
```

## Results

The default config (4 layers, 4 heads, 128-dim embeddings, 256-character context) has about 0.84M parameters.

After 5,000 steps: train loss 1.51, val loss 1.71 (from 4.34 at the start).

Sample:

```
ANGELO:
Gives May and fear away; there, Somer weed's worted to preces
That worting jure deace come and twell;
And notled you Poor them? Come your's to the Iaffe!
Swedeep comen as or cel me her?
It to my sone ign; by he
That bant hitself my fallows
But standies, away? my not for what night.

FRORAR:
Shalt all not marr.

ROLAUDIO:
Hady husbands you belass! Let Nor, I shad have fors!
Our shall bearble tof York and him
As 'Who ile at her heir in; buty.

COMINTES:
O 'TI his heir dread I be sony needs?

JULIET:
There a would thou fully Have too eis her.
```
