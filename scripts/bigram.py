with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()


chars = sorted(set(text))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

print(encode("hi there"))
print(decode(encode("hi there")))
