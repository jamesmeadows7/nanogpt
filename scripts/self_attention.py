import torch
from torch import nn
from torch.nn import functional as F

# version 1
torch.manual_seed(1337)
B, T, C = 4, 8, 2  # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

# we want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t,C)
        xbow[b, t] = torch.mean(xprev, 0)
print(x)
print(xbow)

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))  # use lower triangular matrix
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print(a)
print(b)
print(c)

# version 2
torch.manual_seed(1337)
B, T, C = 4, 8, 2  # batch, time, channels
x = torch.randn(B, T, C)
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (T, T) @ (B, T, C) -> (B, T, C)
print(xbow2)
print(torch.allclose(xbow, xbow2))

# version 3
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(xbow3)
print(torch.allclose(xbow, xbow3))

# version 4: self-attention
torch.manual_seed(1337)
B, T, C = 4, 8, 32  # batch, time, channels
x = torch.randn(B, T, C)

# single head of self attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B, T, 16)
q = query(x)  # (B, T, 16)
wei = q @ k.transpose(-2, -1) * head_size**0.5  # (B, T, 16) @ (B, 16, T) -> (B, T, T)

tril = torch.tril(torch.ones(T, T))
# lower triangular structure allows us to mask out weight matrix
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
print(wei)

v = value(x)
out = wei @ v
print(out.shape)
