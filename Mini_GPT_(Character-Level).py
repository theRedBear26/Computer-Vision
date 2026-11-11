import torch
import torch.nn as nn
import torch.optim as optim

text = "hello world, this is a mini gpt model. learning to predict the next character is fun!"
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))
        att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        return self.c_proj(att @ v)

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=32, block_size=16):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.attn = CausalSelfAttention(d_model)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.ln(self.attn(tok_emb + pos_emb))
        return self.head(x)

block_size = 16
model = MiniGPT(vocab_size=vocab_size, block_size=block_size)
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Train
print("Training Mini GPT...")
for step in range(500):
    ix = torch.randint(len(data) - block_size, (16,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    optimizer.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Generate text
context = torch.tensor([encode("hello")], dtype=torch.long)
generated = []
model.eval()
with torch.no_grad():
    for _ in range(30):
        logits = model(context[:, -block_size:])
        next_char = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated.append(next_char)
        context = torch.cat((context, torch.tensor([[next_char]])), dim=1)

print(f"\nGenerated Text: {decode(generated)}")