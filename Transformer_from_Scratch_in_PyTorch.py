import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.fc_out(x)

# Task: Reverse a sequence
vocab_size = 10
seq_len = 5
model = MiniTransformer(vocab_size=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
print("Training Transformer to reverse sequences...")
for epoch in range(300):
    # Generate random sequences (1 to 9)
    inputs = torch.randint(1, 10, (32, seq_len))
    targets = torch.flip(inputs, [1]) # Reversed target
    
    optimizer.zero_grad()
    output = model(inputs)
    
    # Reshape for loss calculation
    loss = criterion(output.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test
test_input = torch.tensor([[1, 2, 3, 4, 5]])
with torch.no_grad():
    pred = model(test_input)
    pred_seq = torch.argmax(pred, dim=-1).squeeze()
print(f"\nInput: {test_input.squeeze().tolist()}")
print(f"Predicted Reversal: {pred_seq.tolist()}")