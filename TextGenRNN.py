"""
Character-level RNN Text Generation (PyTorch)
Tiny Shakespeare + one-hot encoded inputs

Install dependencies:
    pip install torch urllib3

Run:
    python rnn_shakespeare_onehot_torch.py
"""

import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load dataset ───────────────────────────────────────────────
CACHE = "tiny_shakespeare.txt"
if not os.path.exists(CACHE):
    print("Downloading Tiny Shakespeare …")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        CACHE)
with open(CACHE, encoding="utf-8") as f:
    train_data = f.read()
print(f"Dataset: {len(train_data):,} characters")

# ── Vocabulary ─────────────────────────────────────────────────
chars      = sorted(set(train_data))
vocab_size = len(chars)
c2i        = {c: i for i, c in enumerate(chars)}
i2c        = np.array(chars)
text_as_int = np.array([c2i[c] for c in train_data], dtype=np.int64)

print(f"Vocabulary size: {vocab_size}")

# ── Dataset ────────────────────────────────────────────────────
SEQ_LEN    = 100
BATCH_SIZE = 64

class ShakespeareDataset(Dataset):
    def __init__(self, encoded, seq_len):
        # one-hot inputs computed in __getitem__ to avoid huge memory allocation
        n = (len(encoded) - 1) // seq_len
        self.x = torch.tensor(
            np.stack([encoded[i*seq_len : i*seq_len + seq_len]     for i in range(n)]),
            dtype=torch.long)
        self.y = torch.tensor(
            np.stack([encoded[i*seq_len + 1 : i*seq_len + seq_len + 1] for i in range(n)]),
            dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_oh = torch.nn.functional.one_hot(
            self.x[idx], num_classes=vocab_size).float()   # (SEQ_LEN, vocab_size)
        return x_oh, self.y[idx]

ds = ShakespeareDataset(text_as_int, SEQ_LEN)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ── Model ──────────────────────────────────────────────────────
HIDDEN_SIZE = 1024

class CharRNN(nn.Module):
    """One-hot input RNN — no embedding layer needed."""
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super().__init__()
        self.rnn     = nn.RNN(vocab_size, hidden_size,
                              batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        # x: (B, T, vocab_size)
        out, h = self.rnn(self.dropout(x), h)   # out: (B, T, hidden)
        logits = self.fc(out)                   # (B, T, vocab_size)
        return logits, h

model = CharRNN(vocab_size, HIDDEN_SIZE).to(DEVICE)
print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# ── Loss, perplexity, optimizer ────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def compute_perplexity(loss: float) -> float:
    return float(np.exp(loss))

# ── Evaluate ───────────────────────────────────────────────────
def evaluate(loader, label="eval"):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x_oh, y in loader:
            x_oh, y = x_oh.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x_oh)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()
            n          += 1
    avg_loss = total_loss / n
    print(f"[{label}]  loss: {avg_loss:.4f}  |  perplexity: {compute_perplexity(avg_loss):.2f}")
    return avg_loss

# ── Train ──────────────────────────────────────────────────────
EPOCHS   = 10
best_loss = float("inf")

print(f"\n── Training for {EPOCHS} epochs ──")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n = 0.0, 0

    for x_oh, y in dl:
        x_oh, y = x_oh.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x_oh)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n          += 1

    avg_loss   = total_loss / n
    perplexity = compute_perplexity(avg_loss)
    print(f"Epoch {epoch:>2}/{EPOCHS}  loss: {avg_loss:.4f}  perplexity: {perplexity:.2f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "shakespeare_rnn_onehot.pt")
        print(f"           ✓ checkpoint saved (loss={best_loss:.4f})")

# ── Final evaluation ───────────────────────────────────────────
model.load_state_dict(torch.load("shakespeare_rnn_onehot.pt", map_location=DEVICE))
print("\n── Final metrics ──")
evaluate(dl, label="train")

# ── Generate ───────────────────────────────────────────────────
def generate_text(seed: str, length: int = 500, temperature: float = 0.8) -> str:
    model.eval()
    ids    = [c2i[c] for c in seed if c in c2i]
    result = list(seed)

    # Build one-hot from seed: (1, T, vocab_size)
    x = torch.nn.functional.one_hot(
        torch.tensor(ids, dtype=torch.long),
        num_classes=vocab_size).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Warm up hidden state on the seed
        _, h = model(x)

        # Generate one character at a time
        x = x[:, -1:, :]                       # last seed token (1, 1, vocab_size)
        for _ in range(length):
            logits, h = model(x, h)             # (1, 1, vocab_size)
            logits = logits[0, 0] / temperature
            probs  = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            result.append(i2c[next_id])
            x = torch.nn.functional.one_hot(
                torch.tensor([[next_id]], dtype=torch.long),
                num_classes=vocab_size).float().to(DEVICE)

    return "".join(result)


print("\n" + "═" * 60)
print(generate_text("ROMEO:\n", length=500))
print("═" * 60)