"""
Word-level LSTM Text Generation (PyTorch)
WikiText-103 + Word2Vec embeddings (two-phase training)

Install dependencies:
    pip install torch datasets gensim

Run:
    python lstm_wikitext_w2v_torch.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from gensim.models import Word2Vec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 1. Load WikiText-103 ───────────────────────────────────────
print("Loading WikiText-103 …")
raw = load_dataset("wikitext", "wikitext-103-v1")

def extract_tokens(split):
    lines  = [row["text"] for row in raw[split] if row["text"].strip()]
    joined = " ".join(lines)
    return joined.lower().split()

train_tokens = extract_tokens("train")
val_tokens   = extract_tokens("validation")

print(f"Train tokens : {len(train_tokens):,}")
print(f"Val tokens   : {len(val_tokens):,}")

# ── 2. Train Word2Vec ──────────────────────────────────────────
EMBED_DIM  = 128
W2V_WINDOW = 5
W2V_MIN    = 5
CHUNK      = 30

print("\nTraining Word2Vec …")
sentences = [train_tokens[i: i + CHUNK]
             for i in range(0, len(train_tokens) - CHUNK, CHUNK)]

w2v = Word2Vec(sentences, vector_size=EMBED_DIM, window=W2V_WINDOW,
               min_count=W2V_MIN, workers=4, epochs=5, seed=42)

print(f"Word2Vec vocabulary: {len(w2v.wv):,} words")

# ── 3. Build vocabulary & embedding matrix ─────────────────────
SPECIAL    = ["<PAD>", "<UNK>"]
vocab      = SPECIAL + list(w2v.wv.index_to_key)
w2i        = {w: i for i, w in enumerate(vocab)}
i2w        = {i: w for w, i in w2i.items()}
VOCAB_SIZE = len(vocab)
UNK_ID     = w2i["<UNK>"]
PAD_ID     = w2i["<PAD>"]

embedding_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM), dtype=np.float32)
for word, idx in w2i.items():
    if word in w2v.wv:
        embedding_matrix[idx] = w2v.wv[word]

print(f"Final vocabulary size: {VOCAB_SIZE:,}")

# ── 4. Encode tokens ───────────────────────────────────────────
def encode(tokens):
    return np.array([w2i.get(t, UNK_ID) for t in tokens], dtype=np.int64)

train_enc = encode(train_tokens)
val_enc   = encode(val_tokens)

# ── 5. Dataset & DataLoader ────────────────────────────────────
SEQ_LEN    = 64
BATCH_SIZE = 128

class TokenDataset(Dataset):
    def __init__(self, encoded, seq_len):
        n = (len(encoded) - 1) // seq_len
        self.x = torch.tensor(
            np.stack([encoded[i*seq_len : i*seq_len + seq_len]       for i in range(n)]),
            dtype=torch.long)
        self.y = torch.tensor(
            np.stack([encoded[i*seq_len + 1 : i*seq_len + seq_len + 1] for i in range(n)]),
            dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_ds = TokenDataset(train_enc, SEQ_LEN)
val_ds   = TokenDataset(val_enc,   SEQ_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# ── 6. Model ───────────────────────────────────────────────────
HIDDEN_SIZE = 512

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, emb_matrix):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.embedding.weight = nn.Parameter(
            torch.tensor(emb_matrix), requires_grad=False)   # frozen in phase 1
        self.lstm    = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc      = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))       # (B, T, E)
        out, hidden = self.lstm(emb, hidden)        # (B, T, H)
        logits = self.fc(self.dropout(out))         # (B, T, V)
        return logits, hidden

    def set_embeddings_trainable(self, trainable: bool):
        self.embedding.weight.requires_grad = trainable

model = LSTMLanguageModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, embedding_matrix).to(DEVICE)
print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)

# ── 7. Loss, perplexity, optimizer ────────────────────────────
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def make_optimizer(trainable_emb=False):
    model.set_embeddings_trainable(trainable_emb)
    return optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# ── 8. Evaluate ───────────────────────────────────────────────
def evaluate(loader, label="eval"):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss   = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            total += loss.item()
            n     += 1
    avg_loss   = total / n
    perplexity = float(np.exp(avg_loss))
    print(f"[{label}]  loss: {avg_loss:.4f}  |  perplexity: {perplexity:.2f}")
    return avg_loss, perplexity

# ── 9. Training loop ──────────────────────────────────────────
best_val_loss = float("inf")

def train_phase(epochs, trainable_emb, label):
    global best_val_loss
    optimizer = make_optimizer(trainable_emb)
    print(f"\n── {label} ──")
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss   = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
            n     += 1

        tr_loss = total / n
        tr_ppl  = float(np.exp(tr_loss))
        vl_loss, vl_ppl = evaluate(val_dl, label="val")
        print(f"  Epoch {epoch:>2}/{epochs} | "
              f"train loss {tr_loss:.4f} ppl {tr_ppl:.1f} | "
              f"val loss {vl_loss:.4f} ppl {vl_ppl:.1f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), "lstm_wikitext_w2v.pt")
            print(f"           ✓ checkpoint saved (val_loss={best_val_loss:.4f})")

train_phase(5,  trainable_emb=False, label="Phase 1: frozen Word2Vec embeddings (5 epochs)")
train_phase(10, trainable_emb=True,  label="Phase 2: fine-tuning embeddings + LSTM (10 epochs)")

# ── 10. Final evaluation ──────────────────────────────────────
model.load_state_dict(torch.load("lstm_wikitext_w2v.pt", map_location=DEVICE))
print("\n── Final metrics ──")
evaluate(train_dl, label="train")
evaluate(val_dl,   label="val  ")

# ── 11. Generate ─────────────────────────────────────────────
def generate_text_wiki(seed: str, length: int = 100,
                       temperature: float = 0.8) -> str:
    model.eval()
    seed_tokens = seed.lower().split()
    ids = [w2i.get(t, UNK_ID) for t in seed_tokens]
    x   = torch.tensor([ids], dtype=torch.long).to(DEVICE)   # (1, T)

    result = list(seed_tokens)
    with torch.no_grad():
        # Warm up hidden state on seed
        _, hidden = model(x)

        # Generate one word at a time
        x = x[:, -1:]                                         # last seed token
        for _ in range(length):
            logits, hidden = model(x, hidden)                 # (1, 1, V)
            logits = logits[0, 0] / temperature
            probs  = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            result.append(i2w[next_id])
            x = torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)

    return " ".join(result)


print("\n" + "═" * 60)
print(generate_text_wiki("the history of science", length=100))
print("═" * 60)