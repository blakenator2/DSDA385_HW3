"""
EN → DE Seq2Seq RNN Translation (PyTorch)
Multi30k (bentrevett/multi30k) + one-hot word embeddings + BLEU scoring

Install dependencies:
    pip install torch datasets nltk

Run:
    python rnn_multi30k_onehot_torch.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. Load Multi30k
# ─────────────────────────────────────────────
print("Loading Multi30k …")
raw = load_dataset("bentrevett/multi30k")

SOS, EOS, PAD, UNK = "<sos>", "<eos>", "<pad>", "<unk>"

def get_pairs(split):
    en_sents, de_sents = [], []
    for row in raw[split]:
        en_sents.append(row["en"].lower().split())
        de_sents.append(row["de"].lower().split())
    return en_sents, de_sents

train_en, train_de = get_pairs("train")
val_en,   val_de   = get_pairs("validation")
test_en,  test_de  = get_pairs("test")

print(f"Train pairs : {len(train_en):,}")
print(f"Val pairs   : {len(val_en):,}")
print(f"Test pairs  : {len(test_en):,}")

# ─────────────────────────────────────────────
# 2. Build vocabularies
# ─────────────────────────────────────────────
def build_vocab(sentences):
    specials = [PAD, UNK, SOS, EOS]
    words    = sorted({t for sent in sentences for t in sent})
    vocab    = specials + words
    w2i      = {w: i for i, w in enumerate(vocab)}
    i2w      = {i: w for w, i in w2i.items()}
    return w2i, i2w, len(vocab)

en_w2i, en_i2w, EN_VOCAB = build_vocab(train_en)
de_w2i, de_i2w, DE_VOCAB = build_vocab(train_de)

print(f"EN vocab: {EN_VOCAB:,}  |  DE vocab: {DE_VOCAB:,}")

# ─────────────────────────────────────────────
# 3. Encode & pad
# ─────────────────────────────────────────────
MAX_EN = 40
MAX_DE = 42   # +2 for SOS / EOS

def pad_seq(seq, max_len, pad_id):
    seq = seq[:max_len]
    return seq + [pad_id] * (max_len - len(seq))

def encode_en(sent):
    return [en_w2i.get(t, en_w2i[UNK]) for t in sent]

def encode_de(sent):
    return ([de_w2i[SOS]]
            + [de_w2i.get(t, de_w2i[UNK]) for t in sent]
            + [de_w2i[EOS]])

# ─────────────────────────────────────────────
# 4. PyTorch Dataset — one-hot applied in __getitem__
# ─────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, en_sents, de_sents):
        self.enc_in, self.dec_in, self.dec_tgt = [], [], []
        for en, de in zip(en_sents, de_sents):
            ei  = pad_seq(encode_en(en),  MAX_EN, en_w2i[PAD])
            de_ = encode_de(de)
            di  = pad_seq(de_[:-1], MAX_DE, de_w2i[PAD])
            dt  = pad_seq(de_[1:],  MAX_DE, de_w2i[PAD])
            self.enc_in.append(ei)
            self.dec_in.append(di)
            self.dec_tgt.append(dt)
        # store as integer tensors — one-hot applied in __getitem__
        self.enc_in  = torch.tensor(self.enc_in,  dtype=torch.long)
        self.dec_in  = torch.tensor(self.dec_in,  dtype=torch.long)
        self.dec_tgt = torch.tensor(self.dec_tgt, dtype=torch.long)

    def __len__(self):
        return len(self.enc_in)

    def __getitem__(self, idx):
        enc_oh = torch.nn.functional.one_hot(
            self.enc_in[idx], num_classes=EN_VOCAB).float()   # (MAX_EN, EN_VOCAB)
        dec_oh = torch.nn.functional.one_hot(
            self.dec_in[idx], num_classes=DE_VOCAB).float()   # (MAX_DE, DE_VOCAB)
        return enc_oh, dec_oh, self.dec_tgt[idx]

BATCH_SIZE = 64

train_ds = TranslationDataset(train_en, train_de)
val_ds   = TranslationDataset(val_en,   val_de)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ─────────────────────────────────────────────
# 5. Seq2Seq model — SimpleRNN encoder-decoder
#    Input is one-hot so no Embedding layer needed
# ─────────────────────────────────────────────
HIDDEN = 512

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.rnn     = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, MAX_EN, EN_VOCAB)
        _, h = self.rnn(self.dropout(x))   # h: (1, B, HIDDEN)
        return h


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super().__init__()
        self.rnn     = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        # x: (B, MAX_DE, DE_VOCAB)  h: (1, B, HIDDEN)
        out, h = self.rnn(self.dropout(x), h)   # out: (B, T, HIDDEN)
        logits = self.fc(out)                   # (B, T, DE_VOCAB)
        return logits, h


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src: (B, MAX_EN, EN_VOCAB)  trg: (B, MAX_DE, DE_VOCAB)
        h      = self.encoder(src)
        logits, _ = self.decoder(trg, h)
        return logits


encoder = Encoder(input_size=EN_VOCAB, hidden_size=HIDDEN)
decoder = Decoder(input_size=DE_VOCAB, hidden_size=HIDDEN, output_size=DE_VOCAB)
model   = Seq2Seq(encoder, decoder).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")
print(model)

# ─────────────────────────────────────────────
# 6. Loss (masked padding), optimizer
# ─────────────────────────────────────────────
PAD_IDX   = de_w2i[PAD]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ─────────────────────────────────────────────
# 7. Train / eval epoch helpers
# ─────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for enc_oh, dec_oh, dec_tgt in loader:
            enc_oh  = enc_oh.to(DEVICE)
            dec_oh  = dec_oh.to(DEVICE)
            dec_tgt = dec_tgt.to(DEVICE)

            logits = model(enc_oh, dec_oh)              # (B, T, DE_VOCAB)
            loss   = criterion(
                logits.reshape(-1, DE_VOCAB),
                dec_tgt.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n          += 1

    avg_loss   = total_loss / n
    perplexity = float(np.exp(avg_loss))
    return avg_loss, perplexity

# ─────────────────────────────────────────────
# 8. Greedy translate (inference)
# ─────────────────────────────────────────────
def translate(src_tokens, max_len=MAX_DE):
    model.eval()
    ei    = pad_seq(encode_en(src_tokens), MAX_EN, en_w2i[PAD])
    enc_oh = torch.nn.functional.one_hot(
        torch.tensor(ei, dtype=torch.long),
        num_classes=EN_VOCAB).float().unsqueeze(0).to(DEVICE)   # (1, MAX_EN, EN_VOCAB)

    with torch.no_grad():
        h   = model.encoder(enc_oh)                             # (1, 1, HIDDEN)
        tok = de_w2i[SOS]
        result = []
        for _ in range(max_len):
            tok_oh = torch.nn.functional.one_hot(
                torch.tensor([[tok]], dtype=torch.long),
                num_classes=DE_VOCAB).float().to(DEVICE)        # (1, 1, DE_VOCAB)
            logits, h = model.decoder(tok_oh, h)                # (1, 1, DE_VOCAB)
            tok = logits[0, 0].argmax().item()
            word = de_i2w[tok]
            if word == EOS:
                break
            result.append(word)
    return result

# ─────────────────────────────────────────────
# 9. BLEU scoring
# ─────────────────────────────────────────────
smoother = SmoothingFunction().method1

def compute_bleu(en_sents, de_refs, label="BLEU", n_samples=500):
    hypotheses, references = [], []
    n = min(n_samples, len(en_sents))
    for en, ref in zip(en_sents[:n], de_refs[:n]):
        hypotheses.append(translate(en))
        references.append([ref])
    score = corpus_bleu(references, hypotheses,
                        smoothing_function=smoother) * 100
    print(f"[{label}]  Corpus BLEU-4: {score:.2f}")
    return score

# ─────────────────────────────────────────────
# 10. Train
# ─────────────────────────────────────────────
EPOCHS      = 10
best_val    = float("inf")

print(f"\n── Training for {EPOCHS} epochs ──")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_ppl = run_epoch(train_dl, train=True)
    vl_loss, vl_ppl = run_epoch(val_dl,   train=False)
    print(f"Epoch {epoch:>2}/{EPOCHS} | "
          f"train loss {tr_loss:.4f} ppl {tr_ppl:.1f} | "
          f"val loss {vl_loss:.4f} ppl {vl_ppl:.1f}")
    if vl_loss < best_val:
        best_val = vl_loss
        torch.save(model.state_dict(), "rnn_multi30k_onehot.pt")
        print(f"           ✓ checkpoint saved (val_loss={best_val:.4f})")

# ─────────────────────────────────────────────
# 11. Final evaluation
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("rnn_multi30k_onehot.pt", map_location=DEVICE))

print("\n── Final Loss & Perplexity ──")
tr_loss, tr_ppl = run_epoch(train_dl, train=False)
vl_loss, vl_ppl = run_epoch(val_dl,   train=False)
print(f"[train]  loss: {tr_loss:.4f}  |  perplexity: {tr_ppl:.2f}")
print(f"[val  ]  loss: {vl_loss:.4f}  |  perplexity: {vl_ppl:.2f}")

print("\n── BLEU scores (greedy, 500 samples) ──")
compute_bleu(train_en, train_de, label="train", n_samples=500)
compute_bleu(val_en,   val_de,   label="val  ", n_samples=500)
compute_bleu(test_en,  test_de,  label="test ", n_samples=500)

# ─────────────────────────────────────────────
# 12. Demo translations
# ─────────────────────────────────────────────
print("\n── Example translations (EN → DE) ──")
demos = [
    "a dog is running through the grass .",
    "two people are sitting on a bench .",
    "a man is riding a bicycle .",
]
for src in demos:
    hyp = translate(src.split())
    print(f"  EN : {src}")
    print(f"  DE : {' '.join(hyp)}\n")