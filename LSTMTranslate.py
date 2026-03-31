import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from gensim.models import Word2Vec
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
# 2. Train Word2Vec — one model per language
# ─────────────────────────────────────────────
EMBED_DIM = 64

print("\nTraining Word2Vec (EN) …")
w2v_en = Word2Vec(train_en, vector_size=EMBED_DIM, window=5,
                  min_count=1, workers=4, epochs=10, seed=42)

print("Training Word2Vec (DE) …")
w2v_de = Word2Vec(train_de, vector_size=EMBED_DIM, window=5,
                  min_count=1, workers=4, epochs=10, seed=42)

# ─────────────────────────────────────────────
# 3. Vocabulary helpers
# ─────────────────────────────────────────────
def build_vocab(w2v_model):
    specials   = [PAD, UNK, SOS, EOS]
    vocab      = specials + list(w2v_model.wv.index_to_key)
    w2i        = {w: i for i, w in enumerate(vocab)}
    i2w        = {i: w for w, i in w2i.items()}
    vocab_size = len(vocab)
    emb_matrix = np.zeros((vocab_size, EMBED_DIM), dtype=np.float32)
    for word, idx in w2i.items():
        if word in w2v_model.wv:
            emb_matrix[idx] = w2v_model.wv[word]
    return w2i, i2w, vocab_size, emb_matrix

en_w2i, en_i2w, EN_VOCAB, en_emb = build_vocab(w2v_en)
de_w2i, de_i2w, DE_VOCAB, de_emb = build_vocab(w2v_de)

print(f"EN vocab: {EN_VOCAB:,}  |  DE vocab: {DE_VOCAB:,}")

# ─────────────────────────────────────────────
# 4. Encode & pad sentence pairs
# ─────────────────────────────────────────────
MAX_EN = 40
MAX_DE = 42

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
# 5. PyTorch Dataset & DataLoader
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
        self.enc_in  = torch.tensor(self.enc_in,  dtype=torch.long)
        self.dec_in  = torch.tensor(self.dec_in,  dtype=torch.long)
        self.dec_tgt = torch.tensor(self.dec_tgt, dtype=torch.long)

    def __len__(self):
        return len(self.enc_in)

    def __getitem__(self, idx):
        return self.enc_in[idx], self.dec_in[idx], self.dec_tgt[idx]

BATCH_SIZE = 64

train_ds = TranslationDataset(train_en, train_de)
val_ds   = TranslationDataset(val_en,   val_de)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ─────────────────────────────────────────────
# 6. Seq2Seq Model
# ─────────────────────────────────────────────
HIDDEN = 256

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, emb_matrix, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(emb_matrix), requires_grad=False)  # frozen phase 1
        self.lstm    = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, MAX_EN)
        emb = self.dropout(self.embedding(x))          # (B, T, E)
        _, (h, c) = self.lstm(emb)                     # h,c: (1, B, H)
        return h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, emb_matrix, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(emb_matrix), requires_grad=False)  # frozen phase 1
        self.lstm    = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h, c):
        # x: (B, MAX_DE)
        emb = self.dropout(self.embedding(x))          # (B, T, E)
        out, (h, c) = self.lstm(emb, (h, c))           # (B, T, H)
        logits = self.fc(out)                          # (B, T, V)
        return logits, h, c


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        h, c = self.encoder(src)
        logits, _, _ = self.decoder(trg, h, c)
        return logits

    def set_embeddings_trainable(self, trainable: bool):
        for param in self.encoder.embedding.parameters():
            param.requires_grad = trainable
        for param in self.decoder.embedding.parameters():
            param.requires_grad = trainable


encoder = Encoder(EN_VOCAB, EMBED_DIM, HIDDEN, en_emb)
decoder = Decoder(DE_VOCAB, EMBED_DIM, HIDDEN, de_emb)
model   = Seq2Seq(encoder, decoder).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")
print(model)

# ─────────────────────────────────────────────
# 7. Loss (masked), perplexity, optimizer
# ─────────────────────────────────────────────
PAD_IDX = de_w2i[PAD]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def masked_loss_and_ppl(logits, targets):
    # logits: (B, T, V)  targets: (B, T)
    loss = criterion(logits.reshape(-1, DE_VOCAB), targets.reshape(-1))
    return loss, float(torch.exp(loss))

def make_optimizer(trainable_emb=False):
    model.set_embeddings_trainable(trainable_emb)
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# ─────────────────────────────────────────────
# 8. Train / eval epoch helpers
# ─────────────────────────────────────────────
def run_epoch(loader, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for enc_in, dec_in, dec_tgt in loader:
            enc_in  = enc_in.to(DEVICE)
            dec_in  = dec_in.to(DEVICE)
            dec_tgt = dec_tgt.to(DEVICE)
            logits  = model(enc_in, dec_in)             # (B, T, V)
            loss, _ = masked_loss_and_ppl(logits, dec_tgt)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            n          += 1
    avg_loss = total_loss / n
    return avg_loss, float(np.exp(avg_loss))

# ─────────────────────────────────────────────
# 9. Greedy translate (inference)
# ─────────────────────────────────────────────
def translate(src_tokens, max_len=MAX_DE):
    model.eval()
    ei  = torch.tensor(
        [pad_seq(encode_en(src_tokens), MAX_EN, en_w2i[PAD])],
        dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        h, c = model.encoder(ei)
        tok  = torch.tensor([[de_w2i[SOS]]], dtype=torch.long, device=DEVICE)
        result = []
        for _ in range(max_len):
            logits, h, c = model.decoder(tok, h, c)    # (1, 1, V)
            next_id = logits[0, 0].argmax().item()
            word    = de_i2w[next_id]
            if word == EOS:
                break
            result.append(word)
            tok = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
    return result

# ─────────────────────────────────────────────
# 10. BLEU scoring
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
# 11. Train (two-phase)
# ─────────────────────────────────────────────
best_val_loss = float("inf")

def train_phase(epochs, trainable_emb, label):
    global best_val_loss
    optimizer = make_optimizer(trainable_emb)
    print(f"\n── {label} ──")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_ppl = run_epoch(train_dl, optimizer)
        vl_loss, vl_ppl = run_epoch(val_dl)
        print(f"  Epoch {epoch:>2}/{epochs} | "
              f"train loss {tr_loss:.4f} ppl {tr_ppl:.1f} | "
              f"val loss {vl_loss:.4f} ppl {vl_ppl:.1f}")
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), "lstm_multi30k_w2v.pt")
            print(f"           ✓ checkpoint saved (val_loss={best_val_loss:.4f})")

train_phase(5,  trainable_emb=False, label="Phase 1: frozen embeddings (5 epochs)")
train_phase(10, trainable_emb=True,  label="Phase 2: fine-tuning embeddings + LSTM (10 epochs)")

# ─────────────────────────────────────────────
# 12. Final evaluation
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("lstm_multi30k_w2v.pt", map_location=DEVICE))

print("\n── Final Loss & Perplexity ──")
tr_loss, tr_ppl = run_epoch(train_dl)
vl_loss, vl_ppl = run_epoch(val_dl)
print(f"[train]  loss: {tr_loss:.4f}  |  perplexity: {tr_ppl:.2f}")
print(f"[val  ]  loss: {vl_loss:.4f}  |  perplexity: {vl_ppl:.2f}")

print("\n── BLEU scores (greedy, 500 samples) ──")
compute_bleu(train_en, train_de, label="train", n_samples=500)
compute_bleu(val_en,   val_de,   label="val  ", n_samples=500)
compute_bleu(test_en,  test_de,  label="test ", n_samples=500)

# ─────────────────────────────────────────────
# 13. Demo translations
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