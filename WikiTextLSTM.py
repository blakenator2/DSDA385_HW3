import numpy as np
import tensorflow as tf
from datasets import load_dataset
from gensim.models import Word2Vec

print("Loading WikiText-103 …")
raw = load_dataset("wikitext", "wikitext-103-v1")

def extract_tokens(split):
    """Concatenate all non-empty lines and split into word tokens."""
    lines  = [row["text"] for row in raw[split] if row["text"].strip()]
    joined = " ".join(lines)
    return joined.lower().split()   # lowercase for cleaner vocab

train_tokens = extract_tokens("train")
val_tokens   = extract_tokens("validation")

print(f"Train tokens : {len(train_tokens):,}")
print(f"Val tokens   : {len(val_tokens):,}")

# ── 2. Train Word2Vec on training tokens ───────────────────────
EMBED_DIM  = 128
W2V_WINDOW = 5
W2V_MIN    = 5      # ignore very rare words

print("\nTraining Word2Vec …")
# Sentence-chunk the token list into ~30-token windows for Word2Vec
CHUNK = 30
sentences = [train_tokens[i: i + CHUNK]
             for i in range(0, len(train_tokens) - CHUNK, CHUNK)]

w2v = Word2Vec(sentences, vector_size=EMBED_DIM, window=W2V_WINDOW,
               min_count=W2V_MIN, workers=4, epochs=5, seed=42)

print(f"Word2Vec vocabulary: {len(w2v.wv):,} words")

# ── 3. Build integer vocabulary aligned to Word2Vec ────────────
SPECIAL = ["<PAD>", "<UNK>"]
vocab      = SPECIAL + list(w2v.wv.index_to_key)
w2i        = {w: i for i, w in enumerate(vocab)}
i2w        = {i: w for w, i in w2i.items()}
VOCAB_SIZE = len(vocab)

# Pretrained embedding matrix (PAD/UNK rows stay as zeros)
embedding_matrix = np.zeros((VOCAB_SIZE, EMBED_DIM), dtype=np.float32)
for word, idx in w2i.items():
    if word in w2v.wv:
        embedding_matrix[idx] = w2v.wv[word]

print(f"Final vocabulary size (with special tokens): {VOCAB_SIZE:,}")

# ── 4. Encode token lists → integer arrays ─────────────────────
UNK_ID = w2i["<UNK>"]

def encode(tokens):
    return np.array([w2i.get(t, UNK_ID) for t in tokens], dtype=np.int64)

train_enc = encode(train_tokens)
val_enc   = encode(val_tokens)

# ── 5. tf.data pipelines ───────────────────────────────────────
SEQ_LEN    = 64
BATCH_SIZE = 128
BUFFER     = 10_000

def make_dataset(encoded, shuffle=True):
    def split_input_target(seq):
        return seq[:-1], seq[1:]

    ds = (tf.data.Dataset.from_tensor_slices(encoded)
          .batch(SEQ_LEN + 1, drop_remainder=True)
          .map(split_input_target))
    if shuffle:
        ds = ds.shuffle(BUFFER)
    return ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_enc, shuffle=True)
val_ds   = make_dataset(val_enc,   shuffle=False)

# ── 6. Model ───────────────────────────────────────────────────
HIDDEN_SIZE = 512

embedding_layer = tf.keras.layers.Embedding(
    input_dim  = VOCAB_SIZE,
    output_dim = EMBED_DIM,
    weights    = [embedding_matrix],
    trainable  = False,             # frozen in phase 1
    name       = "w2v_embedding",
)

model = tf.keras.Sequential([
    embedding_layer,
    tf.keras.layers.LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.2),
    tf.keras.layers.Dense(VOCAB_SIZE),
], name="lstm_wikitext_w2v")

# ── 7. Loss, perplexity metric, compile ────────────────────────
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

class Perplexity(tf.keras.metrics.Mean):
    """Perplexity = exp(cross-entropy loss)."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = loss_fn(y_true, y_pred)
        super().update_state(tf.exp(loss), sample_weight=sample_weight)

def compile_model(trainable_embeddings=False):
    model.get_layer("w2v_embedding").trainable = trainable_embeddings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=[Perplexity(name="perplexity")],
    )

compile_model(trainable_embeddings=False)
model.summary()

# ── 8. Evaluate helper ─────────────────────────────────────────
def evaluate(model, dataset, label="eval"):
    total, n = 0.0, 0
    for x, y in dataset:
        total += loss_fn(y, model(x, training=False)).numpy()
        n     += 1
    avg_loss   = total / n
    perplexity = float(np.exp(avg_loss))
    print(f"[{label}]  loss: {avg_loss:.4f}  |  perplexity: {perplexity:.2f}")
    return avg_loss, perplexity

# ── 9. Train ───────────────────────────────────────────────────
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath          = "lstm_wikitext_w2v.weights.h5",
    save_weights_only = True,
    save_best_only    = True,
    monitor           = "val_loss",
    verbose           = 1,
)

# Phase 1: frozen embeddings — let LSTM adapt quickly
print("\n── Phase 1: frozen Word2Vec embeddings (5 epochs) ──")
model.fit(train_ds, validation_data=val_ds,
          epochs=5, callbacks=[checkpoint_cb])

# Phase 2: unfreeze and fine-tune end-to-end
print("\n── Phase 2: fine-tuning embeddings + LSTM (25 epochs) ──")
compile_model(trainable_embeddings=True)
model.fit(train_ds, validation_data=val_ds,
          epochs=10, callbacks=[checkpoint_cb])

# ── 10. Final evaluation ───────────────────────────────────────
print("\n── Final metrics ──")
evaluate(model, train_ds, label="train")
evaluate(model, val_ds,   label="val  ")

# ── 11. Generate ──────────────────────────────────────────────
def generate_text_wiki(seed: str, length: int = 100,
                  temperature: float = 0.8) -> str:
    seed_tokens = seed.lower().split()
    ids = [w2i.get(t, UNK_ID) for t in seed_tokens]
    x   = tf.expand_dims(ids, 0)                        # (1, T)

    result = list(seed_tokens)
    for _ in range(length):
        logits  = model(x, training=False)              # (1, T, V)
        logits  = logits[:, -1, :] / temperature        # (1, V)
        next_id = tf.random.categorical(logits, 1)[0, 0].numpy()
        result.append(i2w[next_id])
        x = tf.concat([x, [[next_id]]], axis=1)

    return " ".join(result)


