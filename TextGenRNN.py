import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# ── Load dataset ───────────────────────────────────────────────
dataset, info = tfds.load("tiny_shakespeare", with_info=True, as_supervised=False)
train_data = list(dataset["train"])[0]["text"].numpy().decode("utf-8")

# ── Vocabulary ─────────────────────────────────────────────────
chars     = sorted(set(train_data))
vocab_size = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = np.array(chars)

text_as_int = np.array([c2i[c] for c in train_data])

# ── Sequence dataset ───────────────────────────────────────────
SEQ_LEN    = 100
BATCH_SIZE = 64
BUFFER     = 10_000

char_ds = tf.data.Dataset.from_tensor_slices(text_as_int)
seq_ds  = char_ds.batch(SEQ_LEN + 1, drop_remainder=True)

def split_input_target(seq):
    # Targets stay as integer indices (for SparseCategoricalCrossentropy)
    # Inputs are one-hot encoded into (SEQ_LEN, vocab_size) vectors
    x = tf.one_hot(seq[:-1], vocab_size)
    y = seq[1:]
    return x, y

ds = (seq_ds
      .map(split_input_target)
      .shuffle(BUFFER)
      .batch(BATCH_SIZE, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE))

# ── Model ──────────────────────────────────────────────────────
HIDDEN_SIZE = 1024

# Input is already one-hot: (batch, seq_len, vocab_size) -- no Embedding needed
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(HIDDEN_SIZE, return_sequences=True, dropout=0.2,
                              input_shape=(None, vocab_size)),
    tf.keras.layers.Dense(vocab_size),
])

# ── Loss & perplexity ──────────────────────────────────────────
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

class Perplexity(tf.keras.metrics.Mean):
    """Perplexity = exp(mean cross-entropy loss). Logged each epoch."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = loss_fn(y_true, y_pred)
        super().update_state(tf.exp(loss), sample_weight=sample_weight)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=[Perplexity(name="perplexity")],
)
model.summary()

# ── Evaluate: returns (loss, perplexity) ──────────────────────
def evaluate(model, dataset, label="eval"):
    total_loss, n = 0.0, 0
    for x_batch, y_batch in dataset:
        logits      = model(x_batch, training=False)
        total_loss += loss_fn(y_batch, logits).numpy()
        n          += 1
    avg_loss   = total_loss / n
    perplexity = float(np.exp(avg_loss))
    print(f"[{label}]  loss: {avg_loss:.4f}  |  perplexity: {perplexity:.2f}")
    return avg_loss, perplexity

# ── Train ──────────────────────────────────────────────────────
EPOCHS = 10

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="shakespeare_rnn_tf.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="loss",
    verbose=1,
)

model.fit(ds, epochs=EPOCHS, callbacks=[checkpoint_cb])

# ── Final evaluation ───────────────────────────────────────────
print("\n── Final metrics ──")
evaluate(model, ds, label="train")

# ── Generate ───────────────────────────────────────────────────
def generate_text(model, seed: str, length: int = 500, temperature: float = 0.8) -> str:
    ids = [c2i[c] for c in seed if c in c2i]

    result = list(seed)
    # Build one-hot input from seed: (1, T, vocab_size)
    x = tf.one_hot([ids], vocab_size)

    for _ in range(length):
        logits  = model(x, training=False)               # (1, T, vocab_size)
        logits  = logits[:, -1, :] / temperature         # (1, vocab_size)
        next_id = tf.random.categorical(logits, 1)[0, 0].numpy()
        result.append(i2c[next_id])
        # Append next one-hot vector
        next_oh = tf.one_hot([[next_id]], vocab_size)    # (1, 1, vocab_size)
        x = tf.concat([x, next_oh], axis=1)

    return "".join(result)