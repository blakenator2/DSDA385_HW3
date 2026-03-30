import numpy as np
import tensorflow as tf
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)

def runRNN():
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

    def make_arrays(en_sents, de_sents):
        enc_in, dec_in, dec_tgt = [], [], []
        for en, de in zip(en_sents, de_sents):
            ei  = pad_seq(encode_en(en), MAX_EN, en_w2i[PAD])
            de_ = encode_de(de)
            di  = pad_seq(de_[:-1], MAX_DE, de_w2i[PAD])   # SOS … (teacher forcing)
            dt  = pad_seq(de_[1:],  MAX_DE, de_w2i[PAD])   # … EOS (target)
            enc_in.append(ei); dec_in.append(di); dec_tgt.append(dt)
        return (np.array(enc_in,  dtype=np.int32),
                np.array(dec_in,  dtype=np.int32),
                np.array(dec_tgt, dtype=np.int32))

    train_ei, train_di, train_dt = make_arrays(train_en, train_de)
    val_ei,   val_di,   val_dt   = make_arrays(val_en,   val_de)

    # ─────────────────────────────────────────────
    # 4. tf.data — one-hot encode inputs inline
    # ─────────────────────────────────────────────
    # One-hot is applied inside the pipeline so the large float tensors
    # are never stored in memory all at once.
    BATCH_SIZE = 64
    BUFFER     = 5_000

    def onehot_inputs(inputs, targets):
        enc_in, dec_in = inputs
        enc_oh = tf.one_hot(enc_in, EN_VOCAB)   # (B, MAX_EN, EN_VOCAB)
        dec_oh = tf.one_hot(dec_in, DE_VOCAB)   # (B, MAX_DE, DE_VOCAB)
        return (enc_oh, dec_oh), targets

    def make_dataset(ei, di, dt, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(((ei, di), dt))
        if shuffle:
            ds = ds.shuffle(BUFFER)
        return (ds.batch(BATCH_SIZE, drop_remainder=False)
                .map(onehot_inputs, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    train_ds = make_dataset(train_ei, train_di, train_dt, shuffle=True)
    val_ds   = make_dataset(val_ei,   val_di,   val_dt,   shuffle=False)

    # ─────────────────────────────────────────────
    # 5. Seq2Seq model — SimpleRNN encoder-decoder
    #    Input is one-hot so no Embedding layer needed
    # ─────────────────────────────────────────────
    HIDDEN = 512

    # Encoder: reads one-hot EN vectors → hidden state
    enc_input       = tf.keras.Input(shape=(MAX_EN, EN_VOCAB), name="encoder_input")
    enc_drop        = tf.keras.layers.Dropout(0.3)(enc_input)
    _, enc_h        = tf.keras.layers.SimpleRNN(HIDDEN, return_state=True,
                        name="encoder_rnn")(enc_drop)

    # Decoder: reads one-hot DE vectors, initialised from encoder state
    dec_input       = tf.keras.Input(shape=(MAX_DE, DE_VOCAB), name="decoder_input")
    dec_drop        = tf.keras.layers.Dropout(0.3)(dec_input)
    dec_out, _      = tf.keras.layers.SimpleRNN(HIDDEN, return_sequences=True,
                        return_state=True, name="decoder_rnn")(
                        dec_drop, initial_state=enc_h)
    logits          = tf.keras.layers.Dense(DE_VOCAB,
                        name="output_projection")(dec_out)

    model = tf.keras.Model(inputs=[enc_input, dec_input],
                        outputs=logits, name="seq2seq_rnn_onehot")

    # ─────────────────────────────────────────────
    # 6. Masked loss, perplexity, compile
    # ─────────────────────────────────────────────
    _loss_base = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")

    def masked_loss(y_true, y_pred):
        loss = _loss_base(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, de_w2i[PAD]), loss.dtype)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    class Perplexity(tf.keras.metrics.Mean):
        def update_state(self, y_true, y_pred, sample_weight=None):
            super().update_state(tf.exp(masked_loss(y_true, y_pred)),
                                sample_weight=sample_weight)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-3),
        loss      = masked_loss,
        metrics   = [Perplexity(name="perplexity")],
    )
    model.summary()

    # ─────────────────────────────────────────────
    # 7. Inference encoder / decoder
    # ─────────────────────────────────────────────
    enc_model = tf.keras.Model(inputs=enc_input,
                            outputs=enc_h,
                            name="inference_encoder")

    dec_h_in   = tf.keras.Input(shape=(HIDDEN,), name="dec_h_in")
    dec_tok_in = tf.keras.Input(shape=(1, DE_VOCAB), name="dec_token_onehot")

    _out, _h_out = model.get_layer("decoder_rnn")(
        dec_tok_in, initial_state=dec_h_in)
    _logits = model.get_layer("output_projection")(_out)

    dec_model = tf.keras.Model(
        inputs  = [dec_tok_in, dec_h_in],
        outputs = [_logits, _h_out],
        name    = "inference_decoder")


    def translate(src_tokens, max_len=MAX_DE):
        """Greedy decode: src_tokens is a list of English words."""
        ei  = pad_seq(encode_en(src_tokens), MAX_EN, en_w2i[PAD])
        enc_oh = tf.one_hot([ei], EN_VOCAB)                 # (1, MAX_EN, EN_VOCAB)
        h   = enc_model.predict(enc_oh, verbose=0)          # (1, HIDDEN)

        # Seed the decoder with SOS as a one-hot vector
        tok = tf.one_hot([[de_w2i[SOS]]], DE_VOCAB)         # (1, 1, DE_VOCAB)
        result = []
        for _ in range(max_len):
            logits, h = dec_model.predict([tok, h], verbose=0)
            next_id   = int(np.argmax(logits[0]))
            word      = de_i2w[next_id]
            if word == EOS:
                break
            result.append(word)
            tok = tf.one_hot([[next_id]], DE_VOCAB)         # (1, 1, DE_VOCAB)
        return result

    # ─────────────────────────────────────────────
    # 8. BLEU scoring
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
    # 9. Train
    # ─────────────────────────────────────────────
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath          = "rnn_multi30k_onehot.weights.h5",
        save_weights_only = True,
        save_best_only    = True,
        monitor           = "val_loss",
        verbose           = 1,
    )

    EPOCHS = 10
    print(f"\n── Training for {EPOCHS} epochs ──")
    model.fit(train_ds, validation_data=val_ds,
            epochs=EPOCHS, callbacks=[checkpoint_cb])

    # ─────────────────────────────────────────────
    # 10. Final evaluation
    # ─────────────────────────────────────────────
    print("\n── Loss & Perplexity ──")
    print("Train:", end=" "); model.evaluate(train_ds, verbose=1)
    print("Val  :", end=" "); model.evaluate(val_ds,   verbose=1)

    print("\n── BLEU scores (greedy, 500 samples) ──")
    compute_bleu(train_en, train_de, label="train", n_samples=500)
    compute_bleu(val_en,   val_de,   label="val  ", n_samples=500)
    compute_bleu(test_en,  test_de,  label="test ", n_samples=500)

    # ─────────────────────────────────────────────
    # 11. Demo translations
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