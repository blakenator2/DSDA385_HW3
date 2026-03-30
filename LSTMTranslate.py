import numpy as np
import tensorflow as tf
from datasets import load_dataset
from gensim.models import Word2Vec
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)

def run():
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
            ei  = pad_seq(encode_en(en),  MAX_EN, en_w2i[PAD])
            de_ = encode_de(de)
            di  = pad_seq(de_[:-1], MAX_DE, de_w2i[PAD])  # SOS … (teacher forcing)
            dt  = pad_seq(de_[1:],  MAX_DE, de_w2i[PAD])  # … EOS (target)
            enc_in.append(ei); dec_in.append(di); dec_tgt.append(dt)
        return (np.array(enc_in,  dtype=np.int32),
                np.array(dec_in,  dtype=np.int32),
                np.array(dec_tgt, dtype=np.int32))

    train_ei, train_di, train_dt = make_arrays(train_en, train_de)
    val_ei,   val_di,   val_dt   = make_arrays(val_en,   val_de)

    # ─────────────────────────────────────────────
    # 5. tf.data pipelines
    # ─────────────────────────────────────────────
    BATCH_SIZE = 64
    BUFFER     = 5_000

    def make_dataset(ei, di, dt, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(((ei, di), dt))
        if shuffle:
            ds = ds.shuffle(BUFFER)
        return ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_ei, train_di, train_dt, shuffle=True)
    val_ds   = make_dataset(val_ei,   val_di,   val_dt,   shuffle=False)

    # ─────────────────────────────────────────────
    # 6. Seq2Seq model (encoder-decoder LSTM)
    # ─────────────────────────────────────────────
    HIDDEN = 256

    # Encoder
    enc_input        = tf.keras.Input(shape=(MAX_EN,), name="encoder_input")
    enc_emb          = tf.keras.layers.Embedding(EN_VOCAB, EMBED_DIM,
                        weights=[en_emb], trainable=False,
                        name="enc_embedding")(enc_input)
    enc_drop         = tf.keras.layers.Dropout(0.3)(enc_emb)
    _, enc_h, enc_c  = tf.keras.layers.LSTM(HIDDEN, return_state=True,
                        name="encoder_lstm")(enc_drop)

    # Decoder (teacher forcing during training)
    dec_input        = tf.keras.Input(shape=(MAX_DE,), name="decoder_input")
    dec_emb          = tf.keras.layers.Embedding(DE_VOCAB, EMBED_DIM,
                        weights=[de_emb], trainable=False,
                        name="dec_embedding")(dec_input)
    dec_drop         = tf.keras.layers.Dropout(0.3)(dec_emb)
    dec_out, _, _    = tf.keras.layers.LSTM(HIDDEN, return_sequences=True,
                        return_state=True, name="decoder_lstm")(
                        dec_drop, initial_state=[enc_h, enc_c])
    logits           = tf.keras.layers.Dense(DE_VOCAB,
                        name="output_projection")(dec_out)

    model = tf.keras.Model(inputs=[enc_input, dec_input],
                        outputs=logits, name="seq2seq_lstm")

    # ─────────────────────────────────────────────
    # 7. Masked loss, perplexity, compile
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

    def compile_model(trainable_emb=False):
        model.get_layer("enc_embedding").trainable = trainable_emb
        model.get_layer("dec_embedding").trainable = trainable_emb
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss=masked_loss,
                    metrics=[Perplexity(name="perplexity")])

    compile_model(trainable_emb=False)
    model.summary()

    # ─────────────────────────────────────────────
    # 8. Inference encoder / decoder
    # ─────────────────────────────────────────────
    enc_model = tf.keras.Model(inputs=enc_input,
                            outputs=[enc_h, enc_c],
                            name="inference_encoder")

    dec_h_in   = tf.keras.Input(shape=(HIDDEN,), name="dec_h_in")
    dec_c_in   = tf.keras.Input(shape=(HIDDEN,), name="dec_c_in")
    dec_tok_in = tf.keras.Input(shape=(1,),      name="dec_token")

    _emb   = model.get_layer("dec_embedding")(dec_tok_in)
    _out, _h_out, _c_out = model.get_layer("decoder_lstm")(
        _emb, initial_state=[dec_h_in, dec_c_in])
    _logits = model.get_layer("output_projection")(_out)

    dec_model = tf.keras.Model(
        inputs  = [dec_tok_in, dec_h_in, dec_c_in],
        outputs = [_logits, _h_out, _c_out],
        name    = "inference_decoder")


    def translate(src_tokens, max_len=MAX_DE):
        """Greedy decode a list of English word tokens → German word list."""
        ei  = np.array([pad_seq(encode_en(src_tokens), MAX_EN, en_w2i[PAD])])
        h, c = enc_model.predict(ei, verbose=0)
        tok  = np.array([[de_w2i[SOS]]])
        result = []
        for _ in range(max_len):
            logits, h, c = dec_model.predict([tok, h, c], verbose=0)
            next_id = int(np.argmax(logits[0, 0]))
            word    = de_i2w[next_id]
            if word == EOS:
                break
            result.append(word)
            tok = np.array([[next_id]])
        return result

    # ─────────────────────────────────────────────
    # 9. BLEU scoring
    # ─────────────────────────────────────────────
    smoother = SmoothingFunction().method1

    def compute_bleu(en_sents, de_refs, label="BLEU", n_samples=500):
        """Corpus BLEU over the first n_samples pairs."""
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
    # 10. Train (two-phase)
    # ─────────────────────────────────────────────
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath          = "lstm_multi30k_w2v.weights.h5",
        save_weights_only = True,
        save_best_only    = True,
        monitor           = "val_loss",
        verbose           = 1,
    )

    print("\n── Phase 1: frozen embeddings (5 epochs) ──")
    model.fit(train_ds, validation_data=val_ds,
            epochs=5, callbacks=[checkpoint_cb])

    print("\n── Phase 2: fine-tuning embeddings + LSTM (25 epochs) ──")
    compile_model(trainable_emb=True)
    model.fit(train_ds, validation_data=val_ds,
            epochs=10, callbacks=[checkpoint_cb])

    # ─────────────────────────────────────────────
    # 11. Final evaluation
    # ─────────────────────────────────────────────
    print("\n── Loss & Perplexity ──")
    print("Train:", end=" "); model.evaluate(train_ds, verbose=1)
    print("Val  :", end=" "); model.evaluate(val_ds,   verbose=1)

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