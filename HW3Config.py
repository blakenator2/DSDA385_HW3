# from TextGenRNN import generate_text
# from WikiTextLSTM import generate_text_wiki
from LSTMTranslate import run
from RNNTranslate import runRNN
import tensorflow as tf

# model = tf.keras.models.load_model('shakespeare_rnn_tf.weights.h5')
# print(generate_text(model, seed="ROMEO:\n", length=500)) # loss: 1.7888  |  perplexity: 5.98

# print(generate_text_wiki(seed="the history of science", length=100)) 

run() # loss: 3.6690 - perplexity: 40.6524
# ── BLEU scores (greedy, 500 samples) ──
# [train]  Corpus BLEU-4: 11.97
# [val  ]  Corpus BLEU-4: 13.56
runRNN()