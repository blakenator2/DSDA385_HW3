from TextGenRNN import generate_text
from WikiTextLSTM import generate_text_wiki
from LSTMTranslate import run
from RNNTranslate import runRNN


print(generate_text(seed="ROMEO:\n", length=500)) 

print(generate_text_wiki(seed="the history of science", length=100)) 

run()

runRNN()