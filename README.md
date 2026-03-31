# DSDA385_HW3
## **Introduction**
In this homework I compared two machine translation and two text generation models. Both
had one LSTM and one RNN.
---

## **Dataset description**
The Little Shakespeare dataset contains around 40,000 words from Shakespeare, making it a good dataset for
small scale text generation.

The WikiText dataset contains around 103 million tokens, making it one of the most popular dataests for 
English text generation.

The Multi30k dataset contains 30,000 sentences that have both their English and German translations. This
is a small but mighty dataset for small scale machine translation tasks.

---

## **Word Encoding Types**
Both RNN's are using one-hot encoding while both LSTM's are using pre-trained Word2Vec embeddings

### **One-hot**
One-hot encodings allow for machine learning models to learn where specific words show up in sentences
as the target word will be a 1 while the others will be 0. This is a much easier way to do NLP, however,
it is much more shallow and does not learn deeper embeddings of words. This leads to poorer but lighter
models.

### **Word2Vec**
Word2Vec is a collection of around 3 billion english words that have 300 word vectors created by Google.
Each of these vectors represents semantic meaning in a multidimensional space. Words that have a smaller
cosine distance will have similar meaning than those with larger cosine distances. Word2Vec does much better
at capturing deep meaning for NLP, but it does create larger, more expensive, models

---

## **Model explanation**
Both models are using the Adam optimizer and all models are being tested on 10 epochs for fairness.
Both translate models have a batch size of 64.
The WikiText model has a batch size of 128, the Shakespeare model has a batch size of 64.

The translate RNN is a simple encoder-decoder model with one hot embedding. This model has one hidden
layer of 512 and a dropout of 0.3 in order to avoid overfitting. This model trains to minimize the perplexity score.

The translate LSTM is a simple encoder-decoder model with word2vec embeddings. This means that the model needs an embedding
layer in order to translate the embeddings into something that can be learned. This model has a hidden layer of 256 with a 
dropout of 0.3 in order to avoid overfitting. This model trains to minimize perplexity.

The WikiText LSTM is a 3 layer model with word2vec embeddings. This means that the model has an embedding layer that 
translates these embeddings into something that can be learned. This model has a hidden layer of 512 with a dropout of
0.2. This model trains to minimize perplexity.

The WikiText LSTM is a 3 layer model with one hot embedding. This model has a hidden layer of 1024 with a dropout of
0.2. This model trains to minimize perplexity.

### **RNN**
|Dataset | BLEU | Loss | Perplexity |
|--------|-----------|-----------|----------------|
|**Shakespeare**| N/A | 1.7888  | 5.98 |
|**Multi30k**|0.68| 4.3332 |76.19|

---

### **LTSM**
|Dataset | BLEU | Loss | Perplexity |
|--------|-----------|-----------|----------------|
|**WikiText**| N/A | 4.6874 | 73.6844 |
|**Multi30k**|13.56| 3.6690 |40.6524|

---
## **Results**

### **LSTM WikiText**

the history of science in the <UNK> was based on the the work
of <UNK> <UNK> and the period . the the of <UNK> was the
first <UNK> of the the <UNK> <UNK> in <UNK> . <UNK> the the
was <UNK> <UNK> the of in a <UNK> <UNK> the . the <UNK>
<UNK> of the <UNK> was <UNK> the the <UNK> . in the <UNK>
<UNK> <UNK> the the of <UNK> <UNK> was <UNK> <UNK> . the the
<UNK> <UNK> <UNK> of the <UNK> <UNK> <UNK> the .

### **LSTM Multi30k**
── Example translations (EN → DE) ──
  EN : a dog is running through the grass .
  DE : ein hund läuft durch den schnee.

  EN : two people are sitting on a bench .
  DE : zwei personen sitzen auf einer bank.

  EN : a man is riding a bicycle .
  DE : ein mann fährt auf einem fahrrad.

### **RNN Multi30k**
  EN : a dog is running through the grass .
  DE : ein mann in einem roten hemd und mit einem weißen hut und einem schwarzen oberteil und mit einem weißen hut und einem schwarzen oberteil und mit einem weißen hut und einem schwarzen oberteil und mit einem weißen hut und einem schwarzen oberteil

  EN : two people are sitting on a bench .
  DE : ein mann in einem roten hemd und mit einem weißen hut und einem schwarzen oberteil und mit einem weißen hut und einem schwarzen oberteil und mit einem weißen hut und einem schwarzen oberteil und mit einem weißen hut und einem schwarzen oberteil

  EN : a man is riding a bicycle .
  DE : ein mann in einem roten hemd und mit einem weißen hut und einem schwarzen oberteil und mit einem weißen hut und einem schwarzen oberteil und mit einem weißen hut und einem schwarzen oberteil und mit einem weißen hut und einem schwarzen oberteil


### **RNN Shakespeare**

ROMEO:
Now, prove well ant deenon: that with be soil'g s and beirgsedion;
Ialaw, there he hear ia is the aupe of this wousee so say the respland
To brthe son.

LLONTES:
Not bo thas rabet nobl atte
That the dond wat would be sone make our, wor upon thet to ungord to the beall

BRUTUS:
Aup to my erther; be now en thy weea excuseda aighes of your hare yel ant fellite is ealling and heares for howr before, te the wauth'n nat eas bo thy hears all the roulage to les and slat of toe befthew far she had can ou

---

## **Comparison**

Both text generation models fell very quickly after predicting far out into the future. The LSTM
model performed better insofar as it generated actual english words, not just english looking words.
However, the LSTM guessed more and more proper nouns as the prediction progresses, where at least the 
RNN model created english like words that almost follow proper rules.

The LSTM translation model was just incredible compared to the RNN tranlsation model. There is no way
to really compare the output of the models, the LSTM blows the RNN out of the water. LSTMs seem to be 
much better in all forms of text generation, at least for smaller models.

---

## **Discussion and conclusion**

### **Challenges Faced During Implementation**
Many of my challenges came from the sheer size and amount of time these models took to train. Specifically,
the word2vec models. These embeddings were massive and took forever, even on the strong computers, to train
a model.

### **Limitations**
My computing power and simplicity of models limited the strength of the outputs. Simple LSTM models without
attention can only go so far. Also, with the few amount of parameters that I am able to utilize since I 
have limited computing power also resulted in less powerful models.

### **Possible Future Work**
If I were to perform a semester long project on any of these, I would really enjoy improving the translation
models. Translation, language overall, has very strict rules that I believe, with enough data and time, can
be learned very effectively by computers. I would spend more time on improving those models in order to create
something that could have some real world applications

---