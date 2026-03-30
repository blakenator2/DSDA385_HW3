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

### **RNN**


|Dataset | Optimizer | Learning Rate | Image Size |
|--------|-----------|-----------|----------------|


---

### **LTSM**


|Dataset | Optimizer | Learning Rate | Image Size |
|--------|-----------|-----------|----------------|


---

## **Training details**

|Dataset|Train Split | Test | Validation Splits | Epochs| Batch size|
|-------|------------|------|-------------------|-------|-----------|


---

## **Results**

| Dataset | Model   | mAP@0.5 | Precision | Recall | Training Time | Inference Speed |
|---------|---------|---------|-----------|--------|---------------|-----------------|

---

## **Discussion and conclusion**



---