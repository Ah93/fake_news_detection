# 📰 Fake News Detection using Transformer-BiGRU

This project implements a **Fake News Detection** system using a hybrid deep learning model combining a **Transformer** and a **Bidirectional GRU** architecture. The model is trained to classify political and news-related statements as true or false using both the **LIAR dataset** and the **ISOT Fake News Dataset**.

---

## 📌 Project Overview

Fake news has become a significant concern in today's media landscape. This project aims to identify misinformation in short political and general news statements using deep learning techniques. It leverages pre-trained word embeddings and advanced sequence models to improve detection accuracy.

---

## 🧠 Model Architecture

The model architecture includes:

- **Embedding Layer**: Initialized with pre-trained **GloVe (Global Vectors for Word Representation)** embeddings.
- **Transformer Encoder Block**: Captures global contextual relationships.
- **Bidirectional GRU Layer**: Learns sequential dependencies from both forward and backward directions.
- **Dropout Layers**: Prevent overfitting.
- **Fully Connected Output Layer**: Produces classification logits for binary classification.

---

## 📊 Datasets Used

### 1. **LIAR Dataset**
- Source: [POLITIFACT](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- 12,836 labeled short political statements
- Labels: true, mostly-true, half-true, barely-true, false, pants-fire
- Simplified to **binary labels**: Real vs. Fake
- Used for training, validation, and testing

### 2. **ISOT Fake News Dataset**
- Source: [University of Victoria ISOT](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php)
- 44,898 news articles:
  - 21,417 real news (from Reuters)
  - 23,481 fake news (from unreliable sources)
- Used for general news-based fake vs real classification

---

## ⚙️ Preprocessing

- Tokenization using `keras.preprocessing.text.Tokenizer`
- Padding sequences to a fixed length (e.g., 100 tokens)
- Simplifying multi-class labels (in LIAR) to binary
- GloVe Embedding loading (100-dimensional vectors)

---

## 🏃‍♂️ Training

- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Metrics: Accuracy
- Epochs: 10–20 (adjustable)
- Validation accuracy typically exceeds 85%+

---

## 📈 Evaluation

Evaluation is performed using test sets from both datasets. Metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
Make sure the GloVe embeddings (glove.6B.100d.txt) are downloaded and accessible in your project path.
