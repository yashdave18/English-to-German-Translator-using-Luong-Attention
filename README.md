# 🇩🇪 English to German Neural Machine Translator

A sequence-to-sequence neural machine translation model with **Luong Attention** built using TensorFlow/Keras, trained on 220k+ English-German sentence pairs.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.6470 |
| BLEU-2 | 0.5253 |
| BLEU-3 | 0.4412 |
| BLEU-4 | 0.3678 |
| Val Accuracy | ~90.5% |

---

## 🏗️ Architecture

```
English Sentence
      ↓
  Encoder (Embedding + LSTM)
      ↓
  All Hidden States
      ↓
Luong Attention (Dot Product)
      ↓
  Context Vector
      ↓
  Decoder (Embedding + LSTM)
      ↓
  TimeDistributed Dense
      ↓
German Sentence
```

- **Encoder** — Embedding (256) + LSTM (512 units, return_sequences=True)
- **Attention** — Luong dot-product attention over all encoder hidden states
- **Decoder** — Embedding (256) + LSTM (512 units) + context concatenation
- **Output** — TimeDistributed Dense over German vocab (19,568 words)
- **Total params** — 31.1M

---

## 📁 Dataset

[Tatoeba English-German dataset](http://www.manythings.org/anki/) — 221k sentence pairs
- Filtered to max sequence length of 20 tokens
- Final pairs after filtering: 220,353
- Train/Val split: 90/10

---

## 🚀 Usage

### Clone the repository

```bash
git clone https://github.com/yashdave18/English-to-German-Translator-using-Luong-Attention.git
cd English-to-German-Translator-using-Luong-Attention
```

### Install dependencies

```bash
pip install tensorflow numpy pandas scikit-learn nltk matplotlib
```




## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim | 256 |
| LSTM units | 512 |
| Batch size | 64 |
| Max sequence length | 20 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Early stopping patience | 3 |

