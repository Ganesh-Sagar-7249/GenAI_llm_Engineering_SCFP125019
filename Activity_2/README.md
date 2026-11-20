# IMDB Sentiment Classification — RNN vs LSTM

## Overview

This project implements text sentiment classification on the IMDB movie review dataset using two recurrent neural network architectures:

- **Activity 1:** Simple RNN
- **Activity 2:** LSTM

Both models use the same dataset, train/test split, tokenization, padding/truncation, and DataLoader setup so that their performance can be compared fairly.

---

## Dataset

The project uses:

```text
IMDB Dataset.csv
```

For lab-speed training, the notebook loads **3,000 reviews**.

Each review contains:

- `review` — movie review text
- `sentiment` — `positive` or `negative`

Labels are converted to:

```text
negative → 0
positive → 1
```

### Train/Test Split

The 3,000 reviews are divided using an 80/20 split:

| Dataset | Samples |
|---|---:|
| Training | 2,400 |
| Testing | 600 |

A fixed random state and stratified split are used.

---

## Text Preprocessing

### Tokenization

The notebook uses:

```text
tiktoken
```

with:

```text
cl100k_base
```

The token IDs are shifted by `+1` so that:

```text
0 = padding
```

### Padding and Truncation

All reviews are converted to a fixed sequence length:

```text
MAX_LENGTH = 200
```

Therefore:

- Reviews longer than 200 tokens are truncated.
- Reviews shorter than 200 tokens are padded with `0`.

### PyTorch Tensors

The processed reviews and labels are converted to PyTorch tensors and loaded using:

```text
Batch Size = 32
```

---

# Activity 1 — RNN Text Classification

## Architecture

```text
Token IDs
    ↓
Embedding
    ↓
RNN
    ↓
Linear
    ↓
Sentiment
```

The RNN model uses:

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Hidden dimension | 128 |
| Sequence length | 200 |
| Batch size | 32 |

The embedding layer uses `padding_idx=0`.

## Training

The model is trained using:

- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam
- **Learning rate:** `0.0005`
- **Epochs:** 15
- **Gradient clipping:** `max_norm=1.0`

Gradient clipping is used to improve training stability.

## Evaluation

The RNN is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix
- Training Loss

The notebook also tests the trained model on new review sentences.

---

# Activity 2 — LSTM Text Classification

Activity 2 repeats the same text classification task using an LSTM.

The same:

- Dataset
- Train/test split
- Tokenizer
- Sequence length
- Padded tensors
- DataLoader
- Batch size
- Loss function
- Optimizer settings

are reused.

The main architectural change is:

```text
RNN → LSTM
```

## Architecture

```text
Token IDs
    ↓
Embedding
    ↓
LSTM
    ↓
Linear
    ↓
Sentiment
```

The LSTM model uses:

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Hidden dimension | 128 |
| Sequence length | 200 |
| Batch size | 32 |

## Training

The LSTM uses:

- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam
- **Learning rate:** `0.0005`
- **Epochs:** 15
- **Gradient clipping:** `max_norm=1.0`

## Evaluation

The LSTM is evaluated using the same metrics as the RNN:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix
- Training Loss

The same new review sentences are also tested using the LSTM.

---

# RNN vs LSTM Comparison

The notebook compares the two models using their test performance.

The final comparison includes:

| Metric | RNN | LSTM |
|---|---:|---:|
| Accuracy | Notebook result | Notebook result |
| Precision | Notebook result | Notebook result |
| Recall | Notebook result | Notebook result |
| F1 Score | Notebook result | Notebook result |

The notebook determines which model achieves the higher test accuracy.

> **Note:** The exact metric values depend on the actual training run and should be taken from the notebook output rather than hard-coded in this README.

---

## Prediction Pipeline

For a new review, the prediction process is:

```text
New Review
    ↓
tiktoken (cl100k_base)
    ↓
Token IDs
    ↓
Padding / Truncation
    ↓
Embedding
    ↓
RNN / LSTM
    ↓
Linear
    ↓
Sigmoid
    ↓
Probability
    ↓
Positive / Negative
```

A probability of:

```text
>= 0.5 → Positive
<  0.5 → Negative
```

---

## Requirements

Install the required Python packages with:

```bash
pip install torch pandas numpy scikit-learn matplotlib tiktoken
```

The notebook can be run in Jupyter Notebook, JupyterLab, or a compatible notebook environment.

---

## Project Structure

```text
.
├── lstm_imdb.ipynb
├── IMDB Dataset.csv
└── README.md
```

---

## How to Run

1. Place `IMDB Dataset.csv` in the same directory as the notebook.
2. Open `lstm_imdb.ipynb`.
3. Run the cells from top to bottom.
4. Complete Activity 1 to train and evaluate the RNN.
5. Continue with Activity 2 to train and evaluate the LSTM.
6. Compare the final RNN and LSTM metrics.

---

## Learning Objectives

This project demonstrates:

- Text preprocessing for neural networks
- Tokenization using `tiktoken`
- Sequence padding and truncation
- PyTorch tensor preparation
- Embedding layers
- RNN-based text classification
- LSTM-based text classification
- Binary sentiment classification
- Mini-batch training
- Gradient clipping
- Model evaluation
- Confusion matrix analysis
- RNN vs LSTM comparison

---

## Conclusion

The project provides a direct comparison between a basic RNN and an LSTM for IMDB sentiment classification while keeping the data preprocessing and training setup consistent.

The final notebook results should be used to determine which architecture performs better on the held-out test set.
