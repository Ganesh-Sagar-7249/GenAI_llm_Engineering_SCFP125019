# IMDB Sentiment Classification using RNN

## Overview

This project implements binary sentiment classification on the IMDB movie review dataset using a simple Recurrent Neural Network (RNN) in PyTorch.

The notebook performs the complete pipeline:

```text
IMDB Reviews
     ↓
Label Encoding
     ↓
Train/Test Split
     ↓
tiktoken Tokenization
     ↓
Padding / Truncation
     ↓
PyTorch Tensors
     ↓
Embedding
     ↓
RNN
     ↓
Linear
     ↓
Sentiment Prediction
```

The project is designed for lab-speed experimentation by using a subset of **3,000 IMDB reviews**.

---

## Objectives

The notebook demonstrates:

- Loading and inspecting a text classification dataset
- Converting sentiment labels into numerical values
- Splitting data into training and testing sets
- Tokenizing text using `tiktoken`
- Using the `cl100k_base` tokenizer
- Padding and truncating text sequences
- Converting data into PyTorch tensors
- Building an RNN-based text classifier
- Training with mini-batches
- Using gradient clipping for training stability
- Evaluating the trained model
- Visualizing training loss
- Generating a confusion matrix
- Predicting sentiment for new movie reviews

---

## Dataset

The notebook expects the following file in the same directory:

```text
IMDB Dataset.csv
```

The dataset contains two relevant columns:

| Column | Description |
|---|---|
| `review` | Movie review text |
| `sentiment` | `positive` or `negative` |

For faster laboratory training, the notebook loads the first **3,000 rows**.

```python
df = pd.read_csv(
    "IMDB Dataset.csv",
    nrows=3000
)
```

---

## Label Encoding

The textual sentiment labels are converted into binary numerical labels:

```text
negative → 0
positive → 1
```

This allows the neural network to perform binary classification.

---

## Train/Test Split

The 3,000 reviews are divided into:

| Dataset | Samples |
|---|---:|
| Training | 2,400 |
| Testing | 600 |

The notebook uses:

```python
test_size=0.20
random_state=42
stratify=labels
```

The stratified split helps maintain the class distribution between the training and testing sets.

---

# Text Preprocessing

## Tokenization

The notebook uses the `tiktoken` library with:

```text
cl100k_base
```

```python
tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)
```

The tokenizer converts natural-language reviews into numerical token IDs.

The token IDs are shifted by `+1` so that token ID `0` can be reserved for padding.

---

## Padding and Truncation

All reviews are converted to a fixed sequence length:

```text
MAX_LENGTH = 200
```

Therefore:

- Reviews longer than 200 tokens are truncated.
- Reviews shorter than 200 tokens are padded with `0`.

This ensures that every input sequence has the same shape.

---

## PyTorch Tensors

The processed reviews are converted into PyTorch tensors.

Expected shapes:

```text
X_train → [2400, 200]
X_test  → [600, 200]

y_train → [2400]
y_test  → [600]
```

---

# DataLoader

Training and testing data are wrapped using PyTorch `TensorDataset` and `DataLoader`.

The batch size is:

```text
32
```

Training batches are shuffled, while testing batches are not shuffled.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
```

---

# RNN Model

## Architecture

The model follows:

```text
Token IDs
    ↓
Embedding
    ↓
RNN
    ↓
Linear
    ↓
Sentiment Score
```

### Embedding Layer

The embedding layer converts token IDs into dense vectors.

```python
nn.Embedding(
    vocab_size,
    embedding_dim,
    padding_idx=0
)
```

### RNN Layer

The project uses a standard PyTorch RNN:

```python
nn.RNN(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    batch_first=True
)
```

### Linear Layer

The final hidden state is passed to a linear layer producing one output value:

```python
nn.Linear(
    hidden_dim,
    1
)
```

The single output is used for binary sentiment classification.

---

## Model Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Hidden dimension | 128 |
| Maximum sequence length | 200 |
| Batch size | 32 |
| Epochs | 15 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Loss function | BCEWithLogitsLoss |
| Gradient clipping | 1.0 |

---

# Training

The model is trained using mini-batches for **15 epochs**.

For each batch:

```text
Input Batch
    ↓
Embedding
    ↓
RNN
    ↓
Linear
    ↓
Loss
    ↓
Backpropagation
    ↓
Gradient Clipping
    ↓
Adam Optimizer
    ↓
Updated Model
```

The loss function is:

```python
nn.BCEWithLogitsLoss()
```

The optimizer is:

```python
torch.optim.Adam(
    movie.parameters(),
    lr=0.0005
)
```

Gradient clipping is applied using:

```python
torch.nn.utils.clip_grad_norm_(
    movie.parameters(),
    max_norm=1.0
)
```

Gradient clipping helps improve training stability, particularly when working with recurrent networks.

---

# Training Loss Visualization

The notebook records the average training loss for every epoch and plots it.

The resulting graph helps visualize whether the model is learning during training.

---

# Model Evaluation

After training, the model is evaluated on the **600 unseen test reviews**.

The RNN output is converted into a probability using the sigmoid function:

```python
probability = torch.sigmoid(logits)
```

The classification rule is:

```text
Probability >= 0.5 → Positive
Probability <  0.5 → Negative
```

---

## Evaluation Metrics

The notebook calculates:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

### Accuracy

Measures the overall percentage of correctly classified reviews.

### Precision

Measures how reliable positive predictions are.

### Recall

Measures how many actual positive reviews are correctly identified.

### F1 Score

Provides a combined measure of precision and recall.

---

# Confusion Matrix

The notebook generates a confusion matrix to analyze the model's predictions.

The structure is:

```text
                    Predicted
                 Negative  Positive

Actual Negative     TN        FP
Actual Positive     FN        TP
```

The matrix is also visualized using Matplotlib.

---

# New Review Prediction

The notebook includes a prediction function:

```python
predict_sentiment(text)
```

It performs:

```text
New Review
    ↓
tiktoken
    ↓
Token IDs
    ↓
Padding / Truncation
    ↓
PyTorch Tensor
    ↓
Trained RNN
    ↓
Sigmoid Probability
    ↓
Positive / Negative
```

The notebook tests the model using example reviews such as:

```text
"This movie was absolutely amazing. I really enjoyed every moment of it."
```

and:

```text
"This movie was terrible, boring and a complete waste of time."
```

It also tests:

```text
"The acting was excellent and the story was fantastic."
```

---

# Project Structure

```text
.
├── rnn_imdb(1).ipynb
├── IMDB Dataset.csv
└── README.md
```

---

# Requirements

Install the required Python packages using:

```bash
pip install torch pandas numpy scikit-learn matplotlib tiktoken
```

The notebook can be executed using:

- Jupyter Notebook
- JupyterLab
- Google Colab
- Other compatible Jupyter environments

---

# How to Run

1. Place `IMDB Dataset.csv` in the same directory as the notebook.
2. Open `rnn_imdb(1).ipynb`.
3. Install the required packages.
4. Run the notebook cells from top to bottom.
5. Allow the model to complete all 15 training epochs.
6. Review the training-loss graph.
7. Check the evaluation metrics.
8. Examine the confusion matrix.
9. Test the RNN with the provided new review examples.

---

# Learning Outcome

After completing this notebook, you should understand how a basic RNN can be used for text classification, including:

- Text tokenization
- Fixed-length sequence preparation
- Word/token embeddings
- Recurrent sequence processing
- Binary classification
- Mini-batch training
- Gradient clipping
- Model evaluation
- Sentiment prediction

---

## Conclusion

This notebook demonstrates a complete end-to-end RNN sentiment classification workflow using the IMDB movie review dataset.

The implementation keeps the preprocessing pipeline consistent from raw text through tokenization, sequence preparation, tensor conversion, model training, evaluation, and prediction.
