# Consumer-complaint text classification

Classifies consumer complaint narratives into one of five financial product
categories with a bag-of-words representation and a small Keras neural network.

## Data

`complaints_processed.csv` (a processed consumer-complaints CSV
with columns `product` and `narrative`) is not included in the repository;
place it next to the notebook. After dropping 10 rows without text, the notebook uses
162,411 complaints:

| Product | Complaints |
|---|---:|
| credit_reporting | 91,172 |
| debt_collection | 23,148 |
| mortgages_and_loans | 18,990 |
| credit_card | 15,566 |
| retail_banking | 13,535 |

## Method

- First 80% of rows for training (129,928), last 20% for testing (32,483). The split is
  sequential, not shuffled or stratified.
- Keras `Tokenizer` limited to the 1,000 most frequent words; each text becomes a 1,000-dim
  binary bag-of-words vector.
- Model: Dense(512, ReLU) → Dropout(0.5) → Dense(5, softmax); Adam, categorical cross-entropy,
  batch size 32, 10% of the training data held out for validation.

## Result (as recorded in the notebook)

| Split | Accuracy |
|---|---:|
| Validation | 0.844 |
| Test | **0.849** (loss 0.410) |

Note: `model.fit` is called without `epochs`, so Keras trained for its default single
epoch, even though the notebook defines `epochs=5`. The classes are imbalanced, with credit
reporting at 56% of the data, so accuracy should be read together with per-class results
(the notebook does not report a confusion matrix yet).

## Run

```bash
pip install tensorflow pandas scikit-learn matplotlib
jupyter notebook "Text classification.ipynb"
```

Recorded with TensorFlow 2.11.
