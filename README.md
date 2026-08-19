# Roman Urdu Language Classification

This project classifies individual words from Roman Urdu, English, and mixed language text.

## Project Objective

The main objective of this project is to identify whether a word belongs to:

- URD - Roman Urdu
- ENG - English
- MIX - Mixed/Other

## Dataset

The dataset contains 200 sentences and 1404 word records.

Dataset columns:

- Sentence_ID
- Sentence
- Word
- Label

## Machine Learning Approach

The project uses:

1. Train/Test Split
2. TF-IDF Character N-Gram Features
3. Logistic Regression
4. Model Evaluation
5. Word-Level Prediction

## Model Performance

The trained model achieved:

**Accuracy: 97.86%**

### Classification Performance

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| ENG | 0.97 | 0.94 | 0.95 |
| MIX | 0.75 | 0.86 | 0.80 |
| URD | 1.00 | 1.00 | 1.00 |

## Project Files

- `create_dataset.py` - Dataset preparation
- `dataset.csv` - Dataset
- `train_model.py` - Model training
- `predict.py` - User-friendly prediction
- `evaluate_model.py` - Model evaluation
- `model.pkl` - Trained model
- `vectorizer.pkl` - TF-IDF vectorizer
- `confusion_matrix.png` - Confusion matrix

## How to Run

### 1. Install required packages

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Run prediction

```bash
python predict.py
```

### 4. Evaluate the model

```bash
python evaluate_model.py
```

## Example

### Input

```text
please laptop open karo
```

### Output

```text
please          -> ENG
laptop          -> ENG
open            -> ENG
karo            -> URD
```

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Matplotlib
- Joblib

