# 💬 Logistic Regression and DistilBERT for Classification of Social Media Data

**Author:** Faizan Waheed  
**Email:** fawaheed@iu.edu  
**Institution:** Indiana University Bloomington  
**Date:** February 2025  

---

## 📘 Overview

This project explores **toxic comment classification** across social media platforms using two distinct machine learning approaches:

1. **Logistic Regression** — a lightweight, interpretable baseline using TF-IDF features.  
2. **DistilBERT** — a transformer-based deep learning model fine-tuned for contextual toxicity detection.

The project evaluates and compares both models based on performance metrics such as accuracy, precision, recall, and F1-score to determine which best captures the nuances of online toxicity.

The full report is available in this repository as:  
📄 **Project Report.pdf**

---

## 🎯 Objective

The goal of this project is to automatically classify comments as **toxic** or **non-toxic** to help improve online moderation systems.  
Key objectives include:

- Comparing traditional statistical learning and transformer-based NLP methods.  
- Evaluating trade-offs in **accuracy, interpretability, and computational cost**.  
- Producing reliable predictions on unseen social media data.  

---

## 🧠 Definition of “Toxic”

> “A rude, disrespectful, or unreasonable comment that is likely to make readers want to leave a discussion.”

---

## 📊 Dataset

**Source:** Collected from multiple social media platforms (Reddit, Twitter/X, YouTube).  
**Size:** 4,000 comments (train) + unlabeled test set.  

Each comment includes metadata such as:
- Text body  
- Parent comment (if reply)  
- Article title & URL  
- Platform and unique ID  
- Composite toxicity scores (from 5 human annotators)

**Labeling Approach:**  
A **majority vote** from the 5 annotations determined the final label:
- `1` → Toxic  
- `0` → Non-toxic  

The test dataset follows the same structure but excludes toxicity labels.

---

## ⚙️ Data Processing

### 🧹 Preprocessing Steps
1. **Data Loading** — Imported and structured via `pandas`.  
2. **Label Encoding** — Converted multi-annotator results into binary toxicity labels.  
3. **Train/Validation Split** — 80/20 stratified split ensuring balanced classes.  

### 🔢 For Logistic Regression
- **Text Cleaning:** Removed punctuation, special characters, extra spaces.  
- **Vectorization:** Applied **TF-IDF** with:
  - `max_features=20000`
  - `ngram_range=(1,3)`
  - `min_df=3`, `max_df=0.9`
  - `stop_words="english"`
- **Model Training:** Logistic Regression with:
  - `solver="saga"`, `penalty="l2"`, `class_weight="balanced"`
  - Regularization (`C=0.0001`) and 1000 iterations.  

### 🔠 For DistilBERT
- **Tokenizer:** `DistilBertTokenizer` with truncation (max length = 128) and attention masks.  
- **Fine-Tuning:** `distilbert-base-uncased` model trained with:
  - Optimizer: `AdamW`  
  - Learning rate: `2e-5`  
  - Weight decay: `0.3`, dropout: `0.4`  
  - Batch size: 8, epochs: 3  
  - Early stopping with patience = 1  

---

## 🧩 Model Comparison

| Metric | Logistic Regression | DistilBERT |
|--------|--------------------|------------|
| Accuracy | 75.6% | **78.2%** |
| Precision | 52.6% | **56.3%** |
| Recall | 49.3% | **67.3%** |
| F1-Score | 50.9% | **61.3%** |

**Findings:**
- DistilBERT outperformed Logistic Regression on **all key metrics**.  
- It captured **contextual and implicit toxicity** (e.g., sarcasm, indirect insults) more effectively.  
- Logistic Regression was more interpretable but less context-aware.  
- DistilBERT provided better **recall**, ensuring fewer toxic comments went undetected.  

---

## 🧪 Insights

### 🔹 Logistic Regression
- Fast, interpretable, and resource-efficient.  
- Performs well for short, explicit toxic phrases.  
- Struggles with nuanced or context-dependent language.  

### 🔹 DistilBERT
- Leverages pretraining and contextual embeddings.  
- Handles **sarcasm, tone, and implicit toxicity**.  
- More computationally intensive but far more accurate for real-world moderation systems.  

---

## 🧾 Results Summary

**Best Model:** DistilBERT (Epoch 2)  
- Validation Loss: 0.4558  
- Accuracy: 78.2%  
- Precision: 56.3%  
- Recall: 67.3%  
- F1-Score: 61.3%  

**Final Output:**  
Predictions were saved as `distilbert_predictions.csv`, labeling each test comment as toxic (`1`) or non-toxic (`0`).

---

## 🧰 Technologies Used
- **Languages:** Python (3.10+)  
- **Libraries:** `pandas`, `scikit-learn`, `torch`, `transformers`, `numpy`, `matplotlib`  
- **Frameworks:** Hugging Face Transformers, PyTorch  
- **Preprocessing:** NLTK, TF-IDF  
- **Environment:** Jupyter Notebook / Google Colab  


