# 📦 Classify Mahalli: Product Classification System

A robust machine learning pipeline designed to classify local products (Mahalli) using State-of-the-Art **Sentence Embeddings**. This project achieves a high level of semantic understanding by leveraging transformer-based models and the power of the **XGBoost** algorithm.

---

### 📈 Key Performance Metrics:
* **Accuracy:** 93%
* **F1-Score:** 0.92 (Balanced across classes)
* **Classifier:** XGBoost (Extreme Gradient Boosting)

---

## 🛠️ Technical Stack
* **Language:** Python
* **Embedding Engine:** `Sentence-Transformers`
* **Pre-trained Model:** `paraphrase-multilingual-MiniLM-L12-v2`
* **Machine Learning:** `XGBoost`
* **Data Handling:** `Pandas` & `NumPy`


---

## 📂 Project Architecture
The project follows a modular **Object-Oriented Programming (OOP)** structure for scalability:

* **`app/data/loader.py`**: Handles dataset ingestion and automated cleaning (NaN removal).
* **`app/embeddings/embedder.py`**: Encapsulates the Transformer model to generate 384-dimensional dense vectors.
* **`app/models/trainer.py`**: Manages **XGBoost** model training and hyperparameter settings.
* **`app/models/evaluator.py`**: Generates detailed classification reports and performance visualizations.

---

## 📊 Evaluation Results
The model was tested on a hold-out validation set of 120 samples. The results demonstrate high reliability:

| Metric | Class 0 | Class 1 | Macro Avg |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.96 | 0.89 | 0.93 |
| **Recall** | 0.88 | 0.97 | 0.93 |
| **F1-Score** | 0.92 | 0.93 | 0.92 |

---

