# Naive Bayes Text Classifier

## 📌 Overview
This project implements a text classification system based on the Naive Bayes
algorithm. The goal is to classify textual data into predefined categories using
probabilistic machine learning techniques and a clean, reproducible pipeline.

The project is designed with clarity, modularity, and scientific reproducibility
in mind.

---

## 🎯 Objective
- Build a text classifier using Naive Bayes
- Implement a complete NLP pipeline
- Evaluate model performance using standard metrics
- Provide a clean and reusable project structure

---

## 🧠 Model
The classifier is based on the Naive Bayes algorithm, which applies Bayes' theorem
under the assumption of conditional independence between features.

Depending on the feature representation, the model follows:
- Multinomial Naive Bayes (for word frequency-based features)

Feature extraction methods include:
- Bag of Words (BoW)
- TF-IDF (optional)

---

## ⚙️ Pipeline
The workflow of the project is structured as follows:

1. Text cleaning and normalization
2. Tokenization
3. Feature extraction
4. Model training
5. Model evaluation

Each stage is modularized to allow easy experimentation and extension.

---

## 📁 Project Structure
### Estructura del Proyecto

```text
├── README.md
├── requirements.txt
├── data/
│   └── processed/       # Conjuntos de datos limpios y procesados
├── notebooks/           # Análisis exploratorio y experimentos
├── report/              # Informe del proyecto (PDF / LaTeX)
├── results/             # Métricas y resultados generados
└── src/
    ├── __init__.py
    ├── preprocessing.py # Limpieza de texto y tokenización
    ├── model.py         # Implementación de Naive Bayes
    ├── train.py         # Pipeline de entrenamiento
    └── evaluate.py      # Evaluación del modelo

## 🛠️ Installation

pip install -r requirements.txt
