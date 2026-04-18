import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV

from model import build_naive_bayes_model


def train_model(data_path: Path, model_output_path: Path):
    """
    Train the Naive Bayes text classifier using GridSearchCV and save the best model.
    """
    print(f"Loading training data from {data_path}...")
    data = pd.read_csv(data_path)
    
    X_train = data["Comment"]
    y_train = data["Topic"]

    model = build_naive_bayes_model()

    param_grid = {
        'vectorizer__max_features': [10000, 20000, None],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__stop_words': [None, 'english'],
        'classifier__alpha': [0.1, 0.5, 1.0]
    }

    print("Starting Grid Search Cross-Validation...")
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_output_path)
    print(f"Model saved successfully to {model_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Naive Bayes model using TF-IDF and GridSearchCV")
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train.csv"), help="Path to train csv data")
    parser.add_argument("--model-output", type=Path, default=Path("models/naive_bayes.joblib"), help="Output path for the trained model")
    args = parser.parse_args()

    train_model(args.train_data, args.model_output)


if __name__ == "__main__":
    main()
