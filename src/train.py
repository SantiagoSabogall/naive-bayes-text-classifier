from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from model import build_naive_bayes_model


def split_features_labels(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split the dataset into train and test sets.
    """
    X = data["Comment"]
    y = data["Topic"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


def train_model(
    data_path: Path,
    model_output_path: Path
):
    """
    Train the Naive Bayes text classifier and save the trained model.
    """
    data = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = split_features_labels(data)

    model = build_naive_bayes_model()
    model.fit(X_train, y_train)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    return model, X_test, y_test


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "processed" / "data_processed.csv"
    model_path = project_root / "models" / "naive_bayes.joblib"

    train_model(data_path, model_path)
