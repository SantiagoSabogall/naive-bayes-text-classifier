from sklearn.model_selection import train_test_split
import pandas as pd 
from pathlib import Path
from model import build_naive_bayes_model


def train_test_split_data(data: pd.DataFrame,
                          test_size: float = 0.2,
                          random_state: int = 42
                          ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split features and labels into train and test sets.
    """
    X = data["Comment"]
    y = data["Topic"]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "data_processed.csv"

    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = train_test_split_data(df)

    clf = build_naive_bayes_model()
    clf.fit(X_train, y_train)