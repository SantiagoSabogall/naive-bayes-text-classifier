from pathlib import Path
import pandas as pd
import joblib
import json

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def evaluate_model(
    model,
    X_test: pd.Series,
    y_test: pd.Series
) -> dict:
    """
    Evaluate a trained model and return classification metrics.
    """
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    return report, predictions


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred,
    class_labels,
    output_path: Path
) -> None:
    """
    Plot and save the confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )

    disp.plot(cmap=plt.cm.Blues)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "models" / "naive_bayes.joblib"
    data_path = project_root / "data" / "processed" / "data_processed.csv"
    results_path = project_root / "results"

    model = joblib.load(model_path)
    data = pd.read_csv(data_path)


    X = data["Comment"]
    y = data["Topic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    report, predictions = evaluate_model(model, X_test, y_test)

    results_path.mkdir(exist_ok=True)

    with open(results_path / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    plot_confusion_matrix(
        y_test,
        predictions,
        class_labels=model.classes_,
        output_path=results_path / "confusion_matrix.png"
    )
