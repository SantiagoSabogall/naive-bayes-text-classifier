import argparse
from pathlib import Path
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, X_test: pd.Series, y_test: pd.Series) -> dict:
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

def plot_confusion_matrix(y_test: pd.Series, y_pred, class_labels, output_path: Path) -> None:
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained text classification model on test data")
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test.csv"), help="Path to test csv data")
    parser.add_argument("--model-path", type=Path, default=Path("models/naive_bayes.joblib"), help="Path to the trained model")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory to save evaluation results")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = joblib.load(args.model_path)
    
    print(f"Loading test data from {args.test_data}...")
    data = pd.read_csv(args.test_data)
    X_test = data["Comment"]
    y_test = data["Topic"]

    print("Evaluating model...")
    report, predictions = evaluate_model(model, X_test, y_test)

    args.results_dir.mkdir(exist_ok=True)
    
    report_path = args.results_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Classification report saved to {report_path}")

    cm_path = args.results_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_test,
        predictions,
        class_labels=model.classes_,
        output_path=cm_path
    )
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
