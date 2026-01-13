from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.pipeline import make_pipeline
from model import  build_naive_bayes_model
from train import train_test_split_data
from pathlib import Path
import matplotlib.pyplot as plt



def evaluate_model(model:Pipeline,
                    X_test: pd.Series,
                    y_test: pd.Series) -> np.ndarray:
    
    """
    Evaluate a trained model and print classification metrics.
    Returns predictions for further analysis.
    """
    
    prediction = model.predict(X_test)
    print(classification_report(y_test, prediction))
    return prediction
    


def plot_confusion_matrix(y_test: pd.Series,                
                           y_pred: np.ndarray,
                           class_labels: list[str]) -> None:
    """
    Plot confusion matrix using sklearn utilities.
    """
    
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "data_processed.csv"

    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = train_test_split_data(df)

    model = build_naive_bayes_model()
    model.fit(X_train, y_train)

    predictions = evaluate_model(model, X_test, y_test)

    plot_confusion_matrix(
        y_test,
        predictions,
        class_labels=model.classes_
    )
