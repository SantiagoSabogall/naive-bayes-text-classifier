from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent

file = project_root / "data" / "processed" / "data_processed.csv"

df = pd.read_csv(file)

X = df["Comment"]
y = df["Topic"]

X_train, X_test ,y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42)

model = make_pipeline(CountVectorizer(), MultinomialNB(
                                          fit_prior = True,
                                          class_prior= [1/2, 1/3, 1/3],
                                          force_alpha=False))

model.fit(X_train, y_train)



                                                    