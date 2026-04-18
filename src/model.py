from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def build_naive_bayes_model() -> Pipeline:
    """
    Build a text classification pipeline using TfidfVectorizer
    and Multinomial Naive Bayes.
    
    Parameters are left to defaults to be optimized via GridSearchCV.
    """
    vectorizer = TfidfVectorizer(
        # Default initialization; params will be tuned via GridSearch
    )

    classifier = MultinomialNB(
        fit_prior=True
    )

    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])