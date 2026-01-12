from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_naive_bayes_model(max_features: int | None = 20_000,
                            ngram_range: tuple[int, int] = (1, 1),
                            alpha: float = 1.0
                            ) -> Pipeline:
    """
    Build a text classification pipeline using CountVectorizer
    and Multinomial Naive Bayes.
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    classifier = MultinomialNB(
        alpha=alpha,
        fit_prior=True
    )

    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])





                                                 