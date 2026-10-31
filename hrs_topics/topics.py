"""Topic words by class-based TF-IDF (c-TF-IDF), the weighting of BERTopic's ClassTfidfTransformer.

Each topic's abstracts are pooled into one class; outliers (-1) form one more class. A term's weight
in class c is
    tf(c, x) / sum_x' tf(c, x')  *  log(1 + A / f(x))
where f(x) is the term's count summed over all classes and A is the mean number of term occurrences
per class (as an integer). Terms that are common in every topic get a low weight.
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

STOPWORDS_FILE = Path(__file__).resolve().parent / "hrs_stopwords.txt"


def stop_words():
    """scikit-learn's English stop words plus HRS boilerplate (hrs, data, wave, respondent, ...)."""
    extra = {w.strip().lower() for w in STOPWORDS_FILE.read_text().splitlines()
             if w.strip() and not w.startswith("#")}
    return sorted(set(ENGLISH_STOP_WORDS) | extra)


def vectorizer():
    # unigrams and bigrams; tokens made only of digits (95, 2006, ...) are dropped
    return CountVectorizer(ngram_range=(1, 2), stop_words=stop_words(), min_df=5, max_df=0.8,
                           token_pattern=r"(?u)\b(?!\d+\b)\w\w+\b")


def ctfidf(D, labels):
    """c-TF-IDF of a document-term count matrix D. Returns (classes incl. -1, weights, idf)."""
    classes = sorted(int(t) for t in np.unique(labels))
    col = {t: i for i, t in enumerate(classes)}
    R = sp.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), [col[int(l)] for l in labels])),
                      shape=(D.shape[0], len(classes)))
    tf = (R.T @ D).toarray().astype(float)                 # term counts per class
    A = int(tf.sum(axis=1).mean())
    idf = np.log(1.0 + A / np.maximum(tf.sum(axis=0), 1e-12))
    weight = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1e-12) * idf
    return classes, weight, idf


def top_words(texts, labels, n=10):
    """Return {topic_id: [n highest-weighted terms]} for every non-outlier topic."""
    vec = vectorizer()
    D = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    classes, weight, _ = ctfidf(D, np.asarray(labels))
    return {t: [str(terms[j]) for j in np.argsort(-weight[i])[:n]]
            for i, t in enumerate(classes) if t != -1}
