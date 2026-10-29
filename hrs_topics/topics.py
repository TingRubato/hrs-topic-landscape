"""Topic words by class-based TF-IDF (c-TF-IDF, the BERTopic weighting).

Each topic's abstracts are pooled into one document. A term's weight in topic t is
    tf(t, term) * log(1 + A / (1 + tf(t, term)))
where A is the mean number of term occurrences per topic. Outliers (-1) are left out.
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


def top_words(texts, labels, n=10):
    """Return {topic_id: [n highest-weighted terms]} for every non-outlier topic."""
    vec = vectorizer()
    D = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    topics = sorted(int(t) for t in np.unique(labels) if t != -1)
    col = {t: i for i, t in enumerate(topics)}
    rows = np.flatnonzero(labels != -1)
    R = sp.csr_matrix((np.ones(len(rows)), (rows, [col[int(labels[r])] for r in rows])),
                      shape=(D.shape[0], len(topics)))
    tf = (R.T @ D).toarray().astype(float)                 # term counts per topic
    A = tf.sum(axis=1).mean() + 1e-9
    weight = tf * np.log(1.0 + A / (1.0 + tf))
    return {t: [str(terms[j]) for j in np.argsort(-weight[i])[:n]] for t, i in col.items()}
