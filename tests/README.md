# tests/

Small checks on hand-made examples whose correct answer can be worked out by hand. `test_metrics.py`
covers the evaluation metrics in `hrs_topics/metrics.py`: same-topic detection with outliers, balanced
accuracy, ARI with outliers removed, and topic recovery by Jaccard overlap. `test_topics.py` checks
c-TF-IDF against BERTopic's formula and that a term found in every topic is ranked down.

```bash
python -m pytest tests
```

They need no data and run in about a second. Run them after changing `metrics.py` or `topics.py`.
