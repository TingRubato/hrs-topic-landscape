# tests/

Small checks of the evaluation metrics in `hrs_topics/metrics.py`, on hand-made examples whose
correct answer can be worked out by hand: same-topic detection with outliers, balanced accuracy,
ARI with outliers removed, and topic recovery by Jaccard overlap.

```bash
python -m pytest tests
```

They need no data and run in about a second. Run them after changing anything in `metrics.py`.
