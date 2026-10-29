"""Step 2: build the analysis corpus from the repaired articles.

Removes, in this order: exact duplicate rows, duplicate title + abstract, records without a title,
records without an abstract, and the off-domain records in config.OFF_DOMAIN_IDS.
The text the models read (embedding_text) is title + blank line + abstract, with HTML tags removed
and whitespace collapsed. No stemming or stop-word removal. The title and abstract columns keep
the repaired text as it was; they are what the pair raters read in step 5.

Input : data/raw/articles_repaired.csv
Output: data/corpus.csv (one row per document; this row order is used by every later step)

Usage: python scripts/02_build_corpus.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrs_topics import config  # noqa: E402


def clean(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", " ", s.replace("\xa0", " "))
    return re.sub(r"\s+", " ", s).strip()


def main():
    df = pd.read_csv(config.REPAIRED_ARTICLES)
    title, abstract = df.title.map(clean), df.abstract.map(clean)
    exact = df.duplicated(keep="first")
    same_text = pd.DataFrame({"t": title, "a": abstract}).duplicated(keep="first")
    drop = {"exact duplicate": exact,
            "duplicate title + abstract": same_text & ~exact,
            "no title": title.eq(""),
            "no abstract": abstract.eq(""),
            "off-domain": df.article_id.isin(config.OFF_DOMAIN_IDS)}
    keep = ~pd.concat(drop, axis=1).any(axis=1)
    for reason, mask in drop.items():
        print(f"{reason:28s} {int(mask.sum())}")

    corpus = pd.DataFrame({
        "document_id": df.article_id.astype("int64"),
        "title": df.title,
        "abstract": df.abstract,
        "embedding_text": title + "\n\n" + abstract,
        "year": pd.to_numeric(df.year, errors="coerce").astype("Int64"),
        "doi": df.doi,
        "journal": df.journal,
    })[keep]
    assert corpus.document_id.is_unique
    corpus.to_csv(config.CORPUS, index=False)
    print(f"{len(corpus)} documents -> {config.CORPUS}")


if __name__ == "__main__":
    main()
