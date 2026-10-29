"""Step 1: repair abstracts damaged during scraping.

Some scraped abstracts lost symbols ("( = 0.12)", "P 0.05", "95 CI: 1.2 2.3"), carry copyright
lines, have section headers glued to the text ("MethodsWe used"), or contain mojibake ("â€™").
Two passes, each logged per record:

  1. PubMed: every abstract matching an audit rule is replaced by its PubMed abstract
     (looked up by PubMed id, or by DOI when the id is missing).
  2. Other sources and local fixes: abstracts that still have deleted symbols are fetched from
     Europe PMC, OpenAlex, Crossref and Semantic Scholar (by DOI, then by title) and accepted only
     if the title matches (similarity >= 0.90), the text is the same abstract (>= 60% word
     overlap, >= 60% of the old length) and the new text is undamaged. Then every abstract gets
     deterministic fixes: copyright tail removed, glued headers split, mojibake decoded,
     Cyrillic "х" between digits -> "×", "?=?" -> " = ".

Input : data/raw/articles.csv (never modified)
Output: data/raw/articles_repaired.csv, data/raw/repair_log.csv

Usage: python scripts/01_repair_abstracts.py [--no-fetch]   (--no-fetch: local fixes only)
"""
import argparse
import html
import re
import sys
import time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrs_topics import config  # noqa: E402

UA = {"User-Agent": "HRS-topic-modeling/1.0 (research corpus repair)"}

# ---------------------------------------------------------------- pass 1: audit rules + PubMed
AUDIT_RULES = {
    "paren_eq": r"\(\s*=\s*-?\d",                       # symbol stripped before '='
    "paren_letter_eq": r"\(\s*=\s*[A-Za-z]",
    "comma_eq": r",\s*=\s",                              # '%, CI' -> ', ='
    "neg_bare": r"=\s*-\s*\d+\s*,\s*=",
    "bibliographic_tail": r"PUBLICATION ABSTRACT|©|Published online ahead of print",
    "mojibake": r"[Ââ€™â€œâ€“â€”]",
    "glued_header": r"(Objectives|Background|Methods|Results|Conclusion|Introduction|Interventions|"
                    r"Measurements|Findings)(This|We|I|The|A|In|Using|Data|N|\d)",
    "cyrillic_x": r"[хХ]",
    "mage": r"Mage\s*=",
    "qmark_eq": r"\?\s*=\s*\?",
}


def audit_flags(text):
    out = [k for k, rx in AUDIT_RULES.items() if re.search(rx, text)]
    s = text.strip()
    if len(s) > 300 and s[:120] in s[130:]:
        out.append("dup_abstract")
    return out


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pubmed_abstract(pmid):
    r = requests.get(f"{EUTILS}/efetch.fcgi", params={"db": "pubmed", "id": pmid, "rettype": "xml"},
                     headers=UA, timeout=30)
    if r.status_code != 200:
        return None
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return None
    parts = []
    for at in root.iter("AbstractText"):           # structured abstracts keep their labels
        text = "".join(at.itertext()).strip()
        if text:
            parts.append(f"{at.attrib['Label']}: {text}" if at.attrib.get("Label") else text)
    return " ".join(parts).strip() or None


def pmid_from_doi(doi):
    r = requests.get(f"{EUTILS}/esearch.fcgi", params={"db": "pubmed", "term": f"{doi}[doi]", "retmode": "json"},
                     headers=UA, timeout=30)
    try:
        ids = r.json()["esearchresult"]["idlist"]
    except Exception:
        return None
    return ids[0] if ids else None


def pubmed_pass(df, log):
    for i, row in df.iterrows():
        old = row["abstract"] if isinstance(row["abstract"], str) else ""
        if not old or not audit_flags(old):
            continue
        pmid, doi = str(row.get("pubmed_id") or "").strip(), str(row.get("doi") or "").strip()
        new = pubmed_abstract(pmid) if pmid not in ("", "nan") else None
        if not new and doi not in ("", "nan"):
            pmid = pmid_from_doi(doi)
            new = pubmed_abstract(pmid) if pmid else None
        time.sleep(0.34)                             # NCBI limit: 3 requests/s without a key
        if new:
            df.at[i, "abstract"] = new
        log.append({"article_id": row["article_id"], "step": "pubmed",
                    "status": "replaced" if new else "not found", "source": "pubmed" if new else ""})


# ---------------------------------------------------------------- pass 2: other sources + local fixes
HEADERS = (r"Objectives?|Background|Methods?|Results?|Conclusions?|Introduction|Design|"
           r"Setting|Participants|Measurements|Findings|Discussion|Purpose|Aims?")
DETECT = {k: re.compile(v) for k, v in {
    "copyright_tail": r"(?:©|\(c\)\s*\d{4}|Copyright\s*(?:©|\d{4})|PUBLICATION ABSTRACT|Published online ahead of print)",
    "glued_header": r"(?<=[a-z0-9.)%])(?:" + HEADERS + r")(?=[A-Z0-9])",
    "mojibake": r"â€|Ã[\u0080-¿]|Â[\s °±·²³µ¼½¾]|�",
    "cyrillic_x": r"(?<=\d)\s?[хХ]\s?(?=\d)",
    # a deleted symbol cannot be restored locally; the original text is needed
    "deleted_symbol": r"\(\s*=\s*[-\d]|,\s*=\s|\?\s*=\s*\?|\b[pP]\s+0?\.\d|\(\s+\d+\s*(?:years|y)\b|"
                      r"CI[:,]?\s*-?\d+\.\d+\s+-?\d+\.\d+\)|\b95\s+CI\b|\(\s*\)",
}.items()}

# a run of characters that is UTF-8 text wrongly decoded as cp1252
_MOJI_RUN = re.compile(r"[Â-ô][\u0080-¿ŒœŠšŸŽžƒˆ˜–—‘-„†-•…‰‹›€™]+")


def damage(text):
    return [k for k, rx in DETECT.items() if rx.search(text or "")]


def demojibake(s):
    def fix(m):
        try:
            return m.group().encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group()
    s = _MOJI_RUN.sub(fix, s)
    s = re.sub("Â(?=[\\s ])", "", s)      # 'Â' before a space is never real text
    return s.replace(" ", " ")


def local_fix(s):
    s = demojibake(s)
    m = DETECT["copyright_tail"].search(s)
    if m and len(s) - m.start() <= 300:             # cut only a closing tail, not a mid-text mention
        s = s[:m.start()].rstrip(" .;,-—–") + "."
    s = re.sub(r"(?<=[a-z0-9.)%])(" + HEADERS + r")(?=[A-Z0-9])", r" \1: ", s)
    s = DETECT["cyrillic_x"].sub("×", s)
    s = re.sub(r"\s?\?\s*=\s*\?\s?", " = ", s)
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def _get(url, **params):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params or None, headers=UA, timeout=20)
            if r.status_code == 429:
                time.sleep(2 + attempt * 3)
                continue
            return r if r.ok else None
        except requests.RequestException:
            time.sleep(1)
    return None


def _strip_tags(s):
    s = re.sub(r"<jats:title>.*?</jats:title>", " ", s or "", flags=re.S)
    # remove known markup only: a generic '<[^>]+>' would also eat statistics such as 'p<0.001 ... >'
    s = re.sub(r"</?(?:jats:[\w-]+|i|b|em|strong|sup|sub|p|br|span|sc|italic|bold|h\d)(?:\s[^<>]*)?/?>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _openalex_text(work):
    inv = work.get("abstract_inverted_index") or {}
    return " ".join(w for _, w in sorted((p, w) for w, ps in inv.items() for p in ps))


def europepmc(query):
    r = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
             query=query, format="json", resultType="core", pageSize=1)
    hits = r.json().get("resultList", {}).get("result", []) if r else []
    return (hits[0].get("title", ""), _strip_tags(hits[0].get("abstractText", ""))) if hits else None


def openalex_doi(doi):
    r = _get(f"https://api.openalex.org/works/doi:{doi}", select="title,abstract_inverted_index")
    return (r.json().get("title") or "", _openalex_text(r.json())) if r else None


def openalex_title(title):
    q = re.sub(r"[,:|]", " ", title or "").strip()
    r = _get("https://api.openalex.org/works", filter=f"title.search:{q}",
             select="title,abstract_inverted_index", **{"per-page": 1})
    res = r.json().get("results", []) if r else []
    return (res[0].get("title") or "", _openalex_text(res[0])) if res else None


def crossref(doi):
    r = _get(f"https://api.crossref.org/works/{doi}")
    m = r.json().get("message", {}) if r else None
    return ((m.get("title") or [""])[0], _strip_tags(m.get("abstract", ""))) if m else None


def semantic_scholar(doi):
    time.sleep(1.1)                                  # unauthenticated rate limit
    r = _get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}", fields="title,abstract")
    return (r.json().get("title") or "", r.json().get("abstract") or "") if r else None


BY_DOI = [("europepmc", lambda d: europepmc(f'DOI:"{d}"')), ("openalex", openalex_doi),
          ("crossref", crossref), ("semanticscholar", semantic_scholar)]
BY_TITLE = [("europepmc_title", lambda t: europepmc('TITLE:"%s"' % re.sub(r'["\[\]():]', " ", t or "").strip())),
            ("openalex_title", openalex_title)]


def acceptable(title, old, cand_title, cand):
    """Same paper, same abstract, and not itself damaged."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", (s or "").lower())  # noqa: E731
    words = lambda s: set(re.findall(r"[a-z]{3,}", (s or "").lower()))  # noqa: E731
    if not cand or len(cand) < 0.6 * len(old):
        return False
    if SequenceMatcher(None, norm(title), norm(cand_title)).ratio() < 0.90:
        return False
    if len(words(old) & words(cand)) / max(1, len(words(old))) < 0.60:
        return False
    return not DETECT["deleted_symbol"].search(local_fix(cand))


def second_pass(df, log, fetch):
    for i, row in df.iterrows():
        old = row["abstract"] if isinstance(row["abstract"], str) else ""
        if not old:
            continue
        new, source = old, ""
        if fetch and "deleted_symbol" in damage(old):
            doi = str(row.get("doi") or "").strip()
            tries = ([(n, f, doi) for n, f in BY_DOI] if doi not in ("", "nan") else []) + \
                    [(n, f, row["title"]) for n, f in BY_TITLE]
            for name, fn, key in tries:
                got = fn(key)
                if got and acceptable(row["title"], old, *got):
                    new, source = got[1], name
                    break
        new = local_fix(new)
        if new != old or damage(new):
            df.at[i, "abstract"] = new
            log.append({"article_id": row["article_id"], "step": "sources+local",
                        "status": "changed" if new != old else "still damaged", "source": source,
                        "damage_left": ";".join(damage(new))})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-fetch", action="store_true", help="skip all web requests; local fixes only")
    a = ap.parse_args()
    df = pd.read_csv(config.RAW_ARTICLES)
    log = []
    if not a.no_fetch:
        pubmed_pass(df, log)
    second_pass(df, log, fetch=not a.no_fetch)
    df.to_csv(config.REPAIRED_ARTICLES, index=False)
    pd.DataFrame(log).to_csv(config.REPAIR_LOG, index=False)
    changed = (df.abstract.fillna("") != pd.read_csv(config.RAW_ARTICLES).abstract.fillna("")).sum()
    print(f"{changed} abstracts changed; log: {config.REPAIR_LOG}")


if __name__ == "__main__":
    main()
