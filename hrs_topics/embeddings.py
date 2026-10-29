"""The ten document representations and how each one encodes an abstract.

Every model reads the same text: title + blank line + abstract (corpus.csv, column embedding_text).
Only the input format each model's authors prescribe is applied on top:
  - SPECTER2 and SciNCL: "title [SEP] abstract", CLS-token pooling
  - E5: "passage: " prefix for documents
  - SciBERT, BioMedBERT, BGE-M3: CLS-token pooling (BGE-M3 dense output only)
  - MPNet, MiniLM, E5, BGE-large: sentence-transformers defaults (mean pooling)
All vectors are L2-normalised. TF-IDF is a word-level sparse baseline.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ModelSpec:
    key: str
    name: str            # Hugging Face model id
    library: str         # "sentence-transformers" or "transformers"
    max_tokens: int      # longer inputs are truncated
    batch_size: int
    prefix: str = ""
    title_sep_abstract: bool = False


SPECS = {s.key: s for s in [
    ModelSpec("mpnet", "sentence-transformers/all-mpnet-base-v2", "sentence-transformers", 384, 32),
    ModelSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers", 256, 64),
    ModelSpec("e5_large", "intfloat/e5-large-v2", "sentence-transformers", 512, 16, prefix="passage: "),
    ModelSpec("bge_large", "BAAI/bge-large-en-v1.5", "sentence-transformers", 512, 16),
    ModelSpec("bge_m3", "BAAI/bge-m3", "transformers", 8192, 8),
    ModelSpec("specter2", "allenai/specter2_base", "transformers", 512, 8, title_sep_abstract=True),
    ModelSpec("scincl", "malteos/scincl", "transformers", 512, 8, title_sep_abstract=True),
    ModelSpec("scibert", "allenai/scibert_scivocab_uncased", "transformers", 512, 16),
    ModelSpec("biomedbert", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
              "transformers", 512, 16),
]}


def input_texts(spec, corpus):
    if spec.title_sep_abstract:
        return [f"{t} [SEP] {a}" for t, a in zip(corpus.title.fillna(""), corpus.abstract.fillna(""))]
    return [spec.prefix + t for t in corpus.embedding_text.astype(str)]


def encode(spec, texts, device, batch_size=None):
    """Return an (n_documents x dim) float32 matrix of L2-normalised vectors."""
    bs = batch_size or spec.batch_size
    if spec.library == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(spec.name, device=device)
        model.max_seq_length = spec.max_tokens
        return model.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=True).astype(np.float32)

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.name)
    model = AutoModel.from_pretrained(spec.name).to(device).eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=spec.max_tokens, padding=True,
                  return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=device.startswith("cuda")):
            cls = model(**enc).last_hidden_state[:, 0, :]        # CLS-token pooling
        out.append(torch.nn.functional.normalize(cls.float(), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def fit_tfidf(texts):
    """Word-level TF-IDF, L2-normalised rows (sparse CSR)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", stop_words="english",
                          ngram_range=(1, 1), min_df=2, max_df=0.95, sublinear_tf=True, norm="l2")
    return vec.fit_transform(texts)
