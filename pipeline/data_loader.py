"""
data_loader.py — Unified JSONL corpus loader for Brazilian fact-checking articles.

Loads articles from 6 normalized JSONL files into a unified list of dicts.
Ported from mussi_v3 notebook (cells 6-10).
"""

import json
import os
import unicodedata
from pathlib import Path


# ============================================================
# Label sets for verdict strategies
# ============================================================
FALSE_LABELS = {
    "falso", "fake", "enganoso", "distorcido", "golpe", "manipulado",
    "boato", "nao e verdade", "impreciso", "exagerado", "insustentavel",
    "sem evidencia", "sem contexto", "descontextualizado", "alterado",
    "nao ha evidencias", "nao e bem assim", "falso/enganoso"
}

TRUE_LABELS = {
    "verdadeiro", "fato", "verdade", "correto", "real",
    "comprovado", "confirmado", "ainda e verdade"
}


# ============================================================
# Dataset registry
# ============================================================
DATASETS = {
    "g1":       {"subdir": "g1",         "file": "g1_cleaned.jsonl"},
    "lupa":     {"subdir": "lupa",       "file": "lupa_cleaned.jsonl"},
    "aosfatos": {"subdir": "aosfatos",   "file": "aosfatos_cleaned.jsonl"},
    "estadao":  {"subdir": "estadao",    "file": "estadao_cleaned.jsonl"},
    "boatos":   {"subdir": "boatos_org", "file": "boatos_2020_2025_cleaned.jsonl"},
    "confere":  {"subdir": "confere",    "file": "confere_cleaned.jsonl"},
}


def _strip_accents(text: str) -> str:
    """Remove accents from text for label normalization."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_label(label: str) -> str:
    """Lowercase, strip accents and whitespace from a classification label."""
    if not label:
        return ""
    return _strip_accents(label.lower().strip())


def full_text(article: dict) -> str:
    """Build searchable full text from an article dict."""
    parts = [
        article.get("titulo", ""),
        article.get("subtitulo", ""),
        article.get("texto", ""),
    ]
    return " ".join(p for p in parts if p)


def load_corpus(data_dir: str) -> list:
    """
    Load all JSONL datasets from data_dir into a unified corpus.

    Args:
        data_dir: Path to the directory containing source subdirectories
                  (e.g. /mnt/C-SSD/desinformacao/coleta_datasets/data/raw)

    Returns:
        List of article dicts with unified schema:
        {url, titulo, subtitulo, texto, classificacao, source,
         data_publicacao, tags, full_text}
    """
    corpus = []
    data_path = Path(data_dir)

    for source_name, info in DATASETS.items():
        filepath = data_path / info["subdir"] / info["file"]

        if not filepath.exists():
            print(f"[WARN] Dataset not found: {filepath}")
            continue

        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                article = {
                    "url": raw.get("url", ""),
                    "titulo": raw.get("titulo", ""),
                    "subtitulo": raw.get("subtitulo", ""),
                    "texto": raw.get("texto", ""),
                    "classificacao": _normalize_label(raw.get("classificacao", "")),
                    "source": source_name,
                    "data_publicacao": raw.get("data_publicacao", ""),
                    "tags": raw.get("tags", []) or [],
                }
                # Pre-compute full_text for retrieval
                article["full_text"] = full_text(article)
                corpus.append(article)
                count += 1

        print(f"[OK] {source_name}: {count} articles loaded from {info['file']}")

    print(f"\n[TOTAL] {len(corpus)} articles loaded across {len(DATASETS)} sources")
    return corpus


if __name__ == "__main__":
    # Quick test
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/C-SSD/desinformacao/coleta_datasets/data/raw"
    corpus = load_corpus(data_dir)
    print(f"\nSample article:")
    if corpus:
        sample = corpus[0]
        print(f"  Source: {sample['source']}")
        print(f"  Title: {sample['titulo'][:80]}...")
        print(f"  Label: {sample['classificacao']}")
        print(f"  Text length: {len(sample['texto'])} chars")
