"""
rag_retrieval.py — BM25 + Semantic retrieval index for fact-checking articles.

Provides a RAGIndex class that builds and queries BM25 and/or semantic indices
over the article corpus. Supports 3 retrieval methods: bm25, semantic, hybrid.

Always builds BOTH indices at startup so users can switch at runtime.
Uses chunk_pool encoding for semantic (splits articles into 500-char chunks).
Supports embedding caching for instant restarts.
"""

import os
import re
import unicodedata
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


# ============================================================
# Tokenizer (simple, no SpaCy/NLTK dependency)
# ============================================================
def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(text: str) -> List[str]:
    """
    Simple Portuguese tokenizer: lowercase, strip accents,
    split on non-alphanumeric, filter short tokens.
    """
    text = _strip_accents(text.lower())
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2]


# ============================================================
# RAGIndex
# ============================================================
class RAGIndex:
    """
    Manages BM25 and semantic search indices over a fact-checking corpus.

    Always builds both indices at startup so the retrieval method can be
    switched per-query via the `method` parameter in retrieve().

    Semantic encoding uses chunk_pool strategy: articles are split into
    500-char chunks, embedded individually. At retrieval, max-sim across
    chunks determines per-article score.

    Usage:
        corpus = load_corpus(data_dir)
        index = RAGIndex(corpus)
        results = index.retrieve("vacina causa autismo", k=5, method="hybrid")
    """

    CHUNK_SIZE = 500  # Characters per chunk

    def __init__(self, corpus: List[Dict]):
        """
        Build the retrieval index (always builds both BM25 + semantic).

        Args:
            corpus: List of article dicts from data_loader.load_corpus()
        """
        self.corpus = corpus
        self.bm25_index = None
        self.tokenized_corpus = None
        self.semantic_model = None
        self.article_embeddings = None
        self.article_chunk_ranges = None  # (start, end) per article for chunk_pool

        self._build_bm25_index()
        self._build_semantic_index()

    def _build_bm25_index(self):
        """Build BM25 index from article full_text."""
        from rank_bm25 import BM25Okapi

        print("[RAG] Tokenizing corpus for BM25...")
        self.tokenized_corpus = [
            tokenize(article["full_text"]) for article in self.corpus
        ]

        print("[RAG] Building BM25 index...")
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        avg_tokens = np.mean([len(t) for t in self.tokenized_corpus])
        print(f"[RAG] BM25 index built: {len(self.corpus)} docs, avg {avg_tokens:.0f} tokens/doc")

    def _build_semantic_index(self):
        """
        Load sentence-transformer and encode all articles using chunk_pool.

        Chunk_pool strategy:
          1. Split each article's full_text into 500-char chunks
          2. Embed each chunk separately
          3. At retrieval: compute cosine sim for all chunks, take max per article

        Supports caching: saves embeddings + chunk_ranges to .npy files.
        """
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("SEMANTIC_MODEL", "nomic-ai/nomic-embed-text-v1.5")
        cache_dir = os.getenv("EMBEDDINGS_CACHE_DIR", "./data/embeddings_cache")
        encoding_strategy = os.getenv("ENCODING_STRATEGY", "chunk_pool")

        # Resolve cache paths
        cache_path = Path(cache_dir)
        safe_model_name = model_name.replace("/", "_")
        embeddings_file = cache_path / f"embeddings_{safe_model_name}_{len(self.corpus)}_{encoding_strategy}.npy"
        chunk_ranges_file = cache_path / f"chunk_ranges_{safe_model_name}_{len(self.corpus)}_{encoding_strategy}.npy"

        # Try loading from cache
        if embeddings_file.exists():
            print(f"[RAG] Loading cached embeddings from {embeddings_file}")
            self.article_embeddings = np.load(str(embeddings_file))

            if chunk_ranges_file.exists():
                self.article_chunk_ranges = np.load(str(chunk_ranges_file))
                print(f"[RAG] Loaded {self.article_embeddings.shape[0]} chunk embeddings "
                      f"({self.article_embeddings.shape[1]}d) for {len(self.article_chunk_ranges)} articles")
            else:
                self.article_chunk_ranges = None
                print(f"[RAG] Loaded {self.article_embeddings.shape[0]} embeddings "
                      f"({self.article_embeddings.shape[1]}d)")

            print(f"[RAG] Loading semantic model: {model_name}")
            self.semantic_model = SentenceTransformer(
                model_name, trust_remote_code=True
            )
            print(f"[RAG] Semantic model loaded (dim={self.semantic_model.get_sentence_embedding_dimension()}, "
                  f"max_seq={self.semantic_model.max_seq_length})")
            return

        # No cache — build from scratch
        print(f"[RAG] Loading semantic model: {model_name}")
        self.semantic_model = SentenceTransformer(
            model_name, trust_remote_code=True
        )
        dim = self.semantic_model.get_sentence_embedding_dimension()
        max_seq = self.semantic_model.max_seq_length
        print(f"[RAG] Model loaded: dim={dim}, max_seq_length={max_seq}")

        if encoding_strategy == "chunk_pool":
            # Chunk & Pool: split articles into 500-char chunks
            all_chunks = []
            chunk_ranges = []  # (start_idx, end_idx) per article

            for article in self.corpus:
                text = article["full_text"]
                chunks = [text[i:i + self.CHUNK_SIZE]
                          for i in range(0, max(1, len(text)), self.CHUNK_SIZE)]
                start = len(all_chunks)
                all_chunks.extend(chunks)
                chunk_ranges.append((start, len(all_chunks)))

            self.article_chunk_ranges = np.array(chunk_ranges)

            avg_chunks = len(all_chunks) / len(self.corpus)
            print(f"[RAG] Encoding {len(all_chunks)} chunks "
                  f"(avg {avg_chunks:.1f} per article, {self.CHUNK_SIZE} chars each)...")
            print(f"[RAG] This may take ~10 minutes on first run...")

            self.article_embeddings = self.semantic_model.encode(
                all_chunks,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

        elif encoding_strategy == "title_label":
            # Title + Subtitle + Label encoding
            texts = []
            for article in self.corpus:
                parts = [article["titulo"]]
                if article.get("subtitulo"):
                    parts.append(article["subtitulo"])
                if article.get("classificacao"):
                    parts.append(article["classificacao"])
                texts.append(" — ".join(parts))

            self.article_chunk_ranges = None
            print(f"[RAG] Encoding {len(texts)} title+label texts...")

            self.article_embeddings = self.semantic_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

        else:  # truncate (default fallback)
            texts = [article["full_text"][:512] for article in self.corpus]
            self.article_chunk_ranges = None
            print(f"[RAG] Encoding {len(texts)} articles (truncate to 512 chars)...")

            self.article_embeddings = self.semantic_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

        # Save to cache
        cache_path.mkdir(parents=True, exist_ok=True)
        np.save(str(embeddings_file), self.article_embeddings)
        if self.article_chunk_ranges is not None:
            np.save(str(chunk_ranges_file), self.article_chunk_ranges)
        print(f"[RAG] Embeddings cached to {cache_path}/")

        mem_mb = self.article_embeddings.nbytes / 1e6
        print(f"[RAG] Semantic index built: {self.article_embeddings.shape[0]} vectors, "
              f"{self.article_embeddings.shape[1]}d, {mem_mb:.1f} MB")

    # ============================================================
    # Retrieval methods
    # ============================================================
    def _semantic_scores(self, query: str) -> np.ndarray:
        """
        Compute per-article semantic scores.
        For chunk_pool: computes max-sim across article chunks.
        """
        query_emb = self.semantic_model.encode(
            [query], normalize_embeddings=True
        )

        if self.article_chunk_ranges is not None:
            # Chunk_pool: max-sim across chunks for each article
            all_sims = np.dot(self.article_embeddings, query_emb.T).flatten()
            scores = np.zeros(len(self.corpus))
            for i, (start, end) in enumerate(self.article_chunk_ranges):
                scores[i] = all_sims[start:end].max()
            return scores
        else:
            # Truncate or title_label: one embedding per article
            return np.dot(self.article_embeddings, query_emb.T).flatten()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        method: Optional[str] = None,
        bm25_weight: float = 0.5,
    ) -> List[Dict]:
        """
        Retrieve top-k articles for a query.

        Args:
            query: Search query string
            k: Number of results to return
            method: Retrieval method: "bm25", "semantic", or "hybrid"
                    (defaults to "bm25" if not specified)
            bm25_weight: Weight for BM25 in hybrid mode (0.0 to 1.0)

        Returns:
            List of result dicts with keys: title, link, snippet, score, source, label
        """
        method = method or "bm25"

        if method == "bm25":
            return self._retrieve_bm25(query, k)
        elif method == "semantic":
            return self._retrieve_semantic(query, k)
        elif method == "hybrid":
            return self._retrieve_hybrid(query, k, bm25_weight)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

    def _retrieve_bm25(self, query: str, k: int) -> List[Dict]:
        """Retrieve using BM25 keyword matching."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25_index.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:k]
        return [self._format_result(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def _retrieve_semantic(self, query: str, k: int) -> List[Dict]:
        """Retrieve using semantic cosine similarity (with chunk_pool support)."""
        scores = self._semantic_scores(query)

        top_indices = np.argsort(scores)[::-1][:k]
        return [self._format_result(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def _retrieve_hybrid(self, query: str, k: int, bm25_weight: float = 0.5) -> List[Dict]:
        """Retrieve using combined BM25 + semantic scores."""
        # BM25 scores
        query_tokens = tokenize(query)
        if not query_tokens:
            return self._retrieve_semantic(query, k)
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max  # Normalize to 0-1

        # Semantic scores
        semantic_scores = self._semantic_scores(query)

        # Combine
        combined = bm25_weight * bm25_norm + (1 - bm25_weight) * semantic_scores

        top_indices = np.argsort(combined)[::-1][:k]
        return [self._format_result(idx, float(combined[idx])) for idx in top_indices if combined[idx] > 0]

    def _format_result(self, idx: int, score: float) -> Dict:
        """Format an article at index `idx` into the standard result dict."""
        article = self.corpus[idx]
        return {
            "title": article["titulo"],
            "link": article["url"],
            "snippet": article["texto"][:300],
            "score": round(score, 4),
            "source": article["source"],
            "label": article["classificacao"],
        }

    def retrieve_multi_query(
        self,
        queries: List[str],
        k_per_query: int = 3,
        k_total: int = 5,
        method: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve across multiple queries, deduplicate, and return top results.

        Used for the pipeline: retrieve for the claim + each generated question.

        Args:
            queries: List of query strings (claim + generated questions)
            k_per_query: Number of results per individual query
            k_total: Total number of final deduplicated results to return
            method: Override retrieval method

        Returns:
            Deduplicated top results sorted by score
        """
        all_results = {}  # url -> result dict (keep highest score)

        for query in queries:
            results = self.retrieve(query, k=k_per_query, method=method)
            for r in results:
                url = r["link"]
                if url not in all_results or r["score"] > all_results[url]["score"]:
                    all_results[url] = r

        # Sort by score descending, take top k_total
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:k_total]


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pipeline.data_loader import load_corpus

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/C-SSD/desinformacao/coleta_datasets/data/raw"

    print("Loading corpus...")
    corpus = load_corpus(data_dir)

    print("\nBuilding full index (BM25 + Semantic)...")
    index = RAGIndex(corpus)

    test_query = "vacina causa autismo"
    for method in ("bm25", "semantic", "hybrid"):
        print(f"\nSearching: '{test_query}' [method={method}]")
        results = index.retrieve(test_query, k=3, method=method)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['source']}] {r['title'][:80]}... (score={r['score']:.4f}, label={r['label']})")
