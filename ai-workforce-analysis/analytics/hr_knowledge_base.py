"""
Phase 12: RAG-Based HR Knowledge Assistant (Optional Advanced)
------------------------------------------------------------------
Lets HR upload policy documents (PDF/TXT) and ask questions answered
directly from those documents.

This implementation uses TF-IDF retrieval (scikit-learn) rather than
a vector database, so it works fully offline with zero extra
infrastructure - a good stand-in for local dev and demos. For
production, swap `HRKnowledgeBase.retrieve()` for a call to an Amazon
Bedrock Knowledge Base (see aws_setup_guide.md Phase 12 section) -
the rest of this module (chunking, prompt building, answer generation)
stays the same either way.
"""

import os
import re
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
POLICIES_DIR = BASE_DIR / "data" / "policies"
POLICIES_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are an HR policy assistant. Answer the employee's question using ONLY "
    "the policy excerpts provided below. If the excerpts don't contain the "
    "answer, say so plainly rather than guessing. Cite which document each "
    "fact came from."
)


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    """Simple sliding-window chunker on whitespace-normalized text."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_txt(path: Path) -> str:
    return path.read_text(errors="ignore")


class HRKnowledgeBase:
    """
    Loads all documents from data/policies/, chunks them, and builds a
    TF-IDF index in memory for retrieval. Call load() once, then
    retrieve(query) as many times as needed.
    """

    def __init__(self):
        self.chunks = []       # list[str]
        self.sources = []      # list[str] - filename each chunk came from
        self.vectorizer = None
        self.matrix = None

    def load(self):
        self.chunks, self.sources = [], []
        for path in sorted(POLICIES_DIR.glob("*")):
            if path.suffix.lower() == ".pdf":
                text = _read_pdf(path)
            elif path.suffix.lower() in (".txt", ".md"):
                text = _read_txt(path)
            else:
                continue
            for chunk in _chunk_text(text):
                self.chunks.append(chunk)
                self.sources.append(path.name)

        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(self.chunks)
        return self

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list:
        """Returns [{"text": ..., "source": ...}, ...] for the top_k most relevant chunks."""
        if not self.is_ready:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": round(float(sims[i]), 3)}
            for i in top_idx if sims[i] > 0
        ]


def answer_policy_question(kb: HRKnowledgeBase, question: str) -> str:
    """Retrieve relevant chunks, then generate a grounded answer via the LLM."""
    from analytics.ai_hr_assistant import _call_llm  # reuse the same Bedrock/Anthropic dual-mode backend

    hits = kb.retrieve(question)
    if not hits:
        return "No relevant policy documents found. Upload PDFs/TXT files to data/policies/ first."

    excerpts = "\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits)
    prompt = f"POLICY EXCERPTS:\n{excerpts}\n\nEMPLOYEE QUESTION: {question}"
    from analytics import ai_hr_assistant as aha

    old_system = aha.SYSTEM_PROMPT
    aha.SYSTEM_PROMPT = SYSTEM_PROMPT
    try:
        return aha._call_llm(prompt)
    finally:
        aha.SYSTEM_PROMPT = old_system


if __name__ == "__main__":
    kb = HRKnowledgeBase().load()
    print(f"Loaded {len(kb.chunks)} chunks from {len(set(kb.sources))} document(s) in data/policies/")
    if kb.is_ready:
        hits = kb.retrieve("What benefits are available for employees?")
        for h in hits:
            print(f"  [{h['score']}] {h['source']}: {h['text'][:80]}...")
    else:
        print("No policy documents found - add PDFs/TXT to data/policies/ to test retrieval.")
