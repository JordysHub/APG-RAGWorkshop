# rag_core.py
# Core utilities for the RAG workshop Streamlit app:
# - text cleaning
# - token proxy + detokenization
# - chunk span creation (multiple strategies)
# - HTML chunk visualization (readable light theme)

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import List, Tuple


# -----------------------------
# Text cleaning / preprocessing
# -----------------------------
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_NON_PRINTING_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def clean_text(text: str) -> str:
    """
    Lightweight cleanup:
    - normalize newlines
    - collapse excessive whitespace
    - strip non-printing control chars
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _NON_PRINTING_RE.sub("", text)
    # normalize whitespace in lines but preserve newlines
    lines = [(_WS_RE.sub(" ", ln)).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NL_RE.sub("\n\n", text).strip()
    return text


# Optional: useful if you ever go back to Gutenberg texts
_GUT_START = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE)
_GUT_END = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE)

def strip_gutenberg_boilerplate(text: str) -> str:
    if not text:
        return ""
    m1 = _GUT_START.search(text)
    m2 = _GUT_END.search(text)
    if m1 and m2 and m2.start() > m1.end():
        return text[m1.end():m2.start()].strip()
    if m1:
        return text[m1.end():].strip()
    if m2:
        return text[:m2.start()].strip()
    return text.strip()


# -----------------------------
# Token proxy (fast, local)
# -----------------------------
# This is NOT the model tokenizer. It's a workshop-friendly proxy.
# Keeps punctuation as separate "tokens" so overlaps/chunk sizes behave intuitively.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")

def detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""

    s = " ".join(tokens)

    # Fix common spacing before punctuation (simple heuristic)
    s = re.sub(r"\s+([,.;:!?)\]])", r"\1", s)
    s = re.sub(r"([\(\[])\s+", r"\1", s)

    # Handle quotes a bit better
    s = re.sub(r"\s+([’'”\"])", r"\1", s)
    s = re.sub(r"([“\"'])\s+", r"\1", s)

    # Restore paragraph breaks for readability if tokenization removed them
    # (We don't preserve \n tokens in this proxy; cleaning keeps \n in original)
    return s.strip()


# -----------------------------
# Chunking strategies
# -----------------------------
def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(])")

def _split_sentences(text: str) -> List[str]:
    # crude but good enough for workshop; avoids heavy NLP deps
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return parts

def _split_headings(text: str) -> List[str]:
    """
    Split into "sections" using heading-like separators.
    Supports:
      - Markdown headings: #, ##, ###
      - Wiki/LaTeX-ish: == Heading ==
      - All-caps lines as headings (heuristic)
    """
    lines = text.split("\n")
    blocks: List[str] = []
    cur: List[str] = []

    def flush():
        if cur:
            blocks.append("\n".join(cur).strip())
            cur.clear()

    for ln in lines:
        is_md = bool(re.match(r"^\s{0,3}#{1,6}\s+\S", ln))
        is_wiki = bool(re.match(r"^\s*==+\s*\S.*==+\s*$", ln))
        is_caps = (len(ln.strip()) >= 8 and ln.strip().isupper())

        if is_md or is_wiki or is_caps:
            flush()
            cur.append(ln)
        else:
            cur.append(ln)

    flush()
    return [b for b in blocks if b.strip()]


def _pack_units_into_spans(units: List[str], chunk_size: int, overlap: int) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Pack text units (paragraphs/sentences/sections) into token spans.
    Returns (spans, all_tokens) where spans are over all_tokens indices.
    """
    all_tokens: List[str] = []
    unit_token_ranges: List[Tuple[int, int]] = []

    for u in units:
        toks = tokenize(u)
        if not toks:
            continue
        s = len(all_tokens)
        all_tokens.extend(toks)
        e = len(all_tokens)
        unit_token_ranges.append((s, e))

        # Add a soft boundary token to avoid merging punctuation weirdly
        # (visual only; it behaves like a token in chunk sizing)
        all_tokens.append("\n")
        unit_token_ranges[-1] = (s, e)  # keep original end without boundary

    # remove trailing boundary token if present
    if all_tokens and all_tokens[-1] == "\n":
        all_tokens.pop()

    spans: List[Tuple[int, int]] = []
    i = 0
    # We'll pack by units while respecting max chunk_size in tokens.
    while i < len(unit_token_ranges):
        s_tok = unit_token_ranges[i][0]
        e_tok = unit_token_ranges[i][1]
        j = i + 1
        while j < len(unit_token_ranges) and (unit_token_ranges[j][1] - s_tok) <= chunk_size:
            e_tok = unit_token_ranges[j][1]
            j += 1

        # If a single unit is longer than chunk_size, force cut (fallback to fixed)
        if (e_tok - s_tok) > chunk_size:
            # fixed sliding window inside that long unit
            k = s_tok
            while k < unit_token_ranges[i][1]:
                end = min(k + chunk_size, unit_token_ranges[i][1])
                spans.append((k, end))
                if end >= unit_token_ranges[i][1]:
                    break
                k = max(k + 1, end - overlap)
            i += 1
            continue

        spans.append((s_tok, e_tok))

        # Move to next start with overlap in token space:
        # Find the earliest unit whose start >= (e_tok - overlap)
        target = max(0, e_tok - overlap)
        # advance i until unit start >= target, but keep at least forward progress
        new_i = i + 1
        for k in range(i + 1, len(unit_token_ranges)):
            if unit_token_ranges[k][0] >= target:
                new_i = k
                break
        i = max(new_i, i + 1)

    # normalize spans (dedupe / monotonic)
    norm: List[Tuple[int, int]] = []
    last = (-1, -1)
    for s, e in spans:
        s = max(0, s)
        e = max(s, e)
        if (s, e) != last and e > s:
            norm.append((s, e))
            last = (s, e)

    return norm, all_tokens


def make_spans(
    text: str,
    method: str,
    chunk_size: int,
    overlap: int,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Returns:
      spans: list of (start_token_idx, end_token_idx) in a token-proxy space
      tokens: the token-proxy list for the full document
    """
    text = clean_text(text)
    tokens = tokenize(text)

    if not tokens:
        return [], []

    method = (method or "fixed_tokens").strip()

    if method == "fixed_tokens":
        spans: List[Tuple[int, int]] = []
        s = 0
        n = len(tokens)
        while s < n:
            e = min(s + chunk_size, n)
            spans.append((s, e))
            if e >= n:
                break
            # stride with overlap
            s = max(s + 1, e - overlap)
        return spans, tokens

    if method == "paragraph_pack":
        units = _split_paragraphs(text)
        return _pack_units_into_spans(units, chunk_size=chunk_size, overlap=overlap)

    if method == "sentence_pack":
        units = _split_sentences(text)
        return _pack_units_into_spans(units, chunk_size=chunk_size, overlap=overlap)

    if method == "heading_pack":
        units = _split_headings(text)
        # If heading split yields too few units, fall back to paragraph packing
        if len(units) < 3:
            units = _split_paragraphs(text)
        return _pack_units_into_spans(units, chunk_size=chunk_size, overlap=overlap)

    # default fallback
    return make_spans(text, method="fixed_tokens", chunk_size=chunk_size, overlap=overlap)


# -----------------------------
# HTML chunk visualization
# -----------------------------
PALETTE = [
    "#dbeafe",  # blue
    "#dcfce7",  # green
    "#fef3c7",  # amber
    "#fae8ff",  # purple
    "#ffe4e6",  # pink
    "#e0f2fe",  # cyan
    "#ede9fe",  # lavender
    "#fff7ed",  # warm
]

STYLE = """
<style>
  .rag-wrap{
    background:#f3f4f6;
    color:#111827;
    padding:16px;
    border-radius:14px;
    border:1px solid rgba(17,24,39,0.10);
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    line-height:1.55;
  }
  .rag-card{
    background:#ffffff;
    padding:16px 18px;
    border-radius:12px;
    border:1px solid rgba(17,24,39,0.08);
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    overflow-wrap:anywhere;
  }
  .rag-title{
    font-weight:700;
    margin:0 0 8px 0;
    color:#111827;
  }
  .rag-meta{
    color:#6b7280;
    font-size:0.92em;
    margin:0 0 12px 0;
  }
  .chunk{
    color:#111827;
    padding:0.10em 0.18em;
    border-radius:0.35em;
    box-shadow: inset 0 0 0 1px rgba(17,24,39,0.14);
    margin:0 0.03em;
    white-space:normal;
  }
  .chunk:hover{
    box-shadow: inset 0 0 0 2px rgba(17,24,39,0.28);
  }
  .token{
    white-space:pre-wrap;
  }
</style>
"""


def _token_chunk_assignment(spans: List[Tuple[int, int]], n_tokens: int) -> List[int]:
    """
    Assign each token index to a chunk index for display.
    With overlap, later chunks overwrite earlier assignments to create a smooth progression.
    """
    assign = [-1] * n_tokens
    for ci, (s, e) in enumerate(spans):
        s = max(0, s)
        e = min(n_tokens, e)
        for t in range(s, e):
            assign[t] = ci
    # fill gaps (shouldn't happen except if spans malformed)
    last = 0
    for i in range(n_tokens):
        if assign[i] == -1:
            assign[i] = last
        else:
            last = assign[i]
    return assign


def render_colored_chunks_html(
    text: str,
    method: str,
    chunk_size: int,
    overlap: int,
    start_token: int = 0,
    view_tokens: int = 2500,
    title: str = "Document",
) -> str:
    """
    Render a readable chunk visualization:
    - light gray canvas
    - white paper card
    - black text
    - pastel chunk highlights
    """
    text = clean_text(text)
    spans, toks = make_spans(text, method=method, chunk_size=chunk_size, overlap=overlap)
    if not toks:
        return STYLE + '<div class="rag-wrap"><div class="rag-card">No text loaded.</div></div>'

    n = len(toks)
    start_token = max(0, min(start_token, n - 1))
    end_token = min(n, start_token + max(1, view_tokens))

    assign = _token_chunk_assignment(spans, n)

    # Build HTML for the window
    pieces: List[str] = []
    for i in range(start_token, end_token):
        tok = toks[i]

        # Preserve paragraph breaks for display
        if tok == "\n":
            pieces.append("<br/><br/>")
            continue

        ci = assign[i]
        bg = PALETTE[ci % len(PALETTE)]

        # Escape token safely
        safe = html.escape(tok)

        # Add a leading space before most tokens (readability),
        # but not before punctuation.
        if i > start_token and re.match(r"^[,.;:!?)\]]$", tok):
            # punctuation: no leading space
            rendered = safe
        elif i > start_token and re.match(r"^[\(\[]$", tok):
            # opening bracket: add a space before it
            rendered = " " + safe
        else:
            # default: space + token
            rendered = (" " if i > start_token else "") + safe

        pieces.append(f'<span class="chunk" style="background:{bg}">{rendered}</span>')

    meta = (
        f"Chunking: {html.escape(method)} • size={chunk_size} • overlap={overlap} • "
        f"window tokens [{start_token:,}..{end_token:,}) of {n:,}"
    )

    out = (
        STYLE
        + '<div class="rag-wrap">'
        + f'<div class="rag-title">{html.escape(title)}</div>'
        + f'<div class="rag-meta">{meta}</div>'
        + f'<div class="rag-card">{"".join(pieces)}</div>'
        + "</div>"
    )
    return out
