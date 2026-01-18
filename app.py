# app.py
# Streamlit RAG workshop app (performance-focused)
# - Tab 1: choose/load text + chunking + colored chunk view (NO embeddings)
# - Tab 2: 3D embeddings (requires clicking "Apply / Compute embeddings")
# - Tab 3: query + system prompt + answer + sources (requires embeddings)
#
# Key performance choices:
# - scientific pool capped at max 5 docs
# - lower default max chunks embedded
# - embeddings computed ONLY when user clicks Apply

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import plotly.express as px
from streamlit_plotly_events import plotly_events

from sentence_transformers import SentenceTransformer

# Optional: local answer generation
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
    TRANSFORMERS_OK = True
except Exception:
    TRANSFORMERS_OK = False

# Optional: HF scientific corpus
try:
    from datasets import load_dataset  # type: ignore
    DATASETS_OK = True
except Exception:
    DATASETS_OK = False


# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="RAG Workshop App", layout="wide")
st.title("RAG Workshop App")
st.caption("Tab 1: Chunking • Tab 2: Embeddings • Tab 3: Query + Answer + Sources")


# ============================================================
# Text utilities + chunking (simple token-proxy)
# ============================================================
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_NON_PRINTING_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'(])")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _NON_PRINTING_RE.sub("", text)
    lines = [(_WS_RE.sub(" ", ln)).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NL_RE.sub("\n\n", text).strip()
    return text


def tok(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def detok(tokens: List[str]) -> str:
    if not tokens:
        return ""
    s = " ".join(tokens)
    s = re.sub(r"\s+([,.;:!?)\]])", r"\1", s)
    s = re.sub(r"([\(\[])\s+", r"\1", s)
    s = re.sub(r"\s+([’'”\"])", r"\1", s)
    s = re.sub(r"([“\"'])\s+", r"\1", s)
    return s.strip()


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


def split_headings(text: str) -> List[str]:
    """
    Better heading detection for scientific papers.
    Splits on:
      - Markdown headings (# ...)
      - Wiki headings (== ... ==)
      - ALL CAPS lines
      - Common paper headings (Abstract, Introduction, References, ...)
      - Numbered sections (1 Introduction, 2.1 Experiments, 3 Related Work, ...)
    """
    lines = text.split("\n")
    blocks: List[str] = []
    cur: List[str] = []

    common = {
        "abstract", "introduction", "background", "related work", "method", "methods",
        "approach", "model", "experiments", "results", "discussion", "conclusion",
        "conclusions", "references", "acknowledgements", "acknowledgments", "appendix"
    }

    def is_heading_line(ln: str) -> bool:
        s = ln.strip()
        if not s:
            return False

        # Markdown / Wiki
        if re.match(r"^\s{0,3}#{1,6}\s+\S", s):
            return True
        if re.match(r"^\s*==+\s*\S.*==+\s*$", s):
            return True

        # ALL CAPS heuristic
        if len(s) >= 8 and s.isupper():
            return True

        # Common headings
        if s.lower() in common:
            return True

        # Numbered sections: "1 Introduction", "2.1 Experiments", "3.2.1 Details"
        if re.match(r"^\d+(\.\d+)*\s+[\w\(\)\-,:]{3,}", s):
            return True

        return False

    def flush():
        if cur:
            blocks.append("\n".join(cur).strip())
            cur.clear()

    for ln in lines:
        if is_heading_line(ln):
            flush()
        cur.append(ln)

    flush()
    return [b for b in blocks if b.strip()]


def make_spans_fixed(tokens: List[str], chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = len(tokens)
    s = 0
    while s < n:
        e = min(s + chunk_size, n)
        spans.append((s, e))
        if e >= n:
            break
        s = max(s + 1, e - overlap)
    return spans


def pack_units_into_spans(units: List[str], chunk_size: int, overlap: int) -> Tuple[List[Tuple[int, int]], List[str]]:
    all_tokens: List[str] = []
    unit_ranges: List[Tuple[int, int]] = []

    for u in units:
        t = tok(u)
        if not t:
            continue
        s = len(all_tokens)
        all_tokens.extend(t)
        e = len(all_tokens)
        unit_ranges.append((s, e))
        all_tokens.append("\n")  # soft boundary for display

    if all_tokens and all_tokens[-1] == "\n":
        all_tokens.pop()

    spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(unit_ranges):
        s_tok = unit_ranges[i][0]
        e_tok = unit_ranges[i][1]
        j = i + 1
        while j < len(unit_ranges) and (unit_ranges[j][1] - s_tok) <= chunk_size:
            e_tok = unit_ranges[j][1]
            j += 1

        if (e_tok - s_tok) > chunk_size:
            unit_end = unit_ranges[i][1]
            k = s_tok
            while k < unit_end:
                end = min(k + chunk_size, unit_end)
                spans.append((k, end))
                if end >= unit_end:
                    break
                k = max(k + 1, end - overlap)
            i += 1
            continue

        spans.append((s_tok, e_tok))

        target = max(0, e_tok - overlap)
        new_i = i + 1
        for k in range(i + 1, len(unit_ranges)):
            if unit_ranges[k][0] >= target:
                new_i = k
                break
        i = max(new_i, i + 1)

    # normalize
    norm: List[Tuple[int, int]] = []
    last = (-1, -1)
    for s, e in spans:
        s = max(0, s)
        e = max(s, e)
        if (s, e) != last and e > s:
            norm.append((s, e))
            last = (s, e)

    return norm, all_tokens


def make_spans(text: str, method: str, chunk_size: int, overlap: int) -> Tuple[List[Tuple[int, int]], List[str]]:
    text = clean_text(text)
    tokens = tok(text)
    if not tokens:
        return [], []

    if method == "fixed_tokens":
        return make_spans_fixed(tokens, chunk_size, overlap), tokens
    if method == "paragraph_pack":
        return pack_units_into_spans(split_paragraphs(text), chunk_size, overlap)
    if method == "sentence_pack":
        return pack_units_into_spans(split_sentences(text), chunk_size, overlap)
    if method == "heading_pack":
        units = split_headings(text)
        if len(units) < 3:
            units = split_paragraphs(text)
        return pack_units_into_spans(units, chunk_size, overlap)

    return make_spans(text, "fixed_tokens", chunk_size, overlap)


@dataclass
class ChunkRecord:
    doc: str
    chunk_idx: int
    chunk_id: str
    start_token: int
    end_token: int
    n_tokens: int
    text: str


def build_chunk_records(doc_name: str, text: str, method: str, chunk_size: int, overlap: int, max_chunks: int) -> List[ChunkRecord]:
    spans, tokens = make_spans(text, method, chunk_size, overlap)
    recs: List[ChunkRecord] = []
    for i, (s, e) in enumerate(spans):
        recs.append(
            ChunkRecord(
                doc=doc_name,
                chunk_idx=i,
                chunk_id=f"{doc_name}__c{i}",
                start_token=s,
                end_token=e,
                n_tokens=max(0, e - s),
                text=detok(tokens[s:e]),
            )
        )
    # deterministic downsample (uniform) for embedding
    if max_chunks and len(recs) > max_chunks:
        idxs = np.linspace(0, len(recs) - 1, num=max_chunks, dtype=int)
        recs = [recs[i] for i in idxs]
    return recs


# ============================================================
# Chunk-colored HTML renderer (iframe-safe)
# ============================================================
CHUNK_PALETTE = ["#dbeafe", "#dcfce7", "#fef3c7", "#fae8ff", "#ffe4e6", "#e0f2fe", "#ede9fe", "#fff7ed"]


def _assign_chunk_index(spans: List[Tuple[int, int]], n_tokens: int) -> List[int]:
    a = [-1] * n_tokens
    for ci, (s, e) in enumerate(spans):
        s = max(0, s)
        e = min(n_tokens, e)
        for t in range(s, e):
            a[t] = ci
    last = 0
    for i in range(n_tokens):
        if a[i] == -1:
            a[i] = last
        else:
            last = a[i]
    return a


def render_colored_chunk_html(
    text: str,
    method: str,
    chunk_size: int,
    overlap: int,
    title: str,
    start_token: int,
    view_tokens: int,
) -> str:
    import html as _html

    text = clean_text(text)
    spans, toks_ = make_spans(text, method, chunk_size, overlap)
    if not toks_:
        return "<div style='font-family:system-ui'>No text.</div>"

    n = len(toks_)
    start_token = max(0, min(start_token, n - 1))
    end_token = min(n, start_token + max(1, view_tokens))
    assign = _assign_chunk_index(spans, n)

    style = """
    <style>
      body { margin:0; }
      .wrap{
        background:#f3f4f6; color:#111827; padding:14px; border-radius:14px;
        border:1px solid rgba(17,24,39,0.10);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        line-height:1.55;
      }
      .card{
        background:#ffffff; padding:14px 16px; border-radius:12px;
        border:1px solid rgba(17,24,39,0.08);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        overflow-wrap:anywhere;
      }
      .title{ font-weight:700; margin:0 0 6px 0; }
      .meta{ color:#6b7280; font-size:0.92em; margin:0 0 10px 0; }
      .chunk{
        color:#111827; padding:0.10em 0.18em; border-radius:0.35em;
        box-shadow: inset 0 0 0 1px rgba(17,24,39,0.14);
        margin:0 0.03em; white-space:normal;
      }
    </style>
    """

    pieces: List[str] = []
    for i in range(start_token, end_token):
        t = toks_[i]
        if t == "\n":
            pieces.append("<br/><br/>")
            continue

        ci = assign[i]
        bg = CHUNK_PALETTE[ci % len(CHUNK_PALETTE)]
        safe = _html.escape(t)

        if i > start_token and re.match(r"^[,.;:!?)\]]$", t):
            rendered = safe
        elif i > start_token and re.match(r"^[\(\[]$", t):
            rendered = " " + safe
        else:
            rendered = (" " if i > start_token else "") + safe

        pieces.append(f"<span class='chunk' title='chunk {ci}' style='background:{bg}'>{rendered}</span>")


    meta = f"Chunking: {method} • size={chunk_size} • overlap={overlap} • window [{start_token:,}..{end_token:,}) of {n:,}"
    return (
        style
        + "<div class='wrap'>"
        + f"<div class='title'>{_html.escape(title)}</div>"
        + f"<div class='meta'>{_html.escape(meta)}</div>"
        + f"<div class='card'>{''.join(pieces)}</div>"
        + "</div>"
    )


# ============================================================
# Embeddings + retrieval (cached)
# ============================================================
@st.cache_resource
def get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=True)
def embed_texts(model_name: str, texts: List[str], batch_size: int = 32) -> np.ndarray:
    emb = get_embedder(model_name)
    vecs = emb.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine == dot
        show_progress_bar=False,
    )
    return vecs


def topk_cosine(query_vec: np.ndarray, vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = vecs @ query_vec
    k = min(k, len(scores))
    top = np.argpartition(-scores, kth=k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]


def nearest_neighbors(vecs: np.ndarray, idx: int, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    v = vecs[idx]
    sims = vecs @ v
    order = np.argsort(-sims)
    order = order[order != idx]
    return order[:k], sims[order[:k]]


# ============================================================
# Optional: local answer generation (Flan-T5)
# ============================================================
@st.cache_resource
def get_flan():
    name = "google/flan-t5-base"
    tokz = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
    return name, tokz, mdl


def generate_answer_flan(system_prompt: str, question: str, sources: List[ChunkRecord], max_new_tokens: int = 220) -> str:
    _, tokz, mdl = get_flan()
    blocks = []
    for i, r in enumerate(sources, start=1):
        snippet = r.text.strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "…"
        blocks.append(f"[{i}] ({r.doc}, {r.chunk_id}) {snippet}")

    prompt = (
        f"SYSTEM:\n{system_prompt}\n\n"
        f"CONTEXT:\n" + "\n\n".join(blocks) + "\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER (cite sources like [1], [2]):\n"
    )
    inputs = tokz(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokz.decode(out[0], skip_special_tokens=True).strip()


# ============================================================
# Scientific corpus loader (cap at 5 docs)
# ============================================================
@st.cache_data(show_spinner=True)
def load_scientific_pool(
    dataset_config: str,
    n_docs: int,
    min_chars: int,
    max_chars: int,
    keyword: str,
    seed: int,
    max_scan: int,
) -> Dict[str, str]:
    if not DATASETS_OK:
        return {}

    kw = (keyword or "").strip().lower()
    ds = load_dataset(
        "scientific_papers",
        dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    try:
        ds = ds.shuffle(seed=seed, buffer_size=5000)
    except Exception:
        pass

    pool: Dict[str, str] = {}
    scanned = 0
    for ex in ds:
        scanned += 1
        article = clean_text(ex.get("article", "") or "")
        L = len(article)
        if L < min_chars or L > max_chars:
            if scanned >= max_scan:
                break
            continue
        if kw and kw not in article.lower():
            if scanned >= max_scan:
                break
            continue

        name = f"{dataset_config.upper()}_paper_{len(pool)+1:02d}"
        pool[name] = article
        if len(pool) >= n_docs or scanned >= max_scan:
            break

    return pool


# ============================================================
# Session state for "Apply / Compute embeddings"
# ============================================================
def _sig_for_apply(
    scope: str,
    docs_for_embedding: List[str],
    method: str,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
    embed_model: str,
    batch: int,
) -> Tuple:
    return (
        scope,
        tuple(sorted(docs_for_embedding)),
        method,
        chunk_size,
        overlap,
        max_chunks,
        embed_model,
        batch,
    )


# ============================================================
# Sidebar: Tab 1 controls (defaults tuned for speed)
# ============================================================
st.sidebar.header("Text & Chunking")

input_mode = st.sidebar.radio(
    "Text source",
    ["Paste text", "Upload .txt/.md", "Scientific corpus (HF)"],
    index=2,
)

DOC_POOL: Dict[str, str] = {}
single_text = ""
single_title = "Untitled"

if input_mode == "Paste text":
    single_title = "Pasted text"
    single_text = clean_text(st.sidebar.text_area("Paste text", height=140))

elif input_mode == "Upload .txt/.md":
    up = st.sidebar.file_uploader("Upload file", type=["txt", "md"])
    if up is not None:
        single_title = f"Uploaded: {up.name}"
        single_text = clean_text(up.read().decode("utf-8", errors="ignore"))
    else:
        single_title = "Uploaded text"
        single_text = ""

else:
    st.sidebar.caption("For workshops: keep pool <= 5 docs for speed.")
    if not DATASETS_OK:
        st.sidebar.error("Install scientific corpus support: pip install 'datasets==3.6.0'")
    dataset_config = st.sidebar.selectbox("Dataset", ["arxiv", "pubmed"], index=0)
    pool_keyword = st.sidebar.text_input("Keyword filter (optional)", value="retrieval")
    pool_n_docs = st.sidebar.slider("Docs in pool (max 5)", 1, 5, 5, 1)
    pool_min_chars = st.sidebar.slider("Min doc length (chars)", 6000, 60000, 12000, 1000)
    pool_max_chars = st.sidebar.slider("Max doc length (chars)", 20000, 180000, 90000, 5000)
    pool_seed = st.sidebar.number_input("Shuffle seed", 1, 10_000, 42, 1)
    pool_max_scan = st.sidebar.slider("Max scan (speed cap)", 500, 6000, 2000, 250)

    if DATASETS_OK:
        try:
            DOC_POOL = load_scientific_pool(
                dataset_config=dataset_config,
                n_docs=pool_n_docs,
                min_chars=pool_min_chars,
                max_chars=pool_max_chars,
                keyword=pool_keyword,
                seed=pool_seed,
                max_scan=pool_max_scan,
            )
        except Exception as e:
            DOC_POOL = {}
            st.sidebar.error("Failed loading scientific corpus.")
            st.sidebar.code(str(e))

    if DOC_POOL:
        selected_doc = st.sidebar.selectbox("Active doc (Tab 1 display)", list(DOC_POOL.keys()))
        single_title = selected_doc
        single_text = DOC_POOL[selected_doc]
    else:
        single_title = "Scientific corpus"
        single_text = ""

# Chunking controls (defaults tuned)
chunk_method = st.sidebar.selectbox(
    "Chunking algorithm",
    ["fixed_tokens", "paragraph_pack", "sentence_pack", "heading_pack"],
    index=1,
    help=(
        "fixed_tokens: sliding window on tokens\n"
        "paragraph_pack: pack paragraphs up to chunk_size\n"
        "sentence_pack: pack sentences up to chunk_size\n"
        "heading_pack: split on headings, then pack"
    ),
)

chunk_size = st.sidebar.slider("chunk_size (token proxy)", 100, 1000, 600, 50)
overlap = st.sidebar.slider("overlap (token proxy)", 0, 300, 80, 10)
max_chunks_embedded = st.sidebar.slider("max chunks embedded per doc", 50, 1200, 300, 50)

# Embedding controls
st.sidebar.header("Embeddings")
embed_model_name = st.sidebar.selectbox(
    "Embedding model",
    ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
    index=0,
)
embed_batch = st.sidebar.slider("Embedding batch size", 8, 128, 32, 8)

# Embedding / retrieval scope
scope = st.sidebar.radio(
    "Embedding scope",
    ["Single text (active)", "Pool (multiple docs)"],
    index=1 if DOC_POOL else 0,
)

pool_selected_docs: List[str] = []
if scope == "Pool (multiple docs)" and DOC_POOL:
    pool_selected_docs = st.sidebar.multiselect(
        "Docs used for embedding/retrieval",
        options=list(DOC_POOL.keys()),
        default=list(DOC_POOL.keys())[: min(3, len(DOC_POOL))],
    )

# Apply button
st.sidebar.header("Apply")
apply_clicked = st.sidebar.button("Apply / Compute embeddings", type="primary")

# Plot performance (Tab 2)
st.sidebar.header("Plot performance")
downsample_plot = st.sidebar.checkbox("Downsample points (Tab 2)", value=True)
plot_max_points = st.sidebar.slider("Max points in 3D plot", 200, 2500, 900, 100, disabled=not downsample_plot)


# ============================================================
# Compute records for embedding scope (used ONLY on apply)
# ============================================================
def compute_records_for_scope() -> List[ChunkRecord]:
    if scope == "Single text (active)":
        if not single_text.strip():
            return []
        return build_chunk_records(
            doc_name=single_title,
            text=single_text,
            method=chunk_method,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks=max_chunks_embedded,
        )

    # Pool
    if not DOC_POOL:
        return []
    docs = pool_selected_docs or list(DOC_POOL.keys())
    recs: List[ChunkRecord] = []
    for d in docs:
        recs.extend(
            build_chunk_records(
                doc_name=d,
                text=DOC_POOL[d],
                method=chunk_method,
                chunk_size=chunk_size,
                overlap=overlap,
                max_chunks=max_chunks_embedded,
            )
        )
    return recs


# ============================================================
# Handle Apply click (compute and store embeddings)
# ============================================================
if scope == "Single text (active)":
    docs_for_embedding = [single_title]  # just one doc
else:
    if DOC_POOL:
        docs_for_embedding = pool_selected_docs or list(DOC_POOL.keys())
    else:
        docs_for_embedding = []

current_sig = _sig_for_apply(
    scope=scope,
    docs_for_embedding=docs_for_embedding,
    method=chunk_method,
    chunk_size=chunk_size,
    overlap=overlap,
    max_chunks=max_chunks_embedded,
    embed_model=embed_model_name,
    batch=embed_batch,
)

if apply_clicked:
    records = compute_records_for_scope()
    if not records:
        st.sidebar.error("Nothing to embed. Load text or corpus first.")
    else:
        with st.spinner("Computing embeddings (Apply)…"):
            texts = [r.text for r in records]
            vecs = embed_texts(embed_model_name, texts, batch_size=embed_batch)
        st.session_state.applied_sig = current_sig
        st.session_state.applied_records = records
        st.session_state.applied_vecs = vecs
        st.sidebar.success(f"Embeddings ready: {len(records):,} chunks")


def embeddings_ready() -> bool:
    return (
        st.session_state.applied_sig == current_sig
        and st.session_state.applied_records
        and isinstance(st.session_state.applied_vecs, np.ndarray)
        and st.session_state.applied_vecs.size > 0
    )


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["1) Text + Chunking", "2) Embeddings (3D)", "3) Query + Answer + Sources"])


# ============================================================
# TAB 1: show colored chunks for ACTIVE doc (no embeddings)
# ============================================================
with tab1:
    left, right = st.columns([1, 1.7])

    with left:
        st.subheader("Active text (from selected paper)")
        if single_text.strip():
            tks = tok(single_text)
            st.write(
                f"**{single_title}**  \n"
                f"chars={len(single_text):,} • words≈{len(single_text.split()):,} • tokens≈{len(tks):,}"
            )
            st.text_area("Raw preview (first 4000 chars)", value=single_text[:4000], height=260)
        else:
            st.info("No text loaded. Use the sidebar to paste/upload/load corpus.")

    with right:
        st.subheader("Chunk-highlighted text (active doc)")
        if not single_text.strip():
            st.info("Load a text/paper first.")
        else:
            tks = tok(single_text)
            n_tokens = len(tks)

            c1, c2 = st.columns([1, 1])
            with c1:
                start_token = st.slider(
                    "Start token (display window)",
                    0,
                    max(0, n_tokens - 1),
                    0,
                    max(1, n_tokens // 200) if n_tokens else 1,
                )
            with c2:
                view_tokens = st.slider("Tokens to show", 500, 8000, 2200, 250)

            html_doc = render_colored_chunk_html(
                text=single_text,
                method=chunk_method,
                chunk_size=chunk_size,
                overlap=overlap,
                title=single_title,
                start_token=start_token,
                view_tokens=view_tokens,
            )
            components.html(html_doc, height=650, scrolling=True)

            st.caption("Tip: adjust chunk algorithm/size/overlap in the sidebar. Embeddings won’t recompute until you click Apply.")


# ============================================================
# TAB 2: 3D embeddings (requires Apply)
# ============================================================
with tab2:
    st.subheader("3D chunk embedding space (PCA → 3D)")
    st.caption("Requires Apply. Click a point to inspect the chunk + 3 nearest neighbors.")

    if not embeddings_ready():
        st.info("Embeddings are not computed for the current settings. Click **Apply / Compute embeddings** in the sidebar.")
    else:
        records: List[ChunkRecord] = st.session_state.applied_records
        vecs: np.ndarray = st.session_state.applied_vecs

        # Downsample plot points for performance (keeps mapping)
        if downsample_plot and len(records) > plot_max_points:
            plot_idxs = np.linspace(0, len(records) - 1, num=plot_max_points, dtype=int)
            rec_plot = [records[i] for i in plot_idxs]
            vec_plot = vecs[plot_idxs]
            plot_map = plot_idxs
        else:
            rec_plot = records
            vec_plot = vecs
            plot_map = np.arange(len(records))

        pca = PCA(n_components=3, random_state=42)
        pts3 = pca.fit_transform(vec_plot)

        dfp = pd.DataFrame(
            {
                "x": pts3[:, 0],
                "y": pts3[:, 1],
                "z": pts3[:, 2],
                "doc": [r.doc for r in rec_plot],
                "chunk_id": [r.chunk_id for r in rec_plot],
                "tokens": [r.n_tokens for r in rec_plot],
            }
        )

        fig = px.scatter_3d(
            dfp,
            x="x",
            y="y",
            z="z",
            color="doc",
            hover_data=["chunk_id", "tokens"],
            height=720,
            title="Chunk embeddings (PCA → 3D)",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))

        st.plotly_chart(fig, use_container_width=True)  # always show something

        clicked = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
        )


        if clicked:
            plot_point_i = int(clicked[0]["pointIndex"])
            global_i = int(plot_map[plot_point_i])

            r0 = records[global_i]
            st.markdown("### Selected chunk")
            st.write(f"**{r0.chunk_id}** • doc={r0.doc} • tokens≈{r0.n_tokens:,}")
            st.text_area("Selected chunk text", value=r0.text[:9000], height=220)

            nn_idx, nn_sim = nearest_neighbors(vecs, global_i, k=3)
            st.markdown("### 3 nearest neighbors (cosine similarity)")
            for j, (ni, sim) in enumerate(zip(nn_idx, nn_sim), start=1):
                r = records[int(ni)]
                st.write(f"**{j}. {r.chunk_id}** • doc={r.doc} • sim={float(sim):.3f} • tokens≈{r.n_tokens:,}")
                st.text_area(f"nn_{j}", value=r.text[:2200], height=130)
        else:
            st.info("Click a point to inspect.")


# ============================================================
# TAB 3: query + retrieval + answer + sources (requires Apply)
# ============================================================
with tab3:
    st.subheader("Query → Retrieval → Answer + Sources")

    default_system = (
        "You are a careful assistant. Answer the question using ONLY the provided context.\n"
        "If the context is insufficient, say you don't know. Cite sources like [1], [2]."
    )
    system_prompt = st.text_area("System prompt", value=default_system, height=110)
    question = st.text_input("Question / query", value="Explain retrieval-augmented generation (RAG) in one paragraph.")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        top_k = st.slider("Top-k chunks", 1, 10, 5, 1)
    with colB:
        ctx_chars = st.slider("Max chars per chunk (context)", 300, 2000, 900, 100)
    with colC:
        answer_mode = st.selectbox(
            "Answer mode",
            ["Retrieval only (show sources)", "Local LLM (Flan-T5)"],
            index=1 if TRANSFORMERS_OK else 0,
        )

    run_retrieval = st.button("Run retrieval", type="primary")
    gen_answer = st.button("Generate answer", disabled=(answer_mode.startswith("Retrieval") or not TRANSFORMERS_OK))

    if not embeddings_ready():
        st.info("Embeddings are not computed for the current settings. Click **Apply / Compute embeddings** in the sidebar.")
    else:
        records: List[ChunkRecord] = st.session_state.applied_records
        vecs: np.ndarray = st.session_state.applied_vecs

        if "last_retrieval" not in st.session_state:
            st.session_state.last_retrieval = None

        if run_retrieval:
            with st.spinner("Embedding query + retrieving…"):
                q_vec = embed_texts(embed_model_name, [question], batch_size=1)[0]
                idxs, scores = topk_cosine(q_vec, vecs, k=top_k)
                chosen = [records[int(i)] for i in idxs]

            rows = []
            for rank, (i, sc) in enumerate(zip(idxs, scores), start=1):
                r = records[int(i)]
                rows.append(
                    {
                        "rank": rank,
                        "score": float(sc),
                        "doc": r.doc,
                        "chunk_id": r.chunk_id,
                        "tokens": r.n_tokens,
                        "preview": (r.text[:260] + "…") if len(r.text) > 260 else r.text,
                    }
                )

            # build bounded context blocks
            ctx_blocks = []
            for i, r in enumerate(chosen, start=1):
                txt = r.text.strip()
                if len(txt) > ctx_chars:
                    txt = txt[:ctx_chars] + "…"
                ctx_blocks.append(f"[{i}] ({r.doc}, {r.chunk_id}) {txt}")

            st.session_state.last_retrieval = {
                "rows": rows,
                "chosen": chosen,
                "ctx_blocks": ctx_blocks,
            }

        if st.session_state.last_retrieval is None:
            st.info("Click **Run retrieval** to fetch top chunks.")
        else:
            rows = st.session_state.last_retrieval["rows"]
            chosen: List[ChunkRecord] = st.session_state.last_retrieval["chosen"]
            ctx_blocks: List[str] = st.session_state.last_retrieval["ctx_blocks"]

            st.markdown("### Retrieved sources")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

            st.markdown("### Context sent to the model")
            st.text_area("Context", value="\n\n".join(ctx_blocks), height=220)

            if answer_mode.startswith("Retrieval only"):
                st.markdown("### Answer")
                st.info("Retrieval-only mode: show sources above. Switch to Local LLM (Flan-T5) to generate an answer.")
            else:
                if not TRANSFORMERS_OK:
                    st.error("Transformers not installed; cannot run Local LLM mode.")
                elif gen_answer:
                    with st.spinner("Generating answer (Flan-T5)…"):
                        answer = generate_answer_flan(system_prompt, question, chosen, max_new_tokens=220)
                    st.markdown("### Answer")
                    st.write(answer)

            st.markdown("### Sources (full chunks)")
            for i, r in enumerate(chosen, start=1):
                st.markdown(f"**[{i}] {r.doc} — {r.chunk_id}**")
                st.text_area(f"src_full_{i}", value=r.text[:9000], height=170)


# ============================================================
# Installation notes
# ============================================================
with st.expander("Installation notes (only if you need them)"):
    st.markdown(
    """
    **Scientific corpus** requires `datasets==3.6.0` (datasets>=4 removed support for dataset scripts).
    **Local LLM mode** requires `transformers`.

    Example `requirements.txt`:
    ```txt
    numpy<2
    torch==2.2.2
    sentence-transformers==2.6.1
    transformers<5
    streamlit
    plotly
    streamlit-plotly-events
    scikit-learn
    pandas
    datasets==3.6.0
    huggingface_hub<0.24
    """
        )