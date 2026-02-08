# =====================================================
# SEBI RAG INTELLIGENCE — FIXED & GROUNDED VERSION
# =====================================================

import os
import logging
import time
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from endee import Endee

# =====================================================
# ENV + LOGGING
# =====================================================

load_dotenv()

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="SEBI RAG Intelligence",
    page_icon="SEBI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
  .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
  [data-testid="stSidebar"] { border-right: 1px solid rgba(49, 51, 63, 0.15); }
  .endee-header {
    padding: 1.25rem 1.25rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(30, 58, 95, 0.95), rgba(49, 130, 206, 0.95));
    color: white;
    margin-bottom: 1.25rem;
  }
  .endee-header h1 { margin: 0; font-size: 1.9rem; line-height: 1.2; }
  .endee-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
  .panel {
    padding: 1rem 1rem;
    border-radius: 14px;
    border: 1px solid rgba(49, 51, 63, 0.15);
    background: rgba(255, 255, 255, 0.6);
    margin-bottom: 1rem;
  }
  .muted { color: rgba(49, 51, 63, 0.7); font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# SIDEBAR CONTROLS
# =====================================================

with st.sidebar:
    st.markdown("## Control Panel")

    with st.expander("LLM Settings", expanded=True):
        GEMINI_MODEL = st.text_input(
            "Model",
            value="models/gemma-3-27b-it",
        )
        c1, c2 = st.columns(2)
        with c1:
            TEMPERATURE = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        with c2:
            TOP_P = st.slider("Top-P", 0.0, 1.0, 0.95, 0.05)
        MAX_OUTPUT_TOKENS = st.number_input(
            "Max Output Tokens",
            256,
            8192,
            1024,
            step=128,
        )

    with st.expander("Retrieval Settings", expanded=True):
        TOP_K = st.slider("Top-K Chunks", 1, 20, 5, 1)
        SIM_THRESHOLD = st.slider(
            "Similarity Threshold",
            0.0,
            1.0,
            0.0,
            0.05,
        )
        MAX_CHARS = st.slider(
            "Max Chars per Chunk",
            500,
            5000,
            1500,
            100,
        )

    with st.expander("Vector DB", expanded=False):
        INDEX_NAME = st.text_input("Index Name", "regulatory_docs")
        ENDEE_API_KEY = os.getenv("ENDEE_API_KEY")

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_embedder():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

embedder = load_embedder()

# =====================================================
# LOAD ENDEE INDEX
# =====================================================

@st.cache_resource
def load_index():
    client = Endee(ENDEE_API_KEY)  #
    return client.get_index(INDEX_NAME)

index = load_index()

# =====================================================
# LOAD GEMMA
# =====================================================

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemma = genai.GenerativeModel(model_name=GEMINI_MODEL)

# =====================================================
# HELPERS
# =====================================================

def trim(text: str, max_chars: int) -> str:
    return text[:max_chars]

# =====================================================
# RETRIEVAL
# =====================================================

@st.cache_data(show_spinner=False)
def retrieve(query: str, top_k: int, threshold: float, max_chars: int):
    q_vec = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    hits = index.query(
        vector=q_vec,
        top_k=top_k
    )

    cited = []
    context_blocks = []

    for i, hit in enumerate(hits, 1):
        score = hit["similarity"]
        if score < threshold:
            continue

        text = trim(hit["meta"].get("text", ""), max_chars)

        cited.append({
            "idx": i,
            "id": hit["id"],
            "score": score,
            "clause": hit["meta"].get("clause_id"),
            "text": text
        })

        context_blocks.append(
            f"[{i}] (Score: {score:.4f})\n{text}"
        )

    context = (
        "\n\n".join(context_blocks)
        if context_blocks else
        "NO RELEVANT CONTEXT FOUND."
    )

    return context, cited, hits

# =====================================================
# GEMMA STREAMING (SAFE)
# =====================================================

def stream_llm(prompt: str):
    response = gemma.generate_content(
        prompt,
        stream=True,
        generation_config={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_output_tokens": MAX_OUTPUT_TOKENS
        }
    )

    for chunk in response:
        try:
            text = chunk.text
        except ValueError:
            continue

        if text:
            yield text

# =====================================================
# UI
# =====================================================

st.markdown(
    """
<div class="endee-header">
  <h1>SEBI Regulatory Intelligence</h1>
  <p>Ask compliance questions and get grounded answers with citations from retrieved chunks.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class=\"panel\">", unsafe_allow_html=True)
left, right = st.columns([4, 1])
with left:
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Ask a SEBI / KYC / Compliance question:",
            height=120,
            placeholder="e.g., What are the customer due diligence requirements for mutual funds?",
            label_visibility="visible",
        )
        run = st.form_submit_button("Analyze")
with right:
    st.markdown("<div class=\"muted\">Tips</div>", unsafe_allow_html=True)
    st.caption("- Be specific")
    st.caption("- Mention product / entity")
    st.caption("- Ask one question at a time")
st.markdown("</div>", unsafe_allow_html=True)

if run and question.strip():

    tab_ans, tab_src, tab_dbg = st.tabs(
        ["Answer", "Sources", "Debug"]
    )

    post_answer_stats = st.empty()

    t0 = time.time()
    with st.spinner("Retrieving regulatory context..."):
        context, cited_chunks, raw_hits = retrieve(
            question,
            TOP_K,
            SIM_THRESHOLD,
            MAX_CHARS,
        )
    dt = time.time() - t0

    # ================= FIXED PROMPT =================

    prompt = f"""
You are a regulatory compliance assistant.

STRICT RULES:
- Use ONLY the provided context.
- Do NOT use outside knowledge.
- Every factual statement MUST have a citation like [1], [2].
- If the provided context is NOT relevant to the question, say:
  "The provided documents do not specify this."
- If the context is relevant but incomplete, answer conservatively
  using only what is stated and clearly mention what is not specified.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    # ================= ANSWER =================

    with tab_ans:
        st.markdown("### AI Analysis")
        full = ""
        placeholder = st.empty()

        with st.status("Generating answer...", expanded=False) as gen_status:
            for tok in stream_llm(prompt):
                full += tok
                placeholder.markdown(full + "▌")

            gen_status.update(state="complete", label="Answer generated")

        placeholder.markdown(full)

    avg_sim = (
        sum(c["score"] for c in cited_chunks) / len(cited_chunks)
        if cited_chunks
        else 0.0
    )
    with post_answer_stats.container():
        st.caption(f"Retrieved {len(cited_chunks)} chunks in {dt:.2f}s")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Chunks", len(cited_chunks))
        with m2:
            st.metric("Avg similarity", f"{avg_sim:.3f}")
        with m3:
            st.metric("Context size", f"{len(context)} chars")
        with m4:
            st.metric("Retrieval time", f"{dt:.2f}s")

    # ================= SOURCES =================

    with tab_src:
        if not cited_chunks:
            st.warning("No sources met the similarity threshold.")
        else:
            st.markdown(f"### Sources ({len(cited_chunks)})")
            for i, c in enumerate(cited_chunks, 1):
                title = f"[{c['idx']}] Score {c['score']:.4f} | Clause: {c['clause']}"
                with st.expander(title, expanded=(i == 1)):
                    meta1, meta2, meta3, meta4 = st.columns(4)
                    with meta1:
                        st.metric("Similarity", f"{c['score']:.4f}")
                    with meta2:
                        st.metric("Clause", str(c["clause"]))
                    with meta3:
                        st.metric("Chars", len(c["text"]))
                    with meta4:
                        st.metric("Chunk", f"{c['idx']}")

                    st.text_area(
                        "Chunk text",
                        value=c["text"],
                        height=220,
                        key=f"chunk_{c['idx']}",
                        label_visibility="collapsed",
                    )

                    with st.popover("Show chunk id"):
                        st.code(c["id"], language=None)

    # ================= DEBUG =================

    with tab_dbg:
        st.json({
            "query": question,
            "retrieval": {
                "top_k": TOP_K,
                "threshold": SIM_THRESHOLD
            },
            "raw_hits": raw_hits
        })
