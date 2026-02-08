import fitz
import re
import json
from typing import List, Dict

# =====================================================
# CONFIG
# =====================================================
PDF_PATH = r"..\dataset\Master_Circular_For_Depositories.pdf"
SECTION_MARKER = "Section 1: Beneficial Owner (BO) Accounts"

MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 2500
CHUNK_OVERLAP = 150

OUTPUT_FILE = "chunks.json"


SOURCE_META = {
    "source": "SEBI Master Circular for Depositories",
    "regulator": "SEBI",
    "circular_date": "2023-10-06",
    "domain": "Depositories / KYC",
    "document_type": "Master Circular"
}

# =====================================================
# 1. READ PDF
# =====================================================
def read_pdf(path: str):
    doc = fitz.open(path)
    pages = []
    page_map = {}
    pos = 0

    for page_num, page in enumerate(doc, 1):
        txt = page.get_text("text")
        pages.append(txt)
        page_map[pos] = page_num
        pos += len(txt) + 1

    doc.close()
    return "\n".join(pages), page_map


def get_page(position: int, page_map: Dict[int, int]) -> int:
    for p in sorted(page_map.keys(), reverse=True):
        if position >= p:
            return page_map[p]
    return 1

# =====================================================
# 2. NORMALIZE TEXT
# =====================================================
def normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# =====================================================
# 3. DETECT NUMERIC CLAUSES ONLY
# =====================================================
CLAUSE_REGEX = re.compile(
    r"(?m)^(1\.\d+(?:\.\d+)*)(?:\s*[:\-–])?\s*([^\n]{5,160})?\n"
)

def detect_clauses(text: str) -> List[Dict]:
    clauses = []
    for m in CLAUSE_REGEX.finditer(text):
        title = (m.group(2) or "").strip()
        title = re.sub(r"\d+$", "", title).strip()
        clauses.append({
            "id": m.group(1),
            "title": title,
            "start": m.start()
        })
    return clauses

# =====================================================
# 4. EXTRACT CLAUSE BLOCKS
# =====================================================
def extract_blocks(text: str, clauses: List[Dict]):
    blocks = []

    if not clauses:
        blocks.append({
            "clause_id": "1",
            "clause_title": "Section 1",
            "content": text,
            "start_position": 0
        })
        return blocks

    for i, c in enumerate(clauses):
        start = c["start"]
        end = clauses[i + 1]["start"] if i + 1 < len(clauses) else len(text)
        raw = text[start:end].strip()

        body = re.sub(r"^.*?\n", "", raw, count=1).strip()
        body = re.sub(r"\n?\d+\s+Reference:.*$", "", body, flags=re.I | re.M).strip()

        if len(body) < MIN_CHUNK_SIZE:
            continue

        if c["title"] and not body.lower().startswith(c["title"].lower()[:20]):
            body = c["title"] + "\n\n" + body

        blocks.append({
            "clause_id": c["id"],
            "clause_title": c["title"],
            "content": body,
            "start_position": start
        })

    return blocks

# =====================================================
# 5. HARD SPLIT (RAM-SAFE, TERMINATING)
# =====================================================
def hard_split(text: str):
    parts = []
    length = len(text)
    start = 0

    while start < length:
        end = min(start + MAX_CHUNK_SIZE, length)
        parts.append(text[start:end].strip())

        if end == length:
            break  # 🔒 CRITICAL: terminate at EOF

        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = end  # 🔒 forward-only guarantee

        start = next_start

    return [p for p in parts if p]

# =====================================================
# 6. SPLIT BLOCK (BULLET FIRST, HARD FALLBACK)
# =====================================================
BULLET_REGEX = re.compile(
    r"\n(?=(?:\([ivx]+\)|\([a-z]\)|[a-z]\.|[a-z]\)|\d+\.))",
    re.I
)

def split_block(block: Dict):
    text = block["content"]

    if len(text) <= MAX_CHUNK_SIZE:
        return [{
            **block,
            "chunk_index": 0,
            "total_chunks": 1
        }]

    parts = re.split(BULLET_REGEX, text)
    chunks = []
    buffer = ""

    for p in parts:
        if len(buffer) + len(p) <= MAX_CHUNK_SIZE:
            buffer += ("\n" if buffer else "") + p
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = p

    if buffer.strip():
        chunks.append(buffer.strip())

    final_chunks = []
    for c in chunks:
        if len(c) > MAX_CHUNK_SIZE:
            final_chunks.extend(hard_split(c))
        else:
            final_chunks.append(c)

    result = []
    total = len(final_chunks)

    for i, txt in enumerate(final_chunks):
        result.append({
            "clause_id": f"{block['clause_id']}-part{i+1}" if total > 1 else block["clause_id"],
            "clause_title": block["clause_title"],
            "content": txt,
            "start_position": block["start_position"],
            "chunk_index": i,
            "total_chunks": total
        })

    return result

# =====================================================
# 7. TOPIC TAGGING (LIGHT)
# =====================================================
def infer_topics(text: str):
    t = text.lower()
    out = []
    if "kyc" in t or "pan" in t or "aadhaar" in t:
        out.append("KYC")
    if "bsda" in t:
        out.append("BSDA")
    if "closure" in t:
        out.append("Account Closure")
    if "beneficial owner" in t:
        out.append("Beneficial Ownership")
    return out or ["General Compliance"]

# =====================================================
# 8. PIPELINE
# =====================================================
def process():
    raw, page_map = read_pdf(PDF_PATH)
    text = normalize(raw)

    idx = text.find(SECTION_MARKER)
    if idx != -1:
        text = text[idx:]

    clauses = detect_clauses(text)
    blocks = extract_blocks(text, clauses)

    # 🔻 FREE MEMORY EARLY
    del raw
    del text

    chunks = []

    for b in blocks:
        page = get_page(b["start_position"], page_map)
        for c in split_block(b):
            chunks.append({
                "text": c["content"],
                "metadata": {
                    **SOURCE_META,
                    "section": "1",
                    "clause_id": c["clause_id"],
                    "clause_title": c["clause_title"],
                    "page_number": page,
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                    "char_count": len(c["content"]),
                    "topics": infer_topics(c["content"])
                }
            })

    return chunks

# =====================================================
# 9. STATS
# =====================================================
def stats(chunks):
    return {
        "total_chunks": len(chunks),
        "min_size": min(c["metadata"]["char_count"] for c in chunks),
        "max_size": max(c["metadata"]["char_count"] for c in chunks),
        "avg_size": sum(c["metadata"]["char_count"] for c in chunks) / len(chunks)
    }

# =====================================================
# 10. RUN
# =====================================================
if __name__ == "__main__":
    chunks = process()
    s = stats(chunks)

    print("Chunk stats:")
    for k, v in s.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nWritten {len(chunks)} chunks to {OUTPUT_FILE}")

    print("\n--- SAMPLE ---")
    print(chunks[0]["metadata"])
    print(chunks[0]["text"][:400])
