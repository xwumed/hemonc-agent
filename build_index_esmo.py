#!/usr/bin/env python3
from pathlib import Path
import chromadb
import tiktoken  # pip install tiktoken

from config_manager import (
    embed_client,
    EMBED_MODEL,
    ESMO_DB_STORAGE,
    XML2TXT_ESMO_DIR,
    FIGURE_TXT_DIR,
)

# ChromaDB setup
STORAGE_PATH = str(ESMO_DB_STORAGE)
chroma_client = chromadb.PersistentClient(path=STORAGE_PATH)
collection = chroma_client.get_or_create_collection(name="medical_collection")

# ─────────── Tokenizer & Chunking ───────────
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text_by_tokens(text: str, chunk_size: int = 1024, overlap: int = 100) -> list[str]:
    """
    Split text into chunks of up to chunk_size tokens, with overlap tokens.
    """
    if not text:
        return []
    ids = tokenizer.encode(text)
    n = len(ids)
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, n, step):
        end = min(start + chunk_size, n)
        chunks.append(tokenizer.decode(ids[start:end]))
        if end == n:
            break
    return chunks

# ─────────── Main ───────────
def main():
    # 已有的纯文本与图文转录目录
    txt_dir = Path(XML2TXT_ESMO_DIR)
    fig_dir = Path(FIGURE_TXT_DIR)

    if not txt_dir.exists():
        print(f"[ERR] Text folder not found: {txt_dir}")
        return
    if not fig_dir.exists():
        print(f"[WARN] Figure folder not found: {fig_dir} (将仅索引正文)")

    txt_files = sorted(p for p in txt_dir.glob("*.txt") if p.is_file())
    if not txt_files:
        print(f"[WARN] No .txt files found in {txt_dir}")
        return

    docs, metas, ids = [], [], []

    for txt_path in txt_files:
        stem = txt_path.stem                 # e.g. "paper_name"
        main_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()

        # 组合 figure 文本：默认找 {stem}_figures.txt
        fig_text = ""
        fig_path = fig_dir / f"{stem}_figures.txt"
        if fig_path.exists():
            fig_text = fig_path.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            # 可选：也尝试同名 .txt（不带 _figures 后缀）
            alt_fig = fig_dir / f"{stem}.txt"
            if alt_fig.exists():
                fig_text = alt_fig.read_text(encoding="utf-8", errors="ignore").strip()

        combined = main_text
        if fig_text:
            combined = main_text + "\n\n" + fig_text

        if not combined.strip():
            print(f"[skip] Empty combined text: {txt_path.name}")
            continue

        chunks = chunk_text_by_tokens(combined, chunk_size=1024, overlap=100)
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            docs.append(chunk)
            metas.append({"pdf_name": stem, "chunk_index": idx, "has_fig": bool(fig_text)})
            ids.append(f"{stem}_chunk_{idx}")

    if not docs:
        print("[WARN] No valid chunks produced.")
        return

    # ─────────── Embed & Persist ───────────
    print(f"\n🔑 Embedding {len(docs)} chunks...")
    embeddings = []
    for i, doc in enumerate(docs, 1):
        resp = embed_client.embeddings.create(
            model=EMBED_MODEL,
            input=[doc],
            encoding_format="float"
        )
        embeddings.append(resp.data[0].embedding)
        if i % 200 == 0:
            print(f"  ...embedded {i}/{len(docs)}")

    collection.upsert(
        documents=docs,
        embeddings=embeddings,
        metadatas=metas,
        ids=ids
    )
    print(f"✅ Indexed {len(docs)} chunks into Chroma at '{STORAGE_PATH}'")

if __name__ == "__main__":
    main()
