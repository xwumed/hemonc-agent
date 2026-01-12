#!/usr/bin/env python3
from pathlib import Path
import chromadb
import tiktoken  # pip install tiktoken

from config_manager import (
    embed_client,
    EMBED_MODEL,
    HEMA_DB_STORAGE,
    XML2TXT_HEMAGUIDE_DIR,
)

# ChromaDB setup
STORAGE_PATH = str(HEMA_DB_STORAGE)
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
    token_ids = tokenizer.encode(text)
    total_tokens = len(token_ids)
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += step
    return chunks

# ─────────── Main ───────────
def main():
    # 直接用你已有的 txt 输出目录
    txt_dir = Path(XML2TXT_HEMAGUIDE_DIR)
    if not txt_dir.exists():
        print(f"[ERR] Text folder not found: {txt_dir}")
        return

    txt_files = sorted(p for p in txt_dir.glob("*.txt") if p.is_file())
    if not txt_files:
        print(f"[WARN] No .txt files found in {txt_dir}")
        return

    docs, metas, ids = [], [], []

    for txt_path in txt_files:
        stem = txt_path.stem  # 原文件名去掉扩展名
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            print(f"[skip] Empty text: {txt_path.name}")
            continue

        chunks = chunk_text_by_tokens(text, chunk_size=1024, overlap=100)
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            docs.append(chunk)
            metas.append({"pdf_name": stem, "chunk_index": idx})
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
