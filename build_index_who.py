#!/usr/bin/env python3
from pathlib import Path
import chromadb
import tiktoken

from config_manager import (
    embed_client,
    EMBED_MODEL,
    WHO_DB_STORAGE,
    WHO_DIR,
)

# ChromaDB setup
STORAGE_PATH = str(WHO_DB_STORAGE)
chroma_client = chromadb.PersistentClient(path=STORAGE_PATH)
collection = chroma_client.get_or_create_collection(name="medical_collection")

# ─────────── Helpers ───────────

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text_by_tokens(text: str,
                         chunk_size: int = 1024,
                         overlap: int = 100) -> list[str]:
    """
    Split text into chunks of up to chunk_size tokens, with overlap tokens.
    """
    if not text:
        return []

    # Encode full text to token IDs
    token_ids = tokenizer.encode(text)
    total_tokens = len(token_ids)

    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        # slice out the token span, then decode back to string
        chunk_tokens = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += chunk_size - overlap

    return chunks

# ─────────── Main ───────────

def main():
    txt_dir = Path(WHO_DIR)

    docs = []
    metas = []
    ids = []

    for txt_path in sorted(txt_dir.glob("*.txt")):
        stem = txt_path.stem

        # Read text file
        main_text = txt_path.read_text(encoding="utf-8")

        # token-based chunking
        chunks = chunk_text_by_tokens(main_text,
                                      chunk_size=1024,
                                      overlap=100)
        for idx, chunk in enumerate(chunks):
            docs.append(chunk)
            metas.append({"txt_name": stem, "chunk_index": idx})
            ids.append(f"{stem}_chunk_{idx}")

    if not docs:
        print("No documents found—check your folders.")
        return

    # ─────────── Embed & Persist ───────────
    print(f"\n🔑 Embedding {len(docs)} chunks...")
    embeddings = []
    for doc in docs:
        if not doc.strip():
            embeddings.append(None)
            continue
        resp = embed_client.embeddings.create(
            model=EMBED_MODEL,
            input=[doc],
            encoding_format="float"
        )
        embeddings.append(resp.data[0].embedding)

    # filter out empties
    valid_idx = [i for i, emb in enumerate(embeddings) if emb is not None]
    v_docs = [docs[i] for i in valid_idx]
    v_embs = [embeddings[i] for i in valid_idx]
    v_meta = [metas[i] for i in valid_idx]
    v_ids = [ids[i] for i in valid_idx]

    if v_docs:
        collection.upsert(
            documents=v_docs,
            embeddings=v_embs,
            metadatas=v_meta,
            ids=v_ids
        )
        print(f"✅ Indexed {len(v_docs)} chunks into Chroma at '{STORAGE_PATH}'")
    else:
        print("No valid chunks to index.")

if __name__ == "__main__":
    main()