import os
import sys
import chromadb
from typing import List

from config_manager import (
    embed_client,
    EMBED_MODEL,
    UICC_DB_STORAGE,
    UICC_DIR,
)

# ChromaDB settings
STORAGE_PATH = str(UICC_DB_STORAGE)
COLLECTION_NAME = "medical_collection"

# ——— Embedding & Upsert (no chunking) ——————————————————————————————————————————
def embed_txts_to_chroma(text_dir: str = None):
    """
    Read all .txt files in text_dir, embed each file's full content as one document,
    and upsert into a ChromaDB collection.
    """
    # fallback to config or default folder
    text_dir = text_dir or str(UICC_DIR)
    if not os.path.isdir(text_dir):
        raise FileNotFoundError(f"No such directory: {text_dir}")

    files = [f for f in os.listdir(text_dir) if f.lower().endswith(".txt")]
    if not files:
        raise FileNotFoundError(f"No .txt files found in {text_dir}")

    # initialize ChromaDB
    client = chromadb.PersistentClient(path=STORAGE_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # load all documents
    doc_names: List[str] = []
    documents: List[str] = []
    for fn in files:
        path = os.path.join(text_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            documents.append(f.read())
        doc_names.append(os.path.splitext(fn)[0])

    # batch-embed all full-text docs
    resp = embed_client.embeddings.create(
        model=EMBED_MODEL,
        input=documents,
        encoding_format="float"
    )
    embeddings = [d.embedding for d in resp.data]

    # prepare ids & metadata
    ids = doc_names[:]  # each filename (minus .txt) is the ID
    metadatas = [{"doc_name": name} for name in doc_names]

    # upsert into ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Upserted {len(ids)} documents into '{COLLECTION_NAME}' from '{text_dir}'.")


if __name__ == "__main__":
    # allow passing the text directory on the command line
    dir_arg = sys.argv[1] if len(sys.argv) > 1 else None
    embed_txts_to_chroma(dir_arg)