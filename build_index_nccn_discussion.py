import sys
import re
from pathlib import Path
import chromadb

from config_manager import (
    embed_client,
    EMBED_MODEL,
    NCCN_DB_STORAGE,
)

# ────────────────────────── Configuration ──────────────────────────
STORAGE_PATH = str(NCCN_DB_STORAGE)
chroma_client = chromadb.PersistentClient(path=STORAGE_PATH)
collection = chroma_client.get_or_create_collection(name="medical_collection")
OUTPUT_DIR = Path("extracted_content")

# regex to catch both patterns
PAGE_RE = re.compile(r"(?:_page_|_Discussionfrompage_)(\d+)")

def process_extracted_texts(pdf_path: str):
    pdf_file = Path(pdf_path)
    pdf_stem = pdf_file.stem
    workdir = OUTPUT_DIR / pdf_stem

    if not workdir.exists():
        print(f"❌ Directory not found: {workdir}")
        return

    print(f"\n🔑 Embedding extracted text for {pdf_stem} …")
    # grab *all* txt files
    all_text_paths = sorted(workdir.glob("*.txt"))

    for txt_path in all_text_paths:
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"⚠️ {txt_path.name}: empty, skipping")
            continue

        # derive a unique doc_id
        doc_id = f"{pdf_stem}_{txt_path.stem}"

        # extract page number if present
        m = PAGE_RE.search(txt_path.name)
        page_num = int(m.group(1)) if m else None

        # skip if already in Chroma
        existing = collection.get(ids=[doc_id])
        if existing.get("ids"):
            print(f"✅ {doc_id} exists, skipping")
            continue

        # create embedding
        try:
            resp = embed_client.embeddings.create(
                model=EMBED_MODEL,
                input=text,
                encoding_format="float"
            )
            embedding = resp.data[0].embedding
        except Exception as e:
            print(f"⚠️ embedding error for {txt_path.name}: {e}")
            continue

        # upsert into ChromaDB
        collection.upsert(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{
                "pdf_name": pdf_stem,
                "page": page_num,
                "source": str(txt_path)
            }],
        )
        print(f"  ✔ Upserted {txt_path.name} (tokens: {resp.usage.prompt_tokens})")

    print(f"✅ Done with {pdf_stem}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python embed_pdf_deepinfra.py /path/to/pdf_or_directory")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"❌ Path not found: {target}")
        sys.exit(1)

    pdf_paths = []
    if target.is_dir():
        pdf_paths = [str(p) for p in sorted(target.iterdir()) if p.suffix.lower()==".pdf"]
    elif target.suffix.lower() == ".pdf":
        pdf_paths = [str(target)]
    else:
        print("❌ Provide a PDF or directory of PDFs.")
        sys.exit(1)

    for pdf_path in pdf_paths:
        process_extracted_texts(pdf_path)

if __name__ == "__main__":
    main()
