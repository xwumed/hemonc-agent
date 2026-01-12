#!/usr/bin/env python3
import sys
import re
from pathlib import Path
import base64
import mimetypes
import tomllib
from openai import OpenAI
import fitz  # PyMuPDF

# ─────────── Load config ───────────
def load_config(file_path: str = "config.toml") -> dict:
    with open(file_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg["llm"]  # needs an [llm] section with api_key, api_base, timeout, model_name

cfg = load_config()
cli = OpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"], timeout=cfg["timeout"])
MODEL_NAME = cfg["model_name"]

# ─────────── Helpers ───────────
def first_page_png_data_url(pdf_path: Path, zoom: float = 3.0) -> str:
    """
    Render only the first page of the PDF to PNG and return a data URL.
    """
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(0)
    pm = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    png_bytes = pm.tobytes("png")
    doc.close()

    mime = "image/png"
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"

def ask_year_from_llm(image_data_url: str) -> str | None:
    """
    Ask the LLM to output ONLY a 4-digit year 20XX. Returns '20xx' or None.
    """
    blocks = [
        {
            "type": "text",
            "text": (
                "You are given the FIRST PAGE of a paper as an image.\n"
                "Task: Extract the publication year and reply with ONLY a 4-digit year between 2000 and 2099.\n"
                "Rules:\n"
                "- Output ONLY the year digits, with no extra text.\n"
                "- If the year is not visible or you are not sure, reply exactly: None"
            ),
        },
        {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}},
    ]
    resp = cli.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": blocks}],
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Be strict: extract 20XX
    m = re.search(r"\b(20\d{2})\b", raw)
    if m:
        return m.group(1)
    if raw.lower() == "none":
        return None
    return None

def needs_suffix(stem: str) -> bool:
    """Check if filename already ends with _20xx."""
    return re.search(r"_20\d{2}$", stem) is None

def unique_target(path: Path) -> Path:
    """Avoid collisions by appending -1, -2, ... if needed."""
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_stem(f"{path.stem}-{i}")
        if not candidate.exists():
            return candidate
        i += 1

# ─────────── Main workflow ───────────
def process_folder(folder: Path):
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {folder}")
        return

    for pdf in pdfs:
        try:
            print(f"\n=== {pdf.name} ===")
            # Skip if already has _20xx suffix
            if not needs_suffix(pdf.stem):
                print("  -> already has year suffix, skipping.")
                continue

            data_url = first_page_png_data_url(pdf)
            year = ask_year_from_llm(data_url)

            if not year:
                print("  -> year not found; skipping rename.")
                continue

            new_name = f"{pdf.stem}_{year}{pdf.suffix}"
            target = unique_target(pdf.with_name(new_name))
            pdf.rename(target)
            print(f"  -> renamed to: {target.name}")
        except Exception as e:
            print(f"  [ERROR] {pdf.name}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python rename_pdfs_by_year.py /path/to/pdf_folder")
        sys.exit(1)
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)
    process_folder(folder)

if __name__ == "__main__":
    main()
