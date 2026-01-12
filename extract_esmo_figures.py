#!/usr/bin/env python3
import sys
import os
import base64
import mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz               # pip install pymupdf
import cv2                # pip install opencv-python
import numpy as np        # pip install numpy
from openai import OpenAI # pip install openai
import tomllib

# ─────────── Load config ───────────
def load_config(file_path: str = "config.toml") -> dict:
    with open(file_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg["llm"]  # your TOML needs an [llm] section

cfg = load_config()
cli = OpenAI(
    api_key  = cfg["api_key"],
    base_url = cfg["api_base"],
    timeout  = cfg["timeout"],
)
MODEL_NAME = cfg["model_name"]


def png_to_data_url(p: Path) -> str:
    """Reads a PNG and returns a data URL."""
    mime, _ = mimetypes.guess_type(p.name)
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime or 'image/png'};base64,{b64}"


def page_has_figure(pdf_path: str, page_index: int, draw_threshold: int = 200) -> bool:
    """
    Return True if the page has embedded images or significant vector drawings.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    if page.get_images(full=True):
        doc.close()
        return True
    ops = sum(len(d["items"]) for d in page.get_drawings())
    doc.close()
    return ops > draw_threshold


def render_pdf_page_to_numpy(pdf_path: str, page_index: int, zoom: float = 3.0) -> np.ndarray:
    """Render a PDF page to a BGR NumPy array."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = pix.samples
    img = np.frombuffer(arr, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    doc.close()
    return img


def transcribe_flowchart(image_path: Path) -> str:
    """
    Mimics your flowchart script: sends a blocks‐style prompt with text + image_url to Llama-4.
    """
    data_url = png_to_data_url(image_path)
    blocks = [
        {
            "type": "text",
            "text": (
                "Provide a detailed, objective transcription of the figure on this page.\n"
                "1) Legend & captions: Transcribe all legend/caption text verbatim at the top.\n"
                "2) Text in diagram: Transcribe every label inside shapes, arrows/connectors, and callouts. Keep original punctuation and casing.\n"
                "3) Superscripts & footnotes: Capture every superscript marker exactly as it appears in the figure  — preserve its exact character(s), formatting, and order, without alteration or invention.Place the marker in the same position relative to the word or phrase where it appears in the figure. The corresponding footnote text is under the figure, keep all of them exactly as they are.\n"
                "4) Structure: If the workflow is simple/moderate, present it as step-by-step bullet points in reading order (top→bottom, left→right). If it is complex or highly branched, use detailed paragraphs that describe the flow and branches clearly. Do NOT omit any element such as node, arrow, condition, or label.\n"
                "5) Output order:\n"
                "   - Legend\n"
                "   - Main Content (Bullets or Paragraphs, as appropriate)\n"
                "   - Footnotes (marker → note text)\n"
                "If the page contains no workflow figure, reply 'None.'"
            ),
        },
        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
    ]
    resp = cli.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": blocks}],
    )
    return resp.choices[0].message.content.strip()


def process_pdf(pdf_path: Path, output_folder: Path):
    stem = pdf_path.stem
    out_dir = output_folder / f"{stem}_figures"
    out_dir.mkdir(exist_ok=True)

    # count pages
    doc = fitz.open(str(pdf_path))
    total = doc.page_count
    doc.close()

    # find pages with figures, ignoring page one
    fig_pages = [i + 1 for i in range(total) if page_has_figure(str(pdf_path), i) and i != 0]
    print(f"Pages with figures in {stem}.pdf: {fig_pages}")

    if not fig_pages:
        out_txt = output_folder / f"{stem}_figures.txt"
        with out_txt.open("w", encoding="utf-8") as f:
            f.write(f"Figure transcriptions for {stem}.pdf\n\n")
        print(f"Done {stem}.pdf → {out_txt}")
        return

    # 1) Render & save all target pages first
    png_paths = {}
    for pg in fig_pages:
        idx = pg - 1
        print(f"→ Rendering page {pg}…")
        img = render_pdf_page_to_numpy(str(pdf_path), idx)
        png = out_dir / f"{stem}_page_{pg}.png"
        cv2.imwrite(str(png), img)
        png_paths[pg] = png

    # 2) Parallel LLM calls
    max_workers = int(os.getenv("LLM_MAX_WORKERS", "50"))  # tune via env if needed
    print(f"→ Transcribing in parallel with {max_workers} workers…")

    results = {}  # pg -> description
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_pg = {
            ex.submit(transcribe_flowchart, png_paths[pg]): pg
            for pg in fig_pages
        }
        for fut in as_completed(future_to_pg):
            pg = future_to_pg[fut]
            try:
                desc = fut.result()
            except Exception as e:
                desc = f"[ERROR] page {pg}: {e}"
            results[pg] = f"--- Page {pg} ---\n{desc}\n"

    # 3) Assemble in original page order
    descriptions = [results[pg] for pg in fig_pages]

    # 4) Write one .txt per PDF
    out_txt = output_folder / f"{stem}_figures.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"Figure transcriptions for {stem}.pdf\n\n")
        f.write("\n".join(descriptions))

    # 5) Cleanup PNGs
    for pg in fig_pages:
        try:
            png_paths[pg].unlink()
        except Exception:
            pass

    print(f"Done {stem}.pdf → {out_txt}")



def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_fig_describe.py /path/to/pdf_folder /path/to/output_folder")
        sys.exit(1)
    folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)
    if not output_folder.is_dir():
        print(f"Error: {output_folder} is not a directory")
        sys.exit(1)

    for pdf in sorted(folder.glob("*.pdf")):
        print(f"\n=== {pdf.name} ===")
        process_pdf(pdf, output_folder)


if __name__ == "__main__":
    main()