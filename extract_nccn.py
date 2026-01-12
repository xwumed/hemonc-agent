import sys
import os
import tomllib
import base64
import mimetypes
from pathlib import Path
from typing import Optional, Tuple
import fitz
import pdfplumber
from openai import OpenAI
from pydantic import BaseModel, Field
from concurrent.futures import ProcessPoolExecutor, as_completed

# ────────────────────────── Config ──────────────────────────
def load_config(file_path: str = "config.toml") -> dict:
    with open(file_path, "rb") as f:
        return tomllib.load(f)

cfg = load_config()
llm_client = OpenAI(
    api_key=cfg["llm"]["api_key"],
    base_url=cfg["llm"]["api_base"],
    timeout=cfg["llm"]["timeout"],
)
LLM_MODEL = cfg["llm"]["model_name"]

OUTPUT_DIR = Path("extracted_content")
OUTPUT_DIR.mkdir(exist_ok=True)

# ────────────────────────── LLM Output Model ──────────────────────────
class PageCheck(BaseModel):
    has_table_or_flowchart: bool
    reasoning: str= Field(...,
        description="Explain the step-by-step thought process behind the provided value. Include key considerations and how they influenced the final decisions."
    )

# ────────────────────────── Utility Functions ──────────────────────────
def png_to_data_url(p: Path) -> str:
    mime, _ = mimetypes.guess_type(p.name)
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime or 'image/png'};base64,{b64}"

def render_pdf_page_to_png(pdf_path: str, page_index: int, zoom: float = 2.0) -> Optional[Path]:
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_index)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = Path(f"temp_page_{page_index+1}.png")
        pix.save(img_path)
        doc.close()
        return img_path
    except Exception as e:
        print(f"⚠️ Render error on page {page_index+1}: {e}")
        return None

def check_page_has_table_or_flowchart(image_path: Path) -> PageCheck:
    data_url = png_to_data_url(image_path)
    prompt = """You are a document layout analysis expert.
Decide if this page contains any of the following:

1. A **table** — this includes:
   - Standard bordered tables.
   - Non-standard pseudo-tables with no visible grid lines but clear columnar organization.
   - Layouts with multiple side-by-side columns containing repeated headers, lists, or structured data.
   - Bullet lists under aligned headings in separate vertical blocks should also be considered tables.

2. A **flowchart** — diagrams with arrows, boxes, nodes, decision diamonds, or step-by-step pathways, regardless of whether they are drawn as shapes or embedded images.

If any part of the page contains either type, mark it as True.

Output JSON:
- has_table_or_flowchart: true/false
- reasoning: a concise explanation describing what was detected and why it qualifies.
"""
    resp = llm_client.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": data_url}
            ]}
        ],
        response_format=PageCheck
    )
    return resp.choices[0].message.parsed  # ✅ Extract the parsed Pydantic model
 # ✅ Return the parsed PageCheck object directly


def transcribe_page(image_path: Path) -> str:
    data_url = png_to_data_url(image_path)
    blocks = [
        {
            "type": "text",
            "text": (
                "Please transcribe the table or flowchart exactly as it appears on the page into clear, readable paragraphs, and also include all other information on the page.\n"
                "*For a table or non-standard pseudo-table, list all details in bullet points, clearly indicating each header and its corresponding values, especially noting any merged cells.\n"
                "*For a flowchart, follow the diagram's reading order, including every element and arrow.\n"
                "*Preserve any superscripts and their corresponding annotations inline at their original locations.\n"
                "*Do not use phrases like 'This image depicts...' or 'The table depicts...' Present the content itself in prose.\n"
                "If the page contains neither a table nor a flowchart, simply return the text exactly as printed."
            ),
        },
        {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
    ]
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": blocks}],
    )
    return resp.choices[0].message.content.strip()

def extract_text_with_pdfplumber(pdf_path: str, page_index: int) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        text = page.extract_text() or ""
    return text.strip()

# ────────────────────────── Worker for One Page ──────────────────────────
def process_page_worker(args: Tuple[str, int, str]) -> Tuple[int, str]:
    pdf_path, page_index, pdf_stem = args
    page_num = page_index + 1
    workdir = OUTPUT_DIR / pdf_stem
    output_txt = workdir / f"{pdf_stem}_page_{page_num}.txt"

    img_path = render_pdf_page_to_png(pdf_path, page_index)
    if not img_path:
        output_txt.write_text("", encoding="utf-8")
        return page_num, "render_failed"

    check_result = check_page_has_table_or_flowchart(img_path)
    if check_result.has_table_or_flowchart:
        transcription = transcribe_page(img_path)
        output_txt.write_text(transcription, encoding="utf-8")
        img_path.unlink(missing_ok=True)
        return page_num, "llm_transcribed"
    else:
        text = extract_text_with_pdfplumber(pdf_path, page_index)
        output_txt.write_text(text, encoding="utf-8")
        img_path.unlink(missing_ok=True)
        return page_num, "plain_text"

# ────────────────────────── Main PDF Processing ──────────────────────────
def process_single_pdf(pdf_path: str, max_workers: int = 4):
    pdf_file = Path(pdf_path)
    pdf_stem = pdf_file.stem
    workdir = OUTPUT_DIR / pdf_stem
    workdir.mkdir(exist_ok=True)

    # Find Discussion starting page
    discussion_page = None
    for f in workdir.glob(f"{pdf_stem}_Discussionfrompage_*_*.txt"):
        name_parts = f.name.split("_Discussionfrompage_")
        if len(name_parts) > 1:
            try:
                discussion_page = int(name_parts[1].split("_")[0])
                break
            except ValueError:
                pass
    if discussion_page is None:
        print(f"⚠️ No Discussion start page found, processing full PDF")
        discussion_page = fitz.open(pdf_path).page_count + 1

    end_page_index = discussion_page - 2
    print(f"📄 Processing {pdf_stem} until page {end_page_index+1} …")

    tasks = [(pdf_path, i, pdf_stem) for i in range(0, end_page_index + 1)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_page_worker, t): t[1] for t in tasks}
        for future in as_completed(futures):
            page_num, status = future.result()
            print(f"✅ Page {page_num}: {status}")

# ────────────────────────── Entry ──────────────────────────
# def main():
#     if len(sys.argv) not in (2, 3):
#         print("Usage: python extract_pdf.py /path/to/pdf [max_workers]")
#         sys.exit(1)
#
#     target = Path(sys.argv[1])
#     if not target.exists():
#         print(f"❌ Path not found: {target}")
#         sys.exit(1)
#
#     max_workers = int(sys.argv[2]) if len(sys.argv) == 3 else 4
#
#     if target.is_file() and target.suffix.lower() == ".pdf":
#         process_single_pdf(str(target), max_workers=max_workers)
#     else:
#         print("❌ Please provide a single PDF file.")
def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python extract_pdf.py /path/to/pdf_or_folder [max_workers]")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"❌ Path not found: {target}")
        sys.exit(1)

    try:
        max_workers = int(str(sys.argv[2]).strip("[]")) if len(sys.argv) == 3 else 4
    except ValueError:
        max_workers = 4

    if target.is_file() and target.suffix.lower() == ".pdf":
        # Process a single PDF
        process_single_pdf(str(target), max_workers=max_workers)

    elif target.is_dir():
        # Process all PDFs in the folder
        pdf_files = sorted([f for f in target.glob("*.pdf") if f.is_file()])
        if not pdf_files:
            print("❌ No PDF files found in the folder.")
            sys.exit(1)

        for pdf_path in pdf_files:
            print(f"\n=== Processing {pdf_path.name} ===")
            process_single_pdf(str(pdf_path), max_workers=max_workers)

    else:
        print("❌ Please provide a PDF file or a folder containing PDF files.")

if __name__ == "__main__":
    main()
