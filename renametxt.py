#!/usr/bin/env python3
import sys
from pathlib import Path
import re

def sync_txt_years(pdf_folder: Path, txt_folder: Path):
    # Map: old_stem -> new_stem_with_year
    mapping = {}
    for pdf in pdf_folder.glob("*.pdf"):
        m = re.search(r"(.*)_(20\d{2})$", pdf.stem)
        if m:
            old_stem = m.group(1)
            year = m.group(2)
            mapping[old_stem] = f"{old_stem}_{year}"

    if not mapping:
        print("No PDFs with year suffix found.")
        return

    for txt in txt_folder.glob("*.txt"):
        if txt.stem in mapping:
            new_name = mapping[txt.stem] + txt.suffix
            target = txt.with_name(new_name)
            if target.exists():
                print(f"Skipping {txt.name} → {target.name} (already exists)")
                continue
            txt.rename(target)
            print(f"Renamed {txt.name} → {target.name}")
        else:
            print(f"No matching PDF found for {txt.name}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python sync_txt_years.py /path/to/pdf_folder /path/to/txt_folder")
        sys.exit(1)

    pdf_folder = Path(sys.argv[1])
    txt_folder = Path(sys.argv[2])

    if not pdf_folder.is_dir() or not txt_folder.is_dir():
        print("Both arguments must be directories.")
        sys.exit(1)

    sync_txt_years(pdf_folder, txt_folder)

if __name__ == "__main__":
    main()
