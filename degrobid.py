import os
from pathlib import Path
from config_manager import XML2TXT_ESMO_DIR

folder = Path(os.getenv("DEGROBID_FOLDER", XML2TXT_ESMO_DIR))  # 改成你的文件夹路径

for file in folder.glob("*_grobid.txt"):
    new_name = file.name.replace("_grobid", "")
    file.rename(file.with_name(new_name))
    print(f"Renamed: {file.name} -> {new_name}")
