import os, re
import xml.etree.ElementTree as ET

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
LN = lambda tag: tag.split('}', 1)[-1]  # local name

# ------------------------ 基础工具 ------------------------
def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def collapse_blank_lines(lines):
    out = []
    for x in lines:
        x2 = x.strip()
        if not x2:
            if out and out[-1] != "":
                out.append("")
        else:
            out.append(x2)
    return out

# 合并相邻的引文块：...[78][79] / [78,][79] -> [78, 79]
def merge_adjacent_citations(txt: str) -> str:
    # [78,][79] -> [78, 79]
    txt = re.sub(r"\[(\d+(?:,\s*\d+)*)\s*,\]\s*\[(\d+(?:,\s*\d+)*)\]", r"[\1, \2]", txt)
    # [78][79] -> [78, 79]
    txt = re.sub(r"\[(\d+(?:,\s*\d+)*)\]\s*\[(\d+(?:,\s*\d+)*)\]", r"[\1, \2]", txt)
    # 去重多余空格与逗号
    txt = txt.replace(", ,", ", ")
    txt = re.sub(r"\[\s*(\d)", r"[\1", txt)
    txt = re.sub(r"(\d)\s*\]", r"\1]", txt)
    return txt

# ------------------------ 内联序列化 ------------------------
def serialize_inline(el) -> str:
    """
    深度序列化内联内容；特殊处理：
    - <ref type="bibr"> → [编号]
    - <hi rend="sup|sub"> → ^text / _{text}
    - <lb/> → 换行；<pb n="x"/> → [Page x]（可按需改为空）
    """
    name = LN(el.tag)
    parts = [el.text or ""]

    for child in list(el):
        cname = LN(child.tag)

        if cname == "ref" and (child.get("type") == "bibr"):
            # 引文 → [78,79]
            txt = clean_space("".join(child.itertext()))
            txt = txt.replace(" ", "")
            parts.append(f"[{txt}]")

        elif cname == "hi":
            rend = (child.get("rend") or "").lower()
            inner = clean_space("".join(child.itertext()))
            if rend == "sup":
                parts.append(f"^{inner}")           # 可改为 ¹²³ 等
            elif rend == "sub":
                parts.append(f"_{{{inner}}}")       # 可改为 下标样式
            else:
                parts.append(clean_space("".join(child.itertext())))

        elif cname == "lb":
            parts.append("\n")

        elif cname == "pb":
            n = child.get("n")
            if n:
                parts.append(f" [Page {n}] ")
            # 若不想要分页标记，改为 pass 即可

        else:
            parts.append(serialize_inline(child))

        parts.append(child.tail or "")

    return "".join(parts)

def text_of_block(el) -> str:
    return clean_space(serialize_inline(el))

# ------------------------ 表格/图/注释 ------------------------
def extract_table(tbl) -> str:
    """转 TSV：每行一行，单元格用制表符分隔"""
    rows = []
    for row in tbl.findall(".//tei:row", TEI_NS):
        cells = [clean_space("".join(c.itertext())) for c in row.findall("./tei:cell", TEI_NS)]
        if any(cells):
            rows.append("\t".join(cells))
    return "\n".join(rows)

def extract_figure(el) -> str:
    # 支持 <figure type="table"> 用 figDesc 当标题
    ftype = (el.get("type") or "").lower()
    desc = el.find(".//tei:figDesc", TEI_NS)
    caption = text_of_block(desc) if desc is not None else text_of_block(el)
    label = "Table" if ftype == "table" else "Figure"
    return f"{label}: {caption}" if caption else ""

# ------------------------ 主转换 ------------------------
def xml_to_text(xml_str: str) -> str:
    root = ET.fromstring(xml_str)
    # 只处理 body（不输出元数据/摘要/参考文献）
    body = root.find(".//tei:body", TEI_NS)
    if body is None:
        # 没有 body 时，尽量从全文提取段落
        body = root

    out_parts = []

    for el in body.iter():
        name = LN(el.tag)

        if name == "head":
            t = text_of_block(el)
            if t:
                out_parts.append(f"## {t}")

        elif name in {"p", "ab"}:
            t = text_of_block(el)
            if t:
                out_parts.append(t)

        elif name == "list":
            # 交由 item 生成行，这里补一个空行分隔
            out_parts.append("")

        elif name == "item":
            # 有序/无序列表
            parent = el.getparent() if hasattr(el, "getparent") else None  # ElementTree 无 getparent
            bullet = "-"  # 简化处理
            t = text_of_block(el)
            if t:
                out_parts.append(f"{bullet} {t}")

        elif name == "table":
            t = extract_table(el)
            if t:
                out_parts.append(t)

        # elif name == "figure":
        #     t = extract_figure(el)
        #     if t:
        #         out_parts.append(t)

        elif name == "note":
            t = text_of_block(el)
            if t:
                out_parts.append(f"[Note] {t}")

        elif name == "formula":
            # 简单公式转行内，复杂公式可自行加 MathML/TeX 处理
            t = clean_space("".join(el.itertext()))
            if t:
                out_parts.append(f"$ {t} $")

    # 清理空行并合并相邻引文
    out = collapse_blank_lines(out_parts)
    txt = "\n\n".join(out)
    txt = merge_adjacent_citations(txt)
    return txt.strip()

# ------------------------ 批处理入口 ------------------------
def convert_folder(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".xml")]
    if not files:
        print(f"No XML files found in {input_dir}")
        return

    for filename in files:
        xml_path = os.path.join(input_dir, filename)
        txt_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
            text = xml_to_text(xml_content)
            with open(txt_path, "w", encoding="utf-8") as out_f:
                out_f.write(text)
            print(f"Converted: {filename} -> {txt_path}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python xml_to_text_esmo.py <input_folder> <output_folder>")
        sys.exit(1)
    convert_folder(sys.argv[1], sys.argv[2])
