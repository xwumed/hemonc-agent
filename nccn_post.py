#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

PAGE_RE = re.compile(r"^(?P<prefix>.+)_page_(?P<num>\d+)\.txt$", re.IGNORECASE)

def parse_page_info(path: Path):
    """
    返回 (prefix, page_num)；若不匹配 *_page_*.txt 返回 (None, None)
    prefix 用于将同一文档的多页进行分组
    """
    m = PAGE_RE.match(path.name)
    if not m:
        return None, None
    return m.group("prefix"), int(m.group("num"))

def read_text_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def should_delete_basic(path: Path) -> bool:
    """
    基础删除规则（逐页判断）：
      1) 文件名为 *_page_1.txt
      2) 文件内容包含严格大小写 'REFERENCE'
    """
    prefix, page_num = parse_page_info(path)
    if prefix is None:
        return False

    if page_num == 1:
        return True  # 规则1

    text = read_text_safe(path)
    if text is None:
        # 读不到内容就不删，避免误伤
        return False

    # 规则2：严格大小写匹配
    return "REFERENCE" in text

def collect_updates_cutoff(files_in_doc: list[Path]) -> int | None:
    """
    针对“UPDATES”规则，计算需要删到的最大页码（含）：
    逻辑：
      - 找到最早出现“UPDATES”的页 i
      - 向后扩展连续“UPDATES”的区间到 j（i..j 连续页都含“UPDATES”）
      - 返回 j，表示需要删除 <= j 的所有页
    若该文档没有任何“UPDATES”，返回 None
    """
    # 先按页码排序
    items = []
    for p in files_in_doc:
        prefix, n = parse_page_info(p)
        if prefix is None:
            continue
        items.append((n, p))
    items.sort(key=lambda x: x[0])

    # 预读是否含“UPDATES”
    has_updates = {n: ("UPDATES" in (read_text_safe(p) or "")) for n, p in items}

    # 找到最早出现“UPDATES”的页 i
    first = None
    for n, _ in items:
        if has_updates.get(n, False):
            first = n
            break
    if first is None:
        return None

    # 向后扩展连续区间到 j
    j = first
    n_set = set(n for n, _ in items)
    cur = first
    while True:
        nxt = cur + 1
        if (nxt in n_set) and has_updates.get(nxt, False):
            j = nxt
            cur = nxt
        else:
            break
    return j

def postprocess(root_dir: Path, dry_run: bool = False):
    if not root_dir.exists():
        print(f"❌ 路径不存在：{root_dir}")
        return

    # 递归查找所有 *_page_*.txt
    txt_files = list(root_dir.rglob("*_page_*.txt"))
    if not txt_files:
        print("未找到任何 *_page_*.txt 文件。")
        return

    # 按文档前缀分组
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in txt_files:
        prefix, page_num = parse_page_info(f)
        if prefix is not None:
            # 用绝对前缀（含目录）避免不同目录同名前缀混淆
            abs_prefix = str(f.parent / prefix)
            groups[abs_prefix].append(f)

    to_delete: set[Path] = set()

    # 1) 基础规则逐页加入
    for f in txt_files:
        if should_delete_basic(f):
            to_delete.add(f)

    # 2) “UPDATES” 规则：对每个文档计算需要删到的最大页码 j
    for abs_prefix, files_in_doc in groups.items():
        cutoff = collect_updates_cutoff(files_in_doc)  # j 或 None
        if cutoff is None:
            continue
        # 删除该文档中所有页码 <= cutoff 的文件
        for p in files_in_doc:
            _, n = parse_page_info(p)
            if n is not None and n <= cutoff:
                to_delete.add(p)

    if not to_delete:
        print("没有需要删除的文件。")
        return

    to_delete_sorted = sorted(to_delete, key=lambda p: (str(p.parent), parse_page_info(p)[0] or "", parse_page_info(p)[1] or 0))
    print(f"将处理 {len(to_delete_sorted)} 个文件：")
    for f in to_delete_sorted:
        print("DELETE  ", f)
        if not dry_run:
            try:
                f.unlink()
            except Exception as e:
                print(f"  ⚠️ 删除失败：{f} — {e}")

if __name__ == "__main__":
    # 用法：
    #   python cleanup_pages.py extracted_content            # 直接删除
    #   python cleanup_pages.py extracted_content --dry-run  # 只显示将删除哪些文件，不真正删除
    if len(sys.argv) < 2:
        print("Usage: python cleanup_pages.py <extracted_content_root> [--dry-run]")
        sys.exit(1)

    root = Path(sys.argv[1])
    dry = ("--dry-run" in sys.argv[2:])
    postprocess(root, dry_run=dry)
