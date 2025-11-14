import argparse
import csv
import os
import re
import sys
import unicodedata
from pathlib import Path
import shutil


def normalize_for_match(value: str) -> str:
    if value is None:
        return ""
    # Unicode normalize and casefold (handles TC/EN and symbols)
    text = unicodedata.normalize("NFKC", value).casefold()
    # Strip common file extensions
    text = re.sub(r"\.(pdf|docx?|xlsx?|pptx?)$", "", text)
    # Replace non-word characters with underscores
    text = re.sub(r"[^\w]+", "_", text)
    # Collapse underscores
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def collect_etl_json_files(root_dir: str):
    root = Path(root_dir)
    files = []
    for p in root.rglob("*.json"):
        files.append(p)
    return files


def build_index(files):
    # Build maps for quick matching
    # 1) Map by normalized base name (without leading doc id prefix if present)
    # 2) Map by (doc_id, normalized base name)
    by_base = {}
    by_doc_and_base = {}

    for p in files:
        name = p.name
        stem = p.stem  # filename without .json
        # Attempt to split leading doc id (e.g., CA000001_, GI000033_, PA000141_)
        m = re.match(r"^([A-Z]{2}\d{6})_(.+)$", stem)
        if m:
            doc_id = m.group(1)
            remainder = m.group(2)
        else:
            doc_id = None
            remainder = stem

        base_norm = normalize_for_match(remainder)

        by_base.setdefault(base_norm, []).append(p)
        if doc_id:
            by_doc_and_base.setdefault((doc_id, base_norm), []).append(p)

    return by_base, by_doc_and_base


def detect_columns(header):
    lower = [h.casefold() for h in header]
    col = {}
    for key in ["row_index", "index", "row"]:
        if key in lower:
            col["row_index"] = lower.index(key)
            break
    for key in ["doc_id", "docid", "document_id"]:
        if key in lower:
            col["doc_id"] = lower.index(key)
            break
    for key in ["title", "document_title", "filename", "file_name"]:
        if key in lower:
            col["title"] = lower.index(key)
            break
    return col


def main():
    parser = argparse.ArgumentParser(description="Copy ETL JSON files whose filenames match titles in a CSV, ignoring doc_id if needed.")
    parser.add_argument("--csv", required=True, help="Path to ground truth CSV (with title column)")
    parser.add_argument("--etl_dir", required=True, help="Root directory of ETL JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory to copy matched JSON files into")
    parser.add_argument("--prefer_doc_id", action="store_true", help="Prefer matching with doc_id prefix when available")
    args = parser.parse_args()

    etl_files = collect_etl_json_files(args.etl_dir)
    by_base, by_doc_and_base = build_index(etl_files)

    # Prepare output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV rows
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_map = detect_columns(header)
        if "title" not in col_map:
            print("ERROR: CSV must contain a 'title' column.")
            print("Detected header:", header)
            sys.exit(1)
        # doc_id is optional

        total = 0
        matched = 0
        not_found = 0
        copied = 0

        for row in reader:
            total += 1
            title = (row[col_map["title"]] or "").strip()
            doc_id = (row[col_map["doc_id"]] if "doc_id" in col_map else "").strip() or None

            norm_title = normalize_for_match(title)

            candidates = []
            if args.prefer_doc_id and doc_id:
                candidates = by_doc_and_base.get((doc_id, norm_title), [])
            if not candidates:
                candidates = by_base.get(norm_title, [])

            if not candidates:
                not_found += 1
                print(f"NOT FOUND: title='{title}' doc_id='{doc_id or ''}'")
                continue

            matched += 1
            for p in candidates:
                dest = out_dir / p.name
                try:
                    shutil.copy2(p, dest)
                    copied += 1
                    print(f"COPIED: {p} -> {dest}")
                except Exception as e:
                    print(f"ERROR copying {p} -> {dest}: {e}")

        print("Summary:", f"total_rows={total}", f"matched_rows={matched}", f"files_copied={copied}", f"not_found={not_found}")


if __name__ == "__main__":
    main()