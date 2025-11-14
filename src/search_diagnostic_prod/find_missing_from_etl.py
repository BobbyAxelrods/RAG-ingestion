import argparse
import csv
import os
import re
import unicodedata
from pathlib import Path


def normalize_for_match(value: str) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"\.(pdf|docx?|xlsx?|pptx?)$", "", text)
    text = re.sub(r"[^\w]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def exact_for_match(value: str) -> str:
    """Exact filename stem comparison (case-insensitive), no punctuation normalization."""
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", value).casefold()


def simple_for_match(value: str) -> str:
    """Case-insensitive, strip common doc extensions only; keep original punctuation/spaces."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"\.(pdf|docx?|xlsx?|pptx?)$", "", text)
    return text


def collect_raw_files(raw_root: str, include_exts=None):
    # include_exts: None means include ALL files; otherwise a set of extensions
    exts = include_exts
    root = Path(raw_root)
    rows = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if exts is not None and p.suffix.casefold() not in {e.casefold() for e in exts}:
            continue
        # category is the first-level dir under raw_root
        rel = p.relative_to(root)
        parts = rel.parts
        category = parts[0] if len(parts) > 0 else ""
        # doc_id detection: handle both raw_root at project level and category level
        doc_id = None
        if len(parts) > 0 and re.match(r"^[A-Z]{2}\d{6}$", parts[0]):
            doc_id = parts[0]
        elif len(parts) > 1 and re.match(r"^[A-Z]{2}\d{6}$", parts[1]):
            doc_id = parts[1]
        base_norm = normalize_for_match(p.stem)
        rows.append({
            "category": category,
            "doc_id": doc_id,
            "raw_filename": p.name,
            "raw_path": str(p),
            "norm_title": base_norm,
        })
    return rows


def collect_etl_json_files(etl_root: str):
    root = Path(etl_root)
    files = list(root.rglob("*.json"))
    return files


def build_etl_index(files):
    by_base = {}
    by_doc_and_base = {}
    by_base_exact = {}
    by_doc_and_base_exact = {}
    by_base_simple = {}
    by_doc_and_base_simple = {}
    for p in files:
        stem = p.stem
        m = re.match(r"^([A-Z]{2}\d{6})_(.+)$", stem)
        if m:
            doc_id = m.group(1)
            remainder = m.group(2)
        else:
            doc_id = None
            remainder = stem
        base_norm = normalize_for_match(remainder)
        base_exact = exact_for_match(remainder)
        base_simple = simple_for_match(remainder)
        by_base.setdefault(base_norm, []).append(p)
        by_base_exact.setdefault(base_exact, []).append(p)
        by_base_simple.setdefault(base_simple, []).append(p)
        if doc_id:
            by_doc_and_base.setdefault((doc_id, base_norm), []).append(p)
            by_doc_and_base_exact.setdefault((doc_id, base_exact), []).append(p)
            by_doc_and_base_simple.setdefault((doc_id, base_simple), []).append(p)
    return by_base, by_doc_and_base, by_base_exact, by_doc_and_base_exact, by_base_simple, by_doc_and_base_simple


def collect_etl_basenorms(etl_root: str):
    """Collect a flat list of ETL entries as (doc_id, base_norm, base_exact)."""
    entries = []
    root = Path(etl_root)
    for p in root.rglob("*.json"):
        stem = p.stem
        m = re.match(r"^([A-Z]{2}\d{6})_(.+)$", stem)
        if m:
            doc_id = m.group(1)
            remainder = m.group(2)
        else:
            doc_id = None
            remainder = stem
        entries.append({
            "doc_id": doc_id,
            "base_norm": normalize_for_match(remainder),
            "base_exact": exact_for_match(remainder),
        })
    return entries


def collect_etl_basenorms_per_category(etl_root: str):
    """Collect ETL base lists per immediate subdirectory (normalized category name)."""
    root = Path(etl_root)
    per_cat = {}
    if not root.exists():
        return per_cat
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        sub_norm = normalize_category(sub.name)
        base = re.sub(r"_(bulk|unique|bulk_unique)$", "", sub_norm)
        entries = []
        for p in sub.rglob("*.json"):
            stem = p.stem
            m = re.match(r"^([A-Z]{2}\d{6})_(.+)$", stem)
            if m:
                doc_id = m.group(1)
                remainder = m.group(2)
            else:
                doc_id = None
                remainder = stem
            entries.append({
                "doc_id": doc_id,
                "base_norm": normalize_for_match(remainder),
                "base_exact": exact_for_match(remainder),
            })
        per_cat[base] = entries
    # alias mapping
    alias_map = {
        "general_insurance": "gi",
        "ga": "gi",
    }
    for alias_from, alias_to in alias_map.items():
        if alias_from not in per_cat and alias_to in per_cat:
            per_cat[alias_from] = per_cat[alias_to]
    return per_cat


def normalize_category(name: str) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", name).casefold()
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def build_etl_index_per_category(etl_root: str):
    """Scan immediate subdirectories under etl_root and build per-category filename index.
    Category key is derived from subdir name (normalized). For example, 'cashier_bulk_unique' -> 'cashier'.
    """
    root = Path(etl_root)
    per_cat_index = {}
    if not root.exists():
        return per_cat_index
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        sub_norm = normalize_category(sub.name)
        # heuristics: remove suffixes like _bulk_unique, _unique, _bulk
        base = re.sub(r"_(bulk|unique|bulk_unique)$", "", sub_norm)
        # build index for this subdir
        files = list(sub.rglob("*.json"))
        by_base, by_doc_and_base, by_base_exact, by_doc_and_base_exact, by_base_simple, by_doc_and_base_simple = build_etl_index(files)
        index_obj = {
            "by_base": by_base,
            "by_doc": by_doc_and_base,
            "by_base_exact": by_base_exact,
            "by_doc_exact": by_doc_and_base_exact,
            "by_base_simple": by_base_simple,
            "by_doc_simple": by_doc_and_base_simple,
        }
        per_cat_index[base] = index_obj
        # add common alias keys to point to the same index
        alias_map = {
            "general_insurance": "gi",
            "ga": "gi",
        }
        for alias_from, alias_to in alias_map.items():
            if base == alias_to and alias_from not in per_cat_index:
                per_cat_index[alias_from] = index_obj
    return per_cat_index


def main():
    parser = argparse.ArgumentParser(description="Find raw_data files that have not yet been extracted to ETL JSON.")
    parser.add_argument("--raw_dir", required=True, help="Path to raw_data root directory")
    parser.add_argument("--etl_dir", required=True, help="Path to etl_result_json root directory")
    parser.add_argument("--out_csv", required=True, help="Output CSV path for missing files list")
    parser.add_argument("--prefer_doc_id", action="store_true", help="Prefer matching with doc_id where available")
    parser.add_argument("--no_fallback", action="store_true", help="Do not fallback to global filename match when doc_id match fails")
    parser.add_argument("--exts", help="Comma-separated list of file extensions to include (e.g., .pdf,.docx). If omitted, include ALL files.")
    parser.add_argument("--scope", choices=["global", "category"], default="global", help="Matching scope: 'global' searches all ETL JSONs; 'category' restricts matches to the corresponding ETL subfolder.")
    parser.add_argument("--match_mode", choices=["normalized", "simple", "exact", "contains"], default="normalized", help="'simple' strips extensions and lowercases only; 'exact' compares stems case-insensitively; 'contains' matches if either stem contains the other.")
    args = parser.parse_args()

    include_exts = None
    if args.exts:
        include_exts = set([e.strip() for e in args.exts.split(",") if e.strip()])
    raw_rows = collect_raw_files(args.raw_dir, include_exts=include_exts)
    etl_files = collect_etl_json_files(args.etl_dir)
    by_base, by_doc_and_base, by_base_exact, by_doc_and_base_exact, by_base_simple, by_doc_and_base_simple = build_etl_index(etl_files)
    per_category_index = build_etl_index_per_category(args.etl_dir)
    etl_entries = collect_etl_basenorms(args.etl_dir)
    etl_entries_per_cat = collect_etl_basenorms_per_category(args.etl_dir)

    missing = []
    matched_count = 0

    for r in raw_rows:
        candidates = []
        # Determine matching index based on scope
        # choose index maps based on match_mode
        if args.match_mode == "normalized":
            use_by_base = by_base
            use_by_doc = by_doc_and_base
        elif args.match_mode == "simple":
            use_by_base = by_base_simple
            use_by_doc = by_doc_and_base_simple
        else:
            use_by_base = by_base_exact
            use_by_doc = by_doc_and_base_exact
        use_entries = etl_entries
        if args.scope == "category":
            # attempt to find per-category index
            cat_key = normalize_category(r["category"]) if r["category"] else ""
            # Try direct match, or if category is like 'pa' vs subfolder 'pa'
            if cat_key in per_category_index:
                if args.match_mode == "normalized":
                    use_by_base = per_category_index[cat_key]["by_base"]
                    use_by_doc = per_category_index[cat_key]["by_doc"]
                elif args.match_mode == "simple":
                    use_by_base = per_category_index[cat_key]["by_base_simple"]
                    use_by_doc = per_category_index[cat_key]["by_doc_simple"]
                else:
                    use_by_base = per_category_index[cat_key]["by_base_exact"]
                    use_by_doc = per_category_index[cat_key]["by_doc_exact"]
                use_entries = etl_entries_per_cat.get(cat_key, [])
            else:
                # No matching category folder; keep empty indexes to force missing unless global fallback allowed
                use_by_base = {}
                use_by_doc = {}
                use_entries = []

        # matching
        if args.match_mode == "contains":
            raw_key_norm = r["norm_title"]
            raw_key_exact = exact_for_match(Path(r["raw_path"]).stem)
            # constrain by doc_id if preferred
            def entry_matches(entry):
                if args.prefer_doc_id and r["doc_id"] and entry["doc_id"] != r["doc_id"]:
                    return False
                # check contains on both normalized and exact to be permissive
                return (
                    raw_key_norm in entry["base_norm"] or entry["base_norm"] in raw_key_norm or
                    raw_key_exact in entry["base_exact"] or entry["base_exact"] in raw_key_exact
                )
            candidates = [e for e in use_entries if entry_matches(e)]
            # no fallback concept here; contains is the fallback by nature
        else:
            # choose key based on match_mode
            if args.match_mode == "normalized":
                key = r["norm_title"]
            elif args.match_mode == "simple":
                key = simple_for_match(Path(r["raw_path"]).stem)
            else:
                key = exact_for_match(Path(r["raw_path"]).stem)

            if args.prefer_doc_id and r["doc_id"]:
                candidates = use_by_doc.get((r["doc_id"], key), [])
            if not candidates and not args.no_fallback:
                candidates = use_by_base.get(key, [])

        if candidates:
            matched_count += 1
        else:
            missing.append(r)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "doc_id", "raw_filename", "raw_path"])  # concise columns
        for m in missing:
            writer.writerow([m["category"], m["doc_id"] or "", m["raw_filename"], m["raw_path"]])

    total_raw = len(raw_rows)
    total_etl = len(etl_files)
    print(f"Summary: total_raw_files={total_raw} total_etl_json={total_etl} matched_raw_files={matched_count} missing_count={len(missing)}")
    print(f"Missing list saved to: {out_path}")


if __name__ == "__main__":
    main()