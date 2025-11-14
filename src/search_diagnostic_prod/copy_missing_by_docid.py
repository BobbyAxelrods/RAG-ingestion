import argparse
import os
import shutil
from collections import defaultdict
from typing import Dict, Tuple, List, Set


def normalize_category(name: str) -> str:
    n = name.strip().lower()
    if n in {"attachment", "pa", "pa_bulk_unique"}:
        return "PA"
    if n in {"general insurance", "gi", "gi_bulk_unique", "ga", "general_insurance"}:
        return "GI"
    if n in {"cashier", "cashier_bulk_unique"}:
        return "Cashier"
    return name


def count_raw_by_docid(raw_root: str) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"Raw root not found: {raw_root}")
    for cat_name in os.listdir(raw_root):
        cat_path = os.path.join(raw_root, cat_name)
        if not os.path.isdir(cat_path):
            continue
        category = normalize_category(cat_name)
        for docid_name in os.listdir(cat_path):
            docid_path = os.path.join(cat_path, docid_name)
            if not os.path.isdir(docid_path):
                continue
            file_count = 0
            for root, _dirs, files in os.walk(docid_path):
                file_count += len(files)
            counts[(category, docid_name)] += file_count
    return counts


def count_etl_by_docid(etl_root: str) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    if not os.path.isdir(etl_root):
        raise FileNotFoundError(f"ETL root not found: {etl_root}")
    for etl_cat in os.listdir(etl_root):
        etl_cat_path = os.path.join(etl_root, etl_cat)
        if not os.path.isdir(etl_cat_path):
            continue
        category = normalize_category(etl_cat)
        for name in os.listdir(etl_cat_path):
            if not name.lower().endswith('.json'):
                continue
            stem = os.path.splitext(name)[0]
            docid = stem.split('_')[0] if '_' in stem else stem
            counts[(category, docid)] += 1
    return counts


def list_raw_files_for_docid(raw_root: str, category: str, docid: str) -> List[str]:
    cat_folder = {"PA": "PA", "GI": "GI", "Cashier": "Cashier"}.get(category, category)
    base_path = os.path.join(raw_root, cat_folder, docid)
    files: List[str] = []
    if os.path.isdir(base_path):
        for root, _dirs, fns in os.walk(base_path):
            for fn in fns:
                files.append(os.path.join(root, fn))
    return files


def loose_normalize_stem(name: str) -> str:
    # Lowercase, strip extension, collapse non-alphanumeric to spaces, then single spaces
    stem = os.path.splitext(name)[0].lower()
    out_chars = []
    for ch in stem:
        if ch.isalnum():
            out_chars.append(ch)
        else:
            out_chars.append(' ')
    normalized = ''.join(out_chars)
    normalized = ' '.join(normalized.split())
    return normalized


def get_etl_raw_names_for_docid(etl_root: str, category: str, docid: str) -> Set[str]:
    """Return the exact raw filenames (case-insensitive) present in ETL JSON for a DocId.
    This reads each JSON under the category folder with filename prefix '<DocId>_' and
    collects 'system_file_metadata.sys_file_name' (falls back to 'file_index_metadata.file_name').
    """
    # Local import to avoid patching top-level imports unnecessarily
    import json

    cat_map = {"PA": "pa_bulk_unique", "GI": "gi_bulk_unique", "Cashier": "cashier_bulk_unique"}
    etl_cat_folder = cat_map.get(category, category)
    etl_cat_path = os.path.join(etl_root, etl_cat_folder)
    names: Set[str] = set()
    if not os.path.isdir(etl_cat_path):
        return names
    prefix = f"{docid}_".lower()
    for name in os.listdir(etl_cat_path):
        if not name.lower().endswith('.json'):
            continue
        stem = os.path.splitext(name)[0]
        if not stem.lower().startswith(prefix):
            continue
        fp = os.path.join(etl_cat_path, name)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_name = None
            # Prefer system_file_metadata.sys_file_name
            sfm = data.get("system_file_metadata", {}) if isinstance(data, dict) else {}
            raw_name = sfm.get("sys_file_name")
            if not raw_name:
                fim = data.get("file_index_metadata", {}) if isinstance(data, dict) else {}
                raw_name = fim.get("file_name")
            if raw_name:
                names.add(str(raw_name).casefold())
        except Exception:
            # If one JSON cannot be read, skip it
            continue
    return names


def _tokens(s: str) -> Set[str]:
    return set(s.split())


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def select_missing_raw_files(raw_files: List[str], etl_raw_names: Set[str], deficit: int) -> List[str]:
    """Select raw files whose base filename is not present in ETL JSON for this DocId.
    Comparison is case-insensitive and uses the exact filename (including extension).
    Returns up to 'deficit' files (or all missing if deficit <= 0).
    """
    missing: List[str] = []
    etl_names_ci = {n.casefold() for n in etl_raw_names}
    for path in raw_files:
        base = os.path.basename(path).casefold()
        if base not in etl_names_ci:
            missing.append(path)
    if deficit > 0:
        return missing[:deficit]
    return missing


def copy_files(files: List[str], raw_root: str, category: str, docid: str, out_root: str) -> int:
    # Copy preserving category/docid folder structure under out_root
    copied = 0
    target_base = os.path.join(out_root, category, docid)
    os.makedirs(target_base, exist_ok=True)
    for src in files:
        # Compute relative path from docid folder to preserve any substructure
        # raw path looks like <raw_root>/<Category>/<DocId>/...; get rel to that base
        cat_folder = {"PA": "PA", "GI": "GI", "Cashier": "Cashier"}.get(category, category)
        base_path = os.path.join(raw_root, cat_folder, docid)
        rel = os.path.relpath(src, base_path)
        dst = os.path.join(target_base, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description="Copy raw files for DocIds with deficits into missing_file folder.")
    parser.add_argument('--raw_dir', required=False, default=r"c:\\Users\\muhammad.s.safian\\OneDrive - Avanade\\Project\\MAIN_PROJECT\\_avanade_main\\_avanade_main\\_01_prudential\\PRUHK-AgenticRAG-V1\\artifact\\raw_data",
                        help='Path to raw_data root (PA, GI, Cashier).')
    parser.add_argument('--etl_dir', required=False, default=r"c:\\Users\\muhammad.s.safian\\OneDrive - Avanade\\Project\\MAIN_PROJECT\\_avanade_main\\_avanade_main\\_01_prudential\\PRUHK-AgenticRAG-V1\\artifact\\etl_result_json",
                        help='Path to etl_result_json root.')
    parser.add_argument('--out_dir', required=False, default=r"c:\\Users\\muhammad.s.safian\\OneDrive - Avanade\\Project\\MAIN_PROJECT\\_avanade_main\\_avanade_main\\_01_prudential\\PRUHK-AgenticRAG-V1\\artifact\\missing_file",
                        help='Folder to copy missing raw files into, organized by Category/DocId.')
    parser.add_argument('--clean_dest', action='store_true', help='Remove existing destination DocId folder before copying (safe refresh).')
    args = parser.parse_args()

    raw_counts = count_raw_by_docid(args.raw_dir)
    etl_counts = count_etl_by_docid(args.etl_dir)

    deficits = []
    for (cat, docid), rcount in raw_counts.items():
        ecount = etl_counts.get((cat, docid), 0)
        diff = rcount - ecount
        if diff > 0:
            deficits.append((cat, docid, rcount, ecount, diff))

    # Copy files for each deficit DocId
    total_files_copied = 0
    total_docids = 0
    for cat, docid, rcount, ecount, diff in sorted(deficits, key=lambda x: (x[0], -x[4], x[1])):
        raw_files = list_raw_files_for_docid(args.raw_dir, cat, docid)
        etl_raw_names = get_etl_raw_names_for_docid(args.etl_dir, cat, docid)
        missing_files = select_missing_raw_files(raw_files, etl_raw_names, diff)
        # Optional clean of destination to avoid stale or mis-copied files
        if args.clean_dest:
            dest_base = os.path.join(args.out_dir, cat, docid)
            if os.path.isdir(dest_base):
                shutil.rmtree(dest_base)
        # If our matching yields fewer than the deficit, still copy what we identified
        copied = copy_files(missing_files, args.raw_dir, cat, docid, args.out_dir)
        total_files_copied += copied
        total_docids += 1
        note = ""
        if len(missing_files) != diff:
            note = f" [note: identified={len(missing_files)} deficit={diff}]"
        print(f"Copied {copied} missing files for {cat} {docid} (raw={rcount} etl={ecount} deficit={diff}){note}")

    print(f"Done. Copied {total_files_copied} files across {total_docids} DocIds into: {args.out_dir}")


if __name__ == '__main__':
    main()