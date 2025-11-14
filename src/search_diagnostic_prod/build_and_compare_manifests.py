import argparse
import json
import os
from typing import Dict, List, Optional


def scan_raw_files(raw_root: str) -> List[dict]:
    items = []
    # Expect immediate subfolders as categories (e.g., GI, PA, Cashier)
    for category in os.listdir(raw_root):
        cat_dir = os.path.join(raw_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for doc_id in os.listdir(cat_dir):
            doc_dir = os.path.join(cat_dir, doc_id)
            if not os.path.isdir(doc_dir):
                continue
            for fn in os.listdir(doc_dir):
                abs_path = os.path.join(doc_dir, fn)
                if not os.path.isfile(abs_path):
                    continue
                rel_path = os.path.relpath(abs_path, raw_root)
                items.append(
                    {
                        "category": category,
                        "doc_id": doc_id,
                        "filename": fn,
                        "rel_path": rel_path.replace("\\", "/"),
                        "abs_path": abs_path,
                    }
                )
    return items


def _safe_get(d: dict, path: List[str]) -> Optional[str]:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, str) else None


def _infer_doc_id_from_filename(json_filename: str) -> Optional[str]:
    # Many ETL JSONs follow pattern: DOCID_something.json
    base = os.path.basename(json_filename)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    return parts[0] if parts else None


def scan_etl_files(etl_root: str) -> List[dict]:
    items = []
    for category_dir in os.listdir(etl_root):
        cat_dir = os.path.join(etl_root, category_dir)
        if not os.path.isdir(cat_dir):
            continue
        for fn in os.listdir(cat_dir):
            if not fn.lower().endswith(".json"):
                continue
            abs_path = os.path.join(cat_dir, fn)
            try:
                with open(abs_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception:
                data = {}

            # Try multiple paths to get document_id
            doc_id = (
                _safe_get(data, ["document_identity", "document_id"]) or
                _safe_get(data, ["document_index_metadata", "document_id"]) or
                _safe_get(data, ["document_metadata", "document_id"]) or
                _infer_doc_id_from_filename(fn)
            )

            # Try to get the raw file name from the JSON
            sys_file_name = (
                _safe_get(data, ["system_file_metadata", "sys_file_name"]) or
                _safe_get(data, ["file_index_metadata", "file_name"]) or
                _safe_get(data, ["file_index_metadata", "sys_file_name"]) or
                _safe_get(data, ["document_index_metadata", "file_name"]) or
                _safe_get(data, ["document_identity", "file_name"]) or
                None
            )

            items.append(
                {
                    "etl_category_dir": category_dir,
                    "doc_id": doc_id,
                    "etl_json": fn,
                    "sys_file_name": sys_file_name,
                    "abs_path": abs_path,
                }
            )
    return items


def compare_unmatched(raw_items: List[dict], etl_items: List[dict]) -> List[dict]:
    # Build sets per doc_id for raw and ETL
    raw_by_doc: Dict[str, Dict[str, dict]] = {}
    for it in raw_items:
        doc_id = it.get("doc_id")
        fn = it.get("filename")
        if not doc_id or not fn:
            continue
        raw_by_doc.setdefault(doc_id, {})[fn.lower()] = it

    etl_by_doc: Dict[str, set] = {}
    for it in etl_items:
        doc_id = it.get("doc_id")
        sys_fn = it.get("sys_file_name")
        if not doc_id or not isinstance(sys_fn, str) or not sys_fn:
            continue
        etl_by_doc.setdefault(doc_id, set()).add(sys_fn.lower())

    unmatched = []
    for doc_id, raw_map in raw_by_doc.items():
        etl_set = etl_by_doc.get(doc_id, set())
        for fn_lc, it in raw_map.items():
            if fn_lc not in etl_set:
                unmatched.append(
                    {
                        "doc_id": doc_id,
                        "category": it.get("category"),
                        "filename": it.get("filename"),
                        "rel_path": it.get("rel_path"),
                        "abs_path": it.get("abs_path"),
                        "reason": "no exact filename match in ETL",
                    }
                )
    return unmatched


def main():
    parser = argparse.ArgumentParser(description="Build raw/etl manifests and compare unmatched files")
    parser.add_argument("--raw_root", default=r"artifact\\raw_data")
    parser.add_argument("--etl_root", default=r"artifact\\etl_result_json")
    parser.add_argument("--out_dir", default=r"outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw_items = scan_raw_files(args.raw_root)
    raw_manifest_path = os.path.join(args.out_dir, "raw_files_manifest.json")
    with open(raw_manifest_path, "w", encoding="utf-8-sig") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)

    etl_items = scan_etl_files(args.etl_root)
    etl_manifest_path = os.path.join(args.out_dir, "etl_files_manifest.json")
    with open(etl_manifest_path, "w", encoding="utf-8-sig") as f:
        json.dump(etl_items, f, ensure_ascii=False, indent=2)

    unmatched = compare_unmatched(raw_items, etl_items)
    unmatched_path = os.path.join(args.out_dir, "unmatched_files.json")
    with open(unmatched_path, "w", encoding="utf-8-sig") as f:
        json.dump(unmatched, f, ensure_ascii=False, indent=2)

    print(f"Saved raw manifest: {raw_manifest_path}")
    print(f"Saved ETL manifest: {etl_manifest_path}")
    print(f"Saved unmatched list: {unmatched_path}")


if __name__ == "__main__":
    main()