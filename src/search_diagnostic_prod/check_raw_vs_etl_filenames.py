from pathlib import Path
import json
import csv
import argparse
import unicodedata
import re


def normalize_basic(name: str) -> str:
    """Basic normalization: lowercase, remove common extensions, collapse underscores/dashes and whitespace."""
    s = unicodedata.normalize("NFKC", name or "").strip().lower()
    # Remove common extensions
    for ext in (".pdf", ".doc", ".docx", ".ppt", ".pptx"):
        if s.endswith(ext):
            s = s[: -len(ext)]
            break
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


def normalize_punctless(name: str) -> str:
    """Aggressive normalization: lowercase, remove extensions, replace any non-alphanumeric with space, collapse."""
    s = unicodedata.normalize("NFKC", name or "").strip().lower()
    for ext in (".pdf", ".doc", ".docx", ".ppt", ".pptx"):
        if s.endswith(ext):
            s = s[: -len(ext)]
            break
    # Replace any non-alphanumeric with space, to align versions like v1.0 vs v1_0
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = " ".join(s.split())
    return s


def list_raw_files(raw_root: Path):
    results = []
    for category_dir in raw_root.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for docid_dir in category_dir.iterdir():
            if not docid_dir.is_dir():
                # Files directly under category root
                if docid_dir.is_file():
                    results.append({
                        "category": category,
                        "docId": "",
                        "rawPath": str(docid_dir),
                        "rawFileName": docid_dir.name,
                    })
                continue
            doc_id = docid_dir.name
            for f in docid_dir.iterdir():
                if f.is_file():
                    results.append({
                        "category": category,
                        "docId": doc_id,
                        "rawPath": str(f),
                        "rawFileName": f.name,
                    })
    return results


def _candidate_names_from_json(data: dict, json_file: Path):
    candidates = []
    # Primary fields
    for key in ("file_name", "source_file_name"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())
    # System metadata
    sysm = data.get("system_file_metadata") or {}
    for key in ("sys_file_name", "sys_file_path"):
        val = sysm.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())
    # Index metadata
    fim = data.get("file_index_metadata") or {}
    for key in ("file_name", "item_url"):
        val = fim.get(key)
        if isinstance(val, str) and val.strip():
            # If it's a URL/path, take the basename
            p = Path(val)
            candidates.append(p.name.strip())
    # Always add derived tail from JSON filename after first underscore
    base = json_file.stem
    parts = base.split("_", 1)
    tail = parts[1] if len(parts) > 1 else base
    candidates.append(tail)
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for c in candidates:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            uniq.append(c)
    return uniq


def index_etl_json(etl_root: Path):
    exact_index = {}
    normalized_index_basic = {}
    normalized_index_punctless = {}
    all_json_paths = []
    for category_dir in etl_root.iterdir():
        if not category_dir.is_dir():
            continue
        for json_file in category_dir.glob("*.json"):
            all_json_paths.append(str(json_file))
            try:
                with json_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                data = {}
            candidates = _candidate_names_from_json(data, json_file)
            for cand in candidates:
                key_exact = cand.strip().lower()
                exact_index.setdefault(key_exact, []).append(str(json_file))
                key_basic = normalize_basic(cand)
                normalized_index_basic.setdefault(key_basic, []).append(str(json_file))
                key_punct = normalize_punctless(cand)
                normalized_index_punctless.setdefault(key_punct, []).append(str(json_file))
    return exact_index, normalized_index_basic, normalized_index_punctless, all_json_paths


def compare_raw_to_etl(raw_root: Path, etl_root: Path):
    raw_files = list_raw_files(raw_root)
    exact_index, norm_basic_idx, norm_punct_idx, _ = index_etl_json(etl_root)
    results = []
    for item in raw_files:
        raw_name = item["rawFileName"]
        exact_key = raw_name.strip().lower()
        basic_key = normalize_basic(raw_name)
        punct_key = normalize_punctless(raw_name)
        matched_exact = exact_index.get(exact_key, [])
        matched_basic = norm_basic_idx.get(basic_key, [])
        matched_punct = norm_punct_idx.get(punct_key, [])
        exists_exact = bool(matched_exact)
        exists_basic = bool(matched_basic)
        exists_punct = bool(matched_punct)
        # Determine best match and strategy
        if exists_exact:
            matched_paths = matched_exact
            strategy = "exact"
        elif exists_basic:
            matched_paths = matched_basic
            strategy = "basic"
        elif exists_punct:
            matched_paths = matched_punct
            strategy = "punctless"
        else:
            matched_paths = []
            strategy = "none"
        results.append({
            "category": item["category"],
            "docId": item["docId"],
            "rawFileName": raw_name,
            "existsExact": exists_exact,
            "existsNormalized": exists_basic,
            "existsPunctless": exists_punct,
            "existsAny": exists_exact or exists_basic or exists_punct,
            "matchStrategy": strategy,
            "matchedEtlJsonPaths": matched_paths,
        })
    return results


def write_reports(results, outputs_dir: Path):
    outputs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = outputs_dir / "raw_vs_etl_filename_check.csv"
    json_path = outputs_dir / "raw_vs_etl_filename_check.json"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "category",
                "docId",
                "rawFileName",
                "existsExact",
                "existsNormalized",
                "existsPunctless",
                "existsAny",
                "matchStrategy",
                "matchedEtlJsonPaths",
            ],
        )
        writer.writeheader()
        for r in results:
            row = r.copy()
            row["matchedEtlJsonPaths"] = ";".join(r.get("matchedEtlJsonPaths", []))
            writer.writerow(row)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    return str(csv_path), str(json_path)


def main():
    parser = argparse.ArgumentParser(description="Check raw filenames against ETL JSON outputs.")
    parser.add_argument(
        "--raw-root",
        default=str(Path("artifact") / "raw_data"),
        help="Root folder of raw data",
    )
    parser.add_argument(
        "--etl-root",
        default=str(Path("artifact") / "etl_result_json"),
        help="Root folder of ETL JSON outputs",
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(Path("outputs")),
        help="Directory to write CSV and JSON reports",
    )
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    etl_root = Path(args.etl_root)
    outputs_dir = Path(args.outputs_dir)
    results = compare_raw_to_etl(raw_root, etl_root)
    write_reports(results, outputs_dir)


if __name__ == "__main__":
    main()