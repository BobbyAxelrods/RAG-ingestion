import os
import csv
import json
from collections import defaultdict
from typing import Dict, Tuple, List


def normalize_category(name: str) -> str:
    n = (name or "").strip().lower()
    if n in {"pa", "pa_bulk_unique"}:
        return "PA"
    if n in {"gi", "gi_bulk_unique"}:
        return "GI"
    if n in {"cashier", "cashier_bulk_unique"}:
        return "Cashier"
    # Fallback to original, capitalized
    return name.strip()


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
            if not name.lower().endswith(".json"):
                continue
            stem = os.path.splitext(name)[0]
            docid = stem.split("_")[0] if "_" in stem else stem
            counts[(category, docid)] += 1
    return counts


def compare_counts(raw_counts: Dict[Tuple[str, str], int],
                   etl_counts: Dict[Tuple[str, str], int]) -> List[dict]:
    rows: List[dict] = []
    keys = set(raw_counts.keys()) | set(etl_counts.keys())
    for (cat, docid) in keys:
        r = raw_counts.get((cat, docid), 0)
        e = etl_counts.get((cat, docid), 0)
        rows.append({
            "Category": cat,
            "DocId": docid,
            "RawCount": r,
            "EtlCount": e,
            "Deficit": r - e,
        })
    rows.sort(key=lambda x: (x["Category"], -x["Deficit"], x["DocId"]))
    return rows


def totals_by_category(raw_counts: Dict[Tuple[str, str], int],
                       etl_counts: Dict[Tuple[str, str], int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    totals_raw: Dict[str, int] = defaultdict(int)
    totals_etl: Dict[str, int] = defaultdict(int)
    for (cat, _docid), r in raw_counts.items():
        totals_raw[cat] += r
    for (cat, _docid), e in etl_counts.items():
        totals_etl[cat] += e
    return dict(totals_raw), dict(totals_etl)


def write_csv(path: str, rows: List[dict], headers: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_diagnostic(raw_dir: str,
                   etl_dir: str,
                   out_dir: str = "outputs") -> dict:
    raw_counts = count_raw_by_docid(raw_dir)
    etl_counts = count_etl_by_docid(etl_dir)
    comparison = compare_counts(raw_counts, etl_counts)
    totals_raw, totals_etl = totals_by_category(raw_counts, etl_counts)
    deficits = [r for r in comparison if r["Deficit"] > 0]

    raw_rows = [
        {"Category": cat, "DocId": docid, "RawCount": raw_counts[(cat, docid)]}
        for (cat, docid) in sorted(raw_counts.keys())
    ]
    etl_rows = [
        {"Category": cat, "DocId": docid, "EtlCount": etl_counts[(cat, docid)]}
        for (cat, docid) in sorted(etl_counts.keys())
    ]

    write_csv(os.path.join(out_dir, "raw_docid_counts.csv"), raw_rows, ["Category", "DocId", "RawCount"])
    write_csv(os.path.join(out_dir, "etl_docid_counts.csv"), etl_rows, ["Category", "DocId", "EtlCount"])
    write_csv(os.path.join(out_dir, "docid_counts_comparison.csv"), comparison,
              ["Category", "DocId", "RawCount", "EtlCount", "Deficit"])

    summary = {
        "totals_raw": totals_raw,
        "totals_etl": totals_etl,
        "deficits": deficits,
    }
    write_json(os.path.join(out_dir, "docid_counts_summary.json"), summary)

    return {
        "raw_counts": raw_counts,
        "etl_counts": etl_counts,
        "comparison": comparison,
        "totals_raw": totals_raw,
        "totals_etl": totals_etl,
        "deficits": deficits,
        "outputs": {
            "raw_docid_counts": os.path.join(out_dir, "raw_docid_counts.csv"),
            "etl_docid_counts": os.path.join(out_dir, "etl_docid_counts.csv"),
            "docid_counts_comparison": os.path.join(out_dir, "docid_counts_comparison.csv"),
            "docid_counts_summary": os.path.join(out_dir, "docid_counts_summary.json"),
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose Raw vs ETL doc ID counts and emit CSV/JSON reports.")
    parser.add_argument(
        "--raw-dir",
        default=os.path.join("artifact", "raw_data"),
        help="Path to raw_data root (expects subfolders like PA, GI, Cashier).",
    )
    parser.add_argument(
        "--etl-dir",
        default=os.path.join("artifact", "etl_result_json"),
        help="Path to etl_result_json root (expects subfolders like pa_bulk_unique, gi_bulk_unique, cashier_bulk_unique).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory to write CSV and JSON summary outputs.",
    )
    args = parser.parse_args()

    result = run_diagnostic(args.raw_dir, args.etl_dir, out_dir=args.out_dir)

    print("DocID counts diagnostic complete.")
    print(f"Raw totals per category: {result['totals_raw']}")
    print(f"ETL totals per category: {result['totals_etl']}")
    print(f"Deficits (Raw > ETL): {len(result['deficits'])} rows")
    print("Outputs:")
    for k, v in result["outputs"].items():
        print(f"  {k}: {v}")