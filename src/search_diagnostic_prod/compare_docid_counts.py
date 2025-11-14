import argparse
import csv
import os
from collections import defaultdict


def normalize_category(name: str) -> str:
    """Normalize category labels for reporting.
    Raw top-level folders are expected to be 'PA', 'GI', 'Cashier'.
    ETL subfolders are expected to be 'pa_bulk_unique', 'gi_bulk_unique', 'cashier_bulk_unique'.
    """
    n = name.strip().lower()
    if n in {"attachment", "pa", "pa_bulk_unique"}:
        return "PA"
    if n in {"general insurance", "gi", "gi_bulk_unique", "ga", "general_insurance"}:
        return "GI"
    if n in {"cashier", "cashier_bulk_unique"}:
        return "Cashier"
    return name


def count_raw_by_docid(raw_root: str) -> dict:
    """Scan raw_data root structured as <raw_root>/<Category>/<DocId>/files and
    return mapping: (Category, DocId) -> file_count.
    Counts files recursively in each DocId directory.
    """
    counts = defaultdict(int)
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
            # Count files recursively under this docid folder
            file_count = 0
            for root, _dirs, files in os.walk(docid_path):
                file_count += len([f for f in files])
            counts[(category, docid_name)] += file_count
    return counts


def count_etl_by_docid(etl_root: str) -> dict:
    """Scan etl_result_json root structured as <etl_root>/<etl_category>/*.json and
    return mapping: (Category, DocId) -> json_count.
    DocId is the prefix before the first '_' in the JSON filename stem.
    """
    counts = defaultdict(int)
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


def write_csv(path: str, rows: list, headers: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Compare doc ID counts between raw_data and etl_result_json.")
    parser.add_argument('--raw_dir', required=False, default=r"c:\\Users\\muhammad.s.safian\\OneDrive - Avanade\\Project\\MAIN_PROJECT\\_avanade_main\\_avanade_main\\_01_prudential\\PRUHK-AgenticRAG-V1\\artifact\\raw_data",
                        help='Path to raw_data root (PA, GI, Cashier subfolders).')
    parser.add_argument('--etl_dir', required=False, default=r"c:\\Users\\muhammad.s.safian\\OneDrive - Avanade\\Project\\MAIN_PROJECT\\_avanade_main\\_avanade_main\\_01_prudential\\PRUHK-AgenticRAG-V1\\artifact\\etl_result_json",
                        help='Path to etl_result_json root (pa_bulk_unique, gi_bulk_unique, cashier_bulk_unique).')
    parser.add_argument('--out_dir', required=False, default='.', help='Directory to write CSV outputs.')

    args = parser.parse_args()

    raw_counts = count_raw_by_docid(args.raw_dir)
    etl_counts = count_etl_by_docid(args.etl_dir)

    # Emit per-source CSVs
    raw_rows = [
        {"Category": cat, "DocId": docid, "RawCount": raw_counts[(cat, docid)]}
        for (cat, docid) in sorted(raw_counts.keys())
    ]
    etl_rows = [
        {"Category": cat, "DocId": docid, "EtlCount": etl_counts[(cat, docid)]}
        for (cat, docid) in sorted(etl_counts.keys())
    ]

    write_csv(os.path.join(args.out_dir, 'raw_docid_counts.csv'), raw_rows, ['Category', 'DocId', 'RawCount'])
    write_csv(os.path.join(args.out_dir, 'etl_docid_counts.csv'), etl_rows, ['Category', 'DocId', 'EtlCount'])

    # Build comparison keyed by raw DocIds (focus on deficits where raw > ETL)
    comparison_rows = []
    for (cat, docid), rcount in raw_counts.items():
        ecount = etl_counts.get((cat, docid), 0)
        comparison_rows.append({
            'Category': cat,
            'DocId': docid,
            'RawCount': rcount,
            'EtlCount': ecount,
            'Deficit': rcount - ecount
        })

    # Sort by category then deficit desc then docid
    comparison_rows.sort(key=lambda x: (x['Category'], -x['Deficit'], x['DocId']))
    write_csv(os.path.join(args.out_dir, 'docid_counts_comparison.csv'), comparison_rows,
              ['Category', 'DocId', 'RawCount', 'EtlCount', 'Deficit'])

    # Print quick summary
    totals_raw = defaultdict(int)
    totals_etl = defaultdict(int)
    for (cat, _docid), rcount in raw_counts.items():
        totals_raw[cat] += rcount
    for (cat, _docid), ecount in etl_counts.items():
        totals_etl[cat] += ecount

    print('Raw totals by category:')
    for cat in sorted(totals_raw.keys()):
        print(f"  {cat}: {totals_raw[cat]}")
    print('ETL totals by category:')
    for cat in sorted(totals_etl.keys()):
        print(f"  {cat}: {totals_etl[cat]}")

    deficits = [r for r in comparison_rows if r['Deficit'] > 0]
    print(f"DocIds with deficits: {len(deficits)}")
    for r in deficits[:20]:
        print(f"  {r['Category']} {r['DocId']}: raw={r['RawCount']} etl={r['EtlCount']} deficit={r['Deficit']}")


if __name__ == '__main__':
    main()