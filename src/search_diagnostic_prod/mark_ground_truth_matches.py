import argparse
import csv
import json
import os
import re
import sys
import unicodedata


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    # Normalize Unicode (handles TC/EN and symbols), casefold for robust matching
    text = unicodedata.normalize("NFKC", value).casefold()
    # Strip file extension if present
    text = re.sub(r"\.(pdf|docx?|xlsx?|pptx?)$", "", text)
    # Preserve file names more strictly for exact equality by only trimming outer spaces
    text = text.strip()
    return text


def load_index_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = data.get("grouped") or []
    by_doc = {}
    filename_to_docids = {}

    for entry in grouped:
        doc_id = entry.get("document_id")
        filenames = entry.get("filenames") or []
        count = entry.get("count") or 0

        by_doc[doc_id] = {
            "filenames": filenames,
            "normalized_filenames": [normalize_text(fn) for fn in filenames],
            "count": count,
        }

        for fn in filenames:
            nfn = normalize_text(fn)
            filename_to_docids.setdefault(nfn, set()).add(doc_id)

    return by_doc, filename_to_docids


def detect_columns(header):
    # Try to find required columns robustly
    # Expected: doc_id, title
    header_lower = [h.casefold() for h in header]
    col_map = {}

    for target in ["doc_id", "docid", "document_id"]:
        if target in header_lower:
            col_map["doc_id"] = header_lower.index(target)
            break

    for target in ["title", "document_title", "doc_title"]:
        if target in header_lower:
            col_map["title"] = header_lower.index(target)
            break

    return col_map


def main():
    parser = argparse.ArgumentParser(description="Mark ground truth CSV rows if the title matches any filename in the index JSON (ignores doc_id).")
    parser.add_argument("--csv", required=True, help="Path to ground_truth.csv")
    parser.add_argument("--json", required=True, help="Path to docs_pages_experiment_full_2a.json")
    parser.add_argument("--output", required=True, help="Path to output marked CSV")
    args = parser.parse_args()

    by_doc, filename_to_docids = load_index_json(args.json)

    input_rows = []
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_map = detect_columns(header)

        if "title" not in col_map:
            print("ERROR: CSV must contain a 'title' column.")
            print("Detected header:", header)
            sys.exit(1)

        for row in reader:
            input_rows.append(row)

    # Prepare output header: original columns + our markers
    out_header = header + [
        "match_status",
        "matched_doc_ids",
        "matched_filename_sample",
    ]

    matched_exact = 0

    out_rows = []
    for row in input_rows:
        title = row[col_map["title"]].strip()
        title = row[col_map["title"]].strip()

        n_title = normalize_text(title)
        doc_ids = sorted(list(filename_to_docids.get(n_title, set())))
        title_match = len(doc_ids) > 0
        matched_filename_sample = ""
        if title_match:
            # Pick one filename sample from any of the matched doc_ids
            for doc_id in doc_ids:
                fns = by_doc.get(doc_id, {}).get("filenames", [])
                for fn in fns:
                    if normalize_text(fn) == n_title:
                        matched_filename_sample = fn
                        break
                if matched_filename_sample:
                    break

        if title_match:
            matched_exact += 1

        # Append markers
        out_rows.append(
            row
            + [
                "Yes" if title_match else "No",
                ";".join(doc_ids),
                matched_filename_sample,
            ]
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(out_header)
        writer.writerows(out_rows)

    total = len(input_rows)
    print("Total rows:", total)
    print("Filename matches (title == any filename):", matched_exact)
    print("Output written to:", args.output)


if __name__ == "__main__":
    main()