import os
import sys
import json
from pathlib import Path

"""
Batch-run ETL pipeline over files in artifact/missing_file and output JSONs
to artifact/etl_result_json/missedout without indexing.

Usage (from repo root):
  python -u src\search_diagnostic_prod\run_etl_on_missing.py \
    --input-root "artifact\\missing_file" \
    --output-root "artifact\\etl_result_json\\missedout"

This script uses src.etl.main.process_file to generate ETL JSON for each file.
It preserves category subfolders and names output files as:
  <DocId>_<original_stem>.json
"""

import argparse


def discover_files(input_root: Path):
    files: list[Path] = []
    for p in input_root.rglob("*"):
        if p.is_file():
            files.append(p)
    return files


def build_output_path(output_root: Path, file_path: Path) -> Path:
    # Expect structure: .../missing_file/<category>/<DocId>/<filename>
    parts = file_path.parts
    category = None
    doc_id = None
    if len(parts) >= 3:
        # Find 'missing_file' and read next two parts
        try:
            idx = parts.index("missing_file")
            category = parts[idx + 1] if len(parts) > idx + 1 else None
            doc_id = parts[idx + 2] if len(parts) > idx + 2 else None
        except ValueError:
            # Fallback: assume last two folders are category/doc_id
            category = parts[-3] if len(parts) >= 3 else None
            doc_id = parts[-2] if len(parts) >= 2 else None

    # Safe defaults if not found
    category = category or "misc"
    doc_id = doc_id or "UNKNOWN"

    stem = file_path.stem
    out_dir = output_root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{doc_id}_{stem}.json"
    return out_dir / out_name


def main():
    parser = argparse.ArgumentParser(description="Run ETL over missing_file, output JSON only")
    parser.add_argument("--input-root", type=str, default=r"artifact\\missing_file")
    parser.add_argument("--output-root", type=str, default=r"artifact\\etl_result_json\\missedout")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        print(f"Input root not found: {input_root}")
        sys.exit(1)

    files = discover_files(input_root)
    if not files:
        print(f"No files discovered under {input_root}")
        sys.exit(0)

    total = 0
    success = 0
    failures: list[dict] = []

    for fp in files:
        total += 1
        out_path = build_output_path(output_root, fp)
        # Skip if already extracted
        if out_path.exists():
            continue
        try:
            # Invoke ETL CLI per file to avoid import path issues
            import subprocess
            cmd = [
                sys.executable,
                "-m",
                "src.etl.main",
                "process",
                "--file",
                str(fp.resolve()),
                "--local",
                "--no-index",
                "--output-json",
                str(out_path),
            ]
            result = subprocess.run(cmd, cwd=str(Path.cwd()))
            if result.returncode == 0:
                success += 1
            else:
                failures.append({"file": str(fp), "error": f"Return code {result.returncode}"})
        except Exception as e:
            failures.append({"file": str(fp), "error": str(e)})

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "total_files": total,
        "success": success,
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()