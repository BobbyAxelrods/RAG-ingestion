import argparse
import json
import os


def build_manifest(root: str):
    items = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            abs_path = os.path.join(dirpath, fn)
            rel_path = os.path.relpath(abs_path, root)
            parts = rel_path.split(os.sep)
            category = parts[0] if len(parts) > 0 else ""
            doc_id = parts[1] if len(parts) > 1 else ""
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


def main():
    parser = argparse.ArgumentParser(description="Export JSON list of filenames in missing_file")
    parser.add_argument(
        "--root",
        default=r"artifact\\missing_file",
        help="Root directory of copied missing files",
    )
    parser.add_argument(
        "--out",
        default=r"outputs\\missing_manifest.json",
        help="Output JSON manifest path",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(args.out)


if __name__ == "__main__":
    main()