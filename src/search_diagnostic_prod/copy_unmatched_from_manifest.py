import argparse
import json
import os
import shutil


def copy_unmatched(manifest_path: str, dest_root: str, clean: bool = False) -> int:
    if clean and os.path.isdir(dest_root):
        # Remove existing contents safely
        for entry in os.listdir(dest_root):
            entry_path = os.path.join(dest_root, entry)
            try:
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)
            except Exception:
                pass

    # Ensure destination exists
    os.makedirs(dest_root, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        items = json.load(f)

    copied = 0
    for it in items:
        doc_id = it.get("doc_id")
        category = it.get("category")
        filename = it.get("filename")
        src_path = it.get("abs_path")
        if not doc_id or not category or not filename or not src_path:
            continue
        if not os.path.isfile(src_path):
            continue
        dest_dir = os.path.join(dest_root, category, doc_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(src_path, dest_path)
            copied += 1
        except Exception:
            # Skip problematic files silently
            continue

    return copied


def main():
    parser = argparse.ArgumentParser(description="Copy unmatched files (from manifest) into missing_file directory")
    parser.add_argument("--manifest", default=r"outputs\\unmatched_files.json", help="Path to unmatched files JSON")
    parser.add_argument("--dest", default=r"artifact\\missing_file", help="Destination root directory")
    parser.add_argument("--clean", action="store_true", help="Clean destination before copying")
    args = parser.parse_args()

    total = copy_unmatched(args.manifest, args.dest, args.clean)
    print(f"Copied {total} files into: {args.dest}")


if __name__ == "__main__":
    main()