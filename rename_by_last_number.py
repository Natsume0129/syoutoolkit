#!/usr/bin/env python3
# rename_by_last_number.py
# Rename files like: 20251029_15-44-15-51_0_6_176.png -> 176.png
# Keeps extension. Skips files that don't match. Avoids overwriting by auto-suffix.
'''
python rename_by_last_number.py "E:\your_folder" --dry-run
python rename_by_last_number.py "E:\your_folder"
'''





from pathlib import Path
import re
import argparse


# match "..._<num>.<ext>" (ext can be png/jpg/jpeg/webp/bmp/tif/tiff, case-insensitive)
PATTERN = re.compile(
    r"""^(?P<prefix>.+)_(?P<idx>\d+)\.(?P<ext>png|jpg|jpeg|webp|bmp|tif|tiff)$""",
    re.IGNORECASE,
)


def unique_target(dir_path: Path, base: str, ext: str) -> Path:
    """If target exists, create base_1, base_2, ..."""
    target = dir_path / f"{base}.{ext}"
    if not target.exists():
        return target
    k = 1
    while True:
        cand = dir_path / f"{base}_{k}.{ext}"
        if not cand.exists():
            return cand
        k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder containing images to rename")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without renaming")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    files = [p for p in folder.iterdir() if p.is_file()]
    files.sort()

    renamed = 0
    skipped = 0

    for p in files:
        m = PATTERN.match(p.name)
        if not m:
            skipped += 1
            continue

        idx = m.group("idx")          # "176"
        ext = p.suffix.lstrip(".")    # keep original case? use lower for consistency
        ext = ext.lower()

        dst = unique_target(folder, idx, ext)

        if dst.name == p.name:
            skipped += 1
            continue

        if args.dry_run:
            print(f"[DRY] {p.name} -> {dst.name}")
        else:
            p.rename(dst)
            print(f"{p.name} -> {dst.name}")
        renamed += 1

    print(f"\nDone. Renamed: {renamed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
