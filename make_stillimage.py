# make_stillimage.py
# Usage:
#   python make_stillimage.py --input "E:\path\to\frames" --step 10 --out "still.png"
# Optional:
#   --pattern "*.png" --max_count 120 --tile_w 320 --gap 12 --pad 24 --font_size 20

import argparse
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def natural_key(p: Path):
    # simple numeric sort by stem if possible, fallback to name
    try:
        return int("".join([c for c in p.stem if c.isdigit()]) or -1), p.name
    except Exception:
        return (10**18, p.name)


def pick_grid(n: int, target_aspect: float = 16 / 9):
    """
    Choose (cols, rows) so that (cols / rows) ~ target_aspect and cols*rows >= n.
    """
    if n <= 0:
        return 1, 1
    cols = max(1, round(math.sqrt(n * target_aspect)))
    rows = math.ceil(n / cols)
    # try local adjustments to get closer to aspect while still fitting
    best = (cols, rows)
    best_err = abs((cols / rows) - target_aspect)
    for c in range(max(1, cols - 8), cols + 9):
        r = math.ceil(n / c)
        err = abs((c / r) - target_aspect)
        if c * r >= n and err < best_err:
            best, best_err = (c, r), err
    return best


def load_font(font_size: int):
    # Try common fonts; fallback to default.
    candidates = [
        "arial.ttf",                # Windows
        "msyh.ttc",                 # Microsoft YaHei (often available)
        "msgothic.ttc",             # MS Gothic
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for f in candidates:
        try:
            return ImageFont.truetype(f, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder containing sequential frames")
    ap.add_argument("--out", required=True, help="Output image path (e.g., still.png)")
    ap.add_argument("--step", type=int, default=10, help="Pick one frame every N frames")
    ap.add_argument("--pattern", default="*.png", help="Glob pattern, e.g., '*.png' or '*.jpg'")
    ap.add_argument("--max_count", type=int, default=0, help="Max selected frames (0 = no limit)")
    ap.add_argument("--tile_w", type=int, default=320, help="Tile width in pixels")
    ap.add_argument("--gap", type=int, default=12, help="Gap between tiles")
    ap.add_argument("--pad", type=int, default=24, help="Outer padding")
    ap.add_argument("--font_size", type=int, default=20, help="Index label font size")
    ap.add_argument("--label_h", type=int, default=34, help="Reserved height below each tile for label")
    ap.add_argument("--bg", default="white", help="Background color (white/black/#RRGGBB)")
    ap.add_argument("--text", default="black", help="Text color (black/white/#RRGGBB)")
    ap.add_argument("--target_aspect", type=float, default=16/9, help="Target overall aspect ratio")
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    files = sorted(in_dir.glob(args.pattern), key=natural_key)
    if not files:
        raise RuntimeError(f"No files matched pattern {args.pattern} in {in_dir}")

    step = max(1, args.step)
    picked = files[::step]
    if args.max_count and args.max_count > 0:
        picked = picked[: args.max_count]
    n = len(picked)
    if n == 0:
        raise RuntimeError("No frames selected. Check --step / --pattern / --max_count")

    # Load first image to infer aspect for tile height
    first = Image.open(picked[0]).convert("RGB")
    src_w, src_h = first.size
    tile_w = max(64, args.tile_w)
    tile_h = max(64, round(tile_w * (src_h / src_w)))  # keep original aspect

    cols, rows = pick_grid(n, args.target_aspect)

    pad = max(0, args.pad)
    gap = max(0, args.gap)
    label_h = max(0, args.label_h)

    canvas_w = pad * 2 + cols * tile_w + (cols - 1) * gap
    canvas_h = pad * 2 + rows * (tile_h + label_h) + (rows - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=args.bg)
    draw = ImageDraw.Draw(canvas)
    font = load_font(args.font_size)

    for i, fp in enumerate(picked, start=1):
        r = (i - 1) // cols
        c = (i - 1) % cols
        x0 = pad + c * (tile_w + gap)
        y0 = pad + r * ((tile_h + label_h) + gap)

        img = Image.open(fp).convert("RGB")
        # Fit into tile (center crop to preserve look, optional; here: contain with padding)
        img = img.resize((tile_w, tile_h), Image.BICUBIC)

        canvas.paste(img, (x0, y0))

        label = f"{i}"
        # center text under the tile
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x0 + (tile_w - tw) // 2
        ty = y0 + tile_h + (label_h - th) // 2
        draw.text((tx, ty), label, font=font, fill=args.text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved: {out_path}  ({canvas_w}x{canvas_h}), grid={cols}x{rows}, tiles={n}, step={step}")


if __name__ == "__main__":
    main()
