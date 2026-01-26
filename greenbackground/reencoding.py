"""
Batch: each subfolder = one clip (frames), re-encode to MP4.
Unicode-safe on Windows (Chinese paths OK).

Dependencies: opencv-python, numpy, tqdm
"""

import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# =========================
# CONFIG (edit here)
# =========================
INPUT_DIR = r"E:\Matsuda_data\手动标注( vgg-face分析)\10-29_output"
OUTPUT_DIR = r"E:\Matsuda_data\手动标注( vgg-face分析)\10-29_output_mp4"

USE_META_FPS = True
DEFAULT_FPS = 30.0

FOURCC = "mp4v"       # if fails, try "avc1"
OUT_SUFFIX = ".mp4"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# alpha handling for BGRA PNGs
ALPHA_MODE = "bg_color"   # "keep_black" / "bg_color" / "bg_image"
BG_COLOR_BGR = (0, 255, 0)
BG_IMAGE_PATH = None

STRIDE = 1
MAX_FRAMES = None
# =========================
# END CONFIG
# =========================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def die(msg: str) -> None:
    raise RuntimeError(msg)


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_images(folder: Path):
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(imgs, key=natural_key)


def imread_unicode(path: Path, flags=cv2.IMREAD_UNCHANGED):
    """
    Unicode-safe image read on Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def parse_fps_from_meta(meta_path: Path):
    if not meta_path.exists():
        return None
    try:
        txt = meta_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(r"^\s*fps\s*=\s*([0-9]*\.?[0-9]+)\s*$", txt, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def prepare_bg(h: int, w: int):
    if ALPHA_MODE == "bg_color":
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = np.array(BG_COLOR_BGR, dtype=np.uint8)
        return bg

    if ALPHA_MODE == "bg_image":
        if not BG_IMAGE_PATH:
            die("BG_IMAGE_PATH must be set when ALPHA_MODE='bg_image'")
        bg = imread_unicode(Path(BG_IMAGE_PATH), flags=cv2.IMREAD_COLOR)
        if bg is None:
            die(f"Cannot read BG_IMAGE_PATH: {BG_IMAGE_PATH}")
        if bg.shape[0] != h or bg.shape[1] != w:
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
        return bg

    return None


def composite_bgra_to_bgr(img_bgra: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    a = img_bgra[..., 3:4].astype(np.float32) / 255.0  # HxWx1
    fg = img_bgra[..., 0:3].astype(np.float32)         # BGR
    bg = bg_bgr.astype(np.float32)
    out = fg * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def encode_clip_folder(clip_dir: Path, out_path: Path):
    images = list_images(clip_dir)
    if not images:
        print(f"[WARN] No frames: {clip_dir}")
        return

    if STRIDE > 1:
        images = images[::STRIDE]
    if MAX_FRAMES is not None:
        images = images[:MAX_FRAMES]
    if not images:
        print(f"[WARN] No frames after STRIDE/MAX_FRAMES: {clip_dir}")
        return

    first = imread_unicode(images[0], flags=cv2.IMREAD_UNCHANGED)
    if first is None:
        print(f"[FAIL] Cannot read first frame: {images[0]}")
        return

    if first.ndim != 3 or first.shape[2] not in (3, 4):
        die(f"Unsupported image shape: {first.shape} ({images[0]})")

    h, w = first.shape[:2]

    fps = None
    if USE_META_FPS:
        fps = parse_fps_from_meta(clip_dir / "meta.txt")
    if fps is None:
        fps = DEFAULT_FPS

    ensure_dir(out_path.parent)
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not vw.isOpened():
        die(f"VideoWriter open failed: {out_path} (try FOURCC='avc1')")

    bg = prepare_bg(h, w)

    wrote = 0
    for ip in tqdm(images, desc=f"{clip_dir.name}", total=len(images)):
        img = imread_unicode(ip, flags=cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Skip unreadable: {ip}")
            continue

        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        if img.shape[2] == 4:
            if ALPHA_MODE == "keep_black":
                frame = img[..., 0:3]
            elif ALPHA_MODE in ("bg_color", "bg_image"):
                frame = composite_bgra_to_bgr(img, bg)
            else:
                die(f"Unknown ALPHA_MODE: {ALPHA_MODE}")
        else:
            frame = img

        vw.write(frame)
        wrote += 1

    vw.release()
    print(f"[OK] {clip_dir.name} -> {out_path} | frames={wrote} | fps={fps}")


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    ensure_dir(out_dir)

    if not in_dir.is_dir():
        die(f"INPUT_DIR not found: {in_dir}")

    clip_dirs = sorted([p for p in in_dir.iterdir() if p.is_dir()], key=natural_key)
    if not clip_dirs:
        die(f"No clip folders found in: {in_dir}")

    print(f"[INFO] clip folders: {len(clip_dirs)}")
    print(f"[INFO] output: {out_dir}")

    for cd in clip_dirs:
        out_path = out_dir / f"{cd.name}{OUT_SUFFIX}"
        encode_clip_folder(cd, out_path)

    print("[DONE]")


if __name__ == "__main__":
    main()
