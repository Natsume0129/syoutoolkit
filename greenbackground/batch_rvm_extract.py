import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# =========================
# CONFIG (edit here)
# =========================
INPUT_DIR = r"E:\Matsuda_data\手动标注\10-29"
OUTPUT_DIR = r"E:\Matsuda_data\手动标注\10-29_output"

MODEL_NAME = "mobilenetv3"   # "mobilenetv3" (fast) or "resnet50" (heavier)
DOWNSAMPLE_RATIO = 0.25      # lower -> faster, usually stable enough for your use
DEVICE = "cuda"              # "cuda" or "cpu"

# Output mode:
#   "alpha"       -> save alpha mask only (recommended for expression research)
#   "green"       -> composite on green background (preview)
#   "rgba_cutout" -> transparent PNG cutout (RGBA)
OUTPUT_MODE = "rgba_cutout"

# Frame naming
FRAME_DIGITS = 6             # 000000.png ...
START_INDEX = 0

# Optional: process every N frames (1 = all frames)
FRAME_STRIDE = 1

# Optional: limit frames for quick test (None = full)
MAX_FRAMES = None

# Green color (BGR for OpenCV) used when OUTPUT_MODE="green"
GREEN_BGR = (0, 255, 0)
# =========================
# END CONFIG
# =========================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def die(msg: str) -> None:
    raise RuntimeError(msg)


def load_rvm(model_name: str):
    repo = "PeterL1n/RobustVideoMatting"
    # trust_repo=True avoids future interactive trust prompts
    model = torch.hub.load(repo, model_name, pretrained=True, trust_repo=True).eval()
    return model


def to_uint8_alpha(pha: torch.Tensor) -> np.ndarray:
    # pha: (1,1,H,W) float in ~[0,1]
    return pha.squeeze(0).squeeze(0).clamp(0, 1).mul(255).byte().cpu().numpy()


def imwrite_unicode(path, img: np.ndarray) -> bool:
    """
    Unicode-safe image write on Windows.
    Works around cv2.imwrite sometimes failing on non-ASCII paths.
    """
    path = Path(path)
    ensure_dir(path.parent)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    ensure_dir(out_dir)

    if not in_dir.is_dir():
        die(f"INPUT_DIR not found: {in_dir}")

    videos = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
    if not videos:
        print(f"[WARN] No mp4 found in: {in_dir}")
        return

    if DEVICE == "cuda" and not torch.cuda.is_available():
        die("DEVICE is 'cuda' but torch.cuda.is_available() is False.")

    model = load_rvm(MODEL_NAME).to(DEVICE)
    torch.backends.cudnn.benchmark = True

    print(f"[INFO] Found {len(videos)} videos")
    print(f"[INFO] OUTPUT_MODE={OUTPUT_MODE}, MODEL={MODEL_NAME}, DOWNSAMPLE_RATIO={DOWNSAMPLE_RATIO}, DEVICE={DEVICE}")
    print(f"[INFO] Output root: {out_dir}")

    for vp in videos:
        stem = vp.stem
        video_out = out_dir / stem
        ensure_dir(video_out)

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[FAIL] Cannot open video: {vp}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] {stem}: total={total}, fps={fps}, size=({w},{h})")

        meta_path = video_out / "meta.txt"
        meta_path.write_text(
            f"video={vp}\n"
            f"fps={fps}\n"
            f"frame_count={total}\n"
            f"width={w}\n"
            f"height={h}\n"
            f"output_mode={OUTPUT_MODE}\n"
            f"downsample_ratio={DOWNSAMPLE_RATIO}\n"
            f"frame_stride={FRAME_STRIDE}\n"
            f"start_index={START_INDEX}\n",
            encoding="utf-8",
        )

        rec = [None] * 4  # recurrent states for stability (critical)
        frame_idx = 0
        saved_idx = START_INDEX

        # If total is unknown (0), fall back to a very large loop and break on ret=False
        loop_iter = total if total > 0 else 10**12
        pbar_total = total if total > 0 else None

        t0 = time.time()
        with torch.no_grad():
            for _ in tqdm(range(loop_iter), desc=stem, total=pbar_total):
                ret, bgr = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_STRIDE != 0:
                    frame_idx += 1
                    continue

                if MAX_FRAMES is not None and (saved_idx - START_INDEX) >= MAX_FRAMES:
                    break

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                src = (
                    torch.from_numpy(rgb)
                    .to(DEVICE)
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )

                fgr, pha, *rec = model(src, *rec, downsample_ratio=DOWNSAMPLE_RATIO)
                alpha_u8 = to_uint8_alpha(pha)

                name = f"{saved_idx:0{FRAME_DIGITS}d}.png"
                out_path = video_out / name

                if OUTPUT_MODE == "alpha":
                    ok_write = imwrite_unicode(out_path, alpha_u8)

                elif OUTPUT_MODE == "green":
                    # Composite on green background (preview)
                    a = (alpha_u8.astype(np.float32) / 255.0)[..., None]  # (H,W,1)
                    fg = rgb.astype(np.float32) * a
                    bg = np.zeros_like(fg)
                    bg[..., 0] = GREEN_BGR[2]  # R
                    bg[..., 1] = GREEN_BGR[1]  # G
                    bg[..., 2] = GREEN_BGR[0]  # B
                    comp = fg + bg * (1.0 - a)
                    comp = comp.clip(0, 255).astype(np.uint8)
                    comp_bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
                    ok_write = imwrite_unicode(out_path, comp_bgr)

                elif OUTPUT_MODE == "rgba_cutout":
                    # Transparent cutout (RGBA)
                    a = alpha_u8[..., None]  # (H,W,1) uint8
                    cut_rgb = (rgb.astype(np.float32) * (a.astype(np.float32) / 255.0)).astype(np.uint8)
                    rgba = np.concatenate([cut_rgb, a], axis=2)  # (H,W,4) RGBA
                    bgra = rgba[..., [2, 1, 0, 3]]  # to BGRA for OpenCV encoding
                    ok_write = imwrite_unicode(out_path, bgra)

                else:
                    cap.release()
                    die(f"Unknown OUTPUT_MODE: {OUTPUT_MODE}")

                if not ok_write:
                    cap.release()
                    die(f"Write failed: {out_path}")

                saved_idx += 1
                frame_idx += 1

        cap.release()
        dt = time.time() - t0
        saved = saved_idx - START_INDEX
        print(f"[OK] {stem}: saved {saved} frames in {dt:.1f}s ({(saved/dt if dt>0 else 0):.2f} fps) -> {video_out}")

    print("[DONE]")


if __name__ == "__main__":
    main()
