# rvm_preflight_test.py
# Purpose: rigorous preflight test for RobustVideoMatting inference on Windows + NVIDIA GPU
# What it checks:
# 1) Python/PyTorch/CUDA availability, GPU name & capability
# 2) torch.hub download/load for RVM + converter
# 3) single-frame forward pass (verifies kernels, dtype/device, recurrent states)
# 4) short synthetic "video" recurrence stability sanity (no NaN/Inf, shapes consistent)
# 5) optional real video IO check (requires --video and PyAV installed)

import argparse
import os
import platform
import sys
import time
from typing import Tuple

import numpy as np

def die(msg: str, code: int = 1) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(code)

def ok(msg: str) -> None:
    print(f"[OK]   {msg}")

def info(msg: str) -> None:
    print(f"[INFO] {msg}")

def torch_env_check() -> Tuple["torch.device", "torch.dtype"]:
    try:
        import torch
    except Exception as e:
        die(f"PyTorch import failed: {e}")

    info(f"Python: {sys.version.split()[0]} | OS: {platform.system()} {platform.release()} | Arch: {platform.machine()}")
    info(f"torch: {torch.__version__} | torch.version.cuda: {torch.version.cuda} | cudnn: {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        die("torch.cuda.is_available() is False. CUDA not available to PyTorch.")

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    cap = torch.cuda.get_device_capability(dev)
    info(f"GPU[0]: {name} | compute capability: {cap}")

    # Basic CUDA op sanity
    try:
        x = torch.randn(1024, 1024, device=dev)
        y = x @ x
        _ = y.mean().item()
    except Exception as e:
        die(f"Basic CUDA matmul failed: {e}")

    ok("CUDA is usable in PyTorch.")

    # Prefer float16 for speed, but keep float32 for strict safety on first test.
    dtype = torch.float32
    return dev, dtype

def load_rvm():
    import torch
    repo = "PeterL1n/RobustVideoMatting"

    info("Loading RVM model via torch.hub (this may download weights on first run)...")
    t0 = time.time()
    try:
        model = torch.hub.load(repo, "mobilenetv3", pretrained=True)
    except Exception as e:
        die(f"torch.hub.load(model) failed: {e}")
    ok(f"Loaded model: {type(model)} in {time.time()-t0:.2f}s")

    info("Loading converter via torch.hub...")
    t0 = time.time()
    try:
        converter = torch.hub.load(repo, "converter")
    except Exception as e:
        die(f"torch.hub.load(converter) failed: {e}")
    ok(f"Loaded converter: {type(converter)} in {time.time()-t0:.2f}s")

    return model, converter

def single_frame_forward(model, device, dtype) -> None:
    import torch

    model = model.to(device=device, dtype=dtype).eval()

    # Synthetic RGB frame (normalized to [0,1]) with size divisible by 32 (safe for many backbones)
    H, W = 512, 512
    src = torch.rand(1, 3, H, W, device=device, dtype=dtype)

    rec = [None] * 4

    info("Running single-frame forward pass...")
    with torch.no_grad():
        try:
            fgr, pha, *rec = model(src, *rec, downsample_ratio=0.25)
        except Exception as e:
            die(f"Model forward failed: {e}")

    # Shape checks
    if fgr.shape != (1, 3, H, W):
        die(f"Unexpected fgr shape: {tuple(fgr.shape)} expected (1,3,{H},{W})")
    if pha.shape != (1, 1, H, W):
        die(f"Unexpected pha shape: {tuple(pha.shape)} expected (1,1,{H},{W})")

    # Value checks
    if not torch.isfinite(fgr).all().item():
        die("fgr contains NaN/Inf")
    if not torch.isfinite(pha).all().item():
        die("pha contains NaN/Inf")

    # Alpha range sanity (may exceed slightly before clamp depending on impl, so check loosely)
    pha_min = pha.min().item()
    pha_max = pha.max().item()
    info(f"pha range: min={pha_min:.6f}, max={pha_max:.6f}")
    if pha_min < -0.2 or pha_max > 1.2:
        die("pha range looks abnormal (expected roughly within [0,1])")

    ok("Single-frame forward pass OK (shapes + finiteness + alpha range sanity).")

def recurrence_sanity(model, device, dtype, n_frames: int = 24) -> None:
    import torch
    model = model.to(device=device, dtype=dtype).eval()

    H, W = 512, 512
    rec = [None] * 4

    info(f"Running recurrence sanity for {n_frames} synthetic frames...")
    pha_stats = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_frames):
            # small temporal variation: base + moving square
            frame = torch.rand(1, 3, H, W, device=device, dtype=dtype) * 0.1
            x0 = (i * 7) % (W - 80)
            y0 = (i * 5) % (H - 80)
            frame[:, :, y0:y0+80, x0:x0+80] += 0.9

            fgr, pha, *rec = model(frame, *rec, downsample_ratio=0.25)

            if not torch.isfinite(pha).all().item():
                die(f"pha contains NaN/Inf at frame {i}")

            pha_stats.append((pha.mean().item(), pha.std().item()))

    dt = time.time() - t0
    fps = n_frames / dt
    info(f"Recurrence sanity done. Throughput (synthetic): {fps:.2f} FPS")

    # Check that stats are not degenerate (all same / zero std)
    means = np.array([m for m, s in pha_stats])
    stds = np.array([s for m, s in pha_stats])

    info(f"pha mean: {means.min():.4f} .. {means.max():.4f} | pha std: {stds.min():.4f} .. {stds.max():.4f}")
    if np.allclose(stds, 0.0):
        die("pha std is ~0 for all frames; output seems degenerate.")
    ok("Recurrence sanity OK (no NaN/Inf, non-degenerate outputs).")

def optional_video_io_test(converter, model, video_path: str, out_dir: str) -> None:
    # This checks converter.convert_video on a real file.
    # Requires PyAV: pip install av
    import torch

    if not os.path.isfile(video_path):
        die(f"--video path not found: {video_path}")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_comp = os.path.join(out_dir, f"{base}__composition_png")
    out_alpha = os.path.join(out_dir, f"{base}__alpha_png")

    info("Running converter.convert_video on a short real-video test.")
    info("If this is a long video, please cut a 3-10s clip first for preflight.")

    # Green screen background (RGB)
    bgr = (0.0, 1.0, 0.0)

    try:
        converter(
            model,
            input_source=video_path,
            output_type="png_sequence",
            output_composition=out_comp,
            output_alpha=out_alpha,
            downsample_ratio=0.25,
            seq_chunk=12,
            num_workers=0,
            progress=True,
        )

    except Exception as e:
        die(f"convert_video failed: {e}")

    if not os.path.isfile(out_comp):
        die(f"Expected output not found: {out_comp}")
    if not os.path.isfile(out_alpha):
        die(f"Expected output not found: {out_alpha}")

    ok(f"convert_video OK. Outputs:\n  - {out_comp}\n  - {out_alpha}")

def main():
    parser = argparse.ArgumentParser(description="RVM preflight test (GPU + TorchHub + forward + recurrence + optional video IO)")
    parser.add_argument("--video", type=str, default="", help="Optional: path to a short video clip for convert_video test")
    parser.add_argument("--out", type=str, default="rvm_preflight_out", help="Output dir for optional video test")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Test dtype (fp32 recommended first)")
    args = parser.parse_args()

    device, _ = torch_env_check()

    import torch
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    model, converter = load_rvm()

    # Forward tests
    single_frame_forward(model, device, dtype)
    recurrence_sanity(model, device, dtype)

    # Optional real-video IO
    if args.video:
        # Ensure PyAV is importable before running
        try:
            import av  # noqa: F401
        except Exception as e:
            die(f"PyAV not available (pip install av). Import error: {e}")
        optional_video_io_test(converter, model.to(device=device, dtype=dtype).eval(), args.video, args.out)
    else:
        info("Skipped real-video IO test (no --video provided).")

    ok("All preflight checks passed.")

if __name__ == "__main__":
    main()
