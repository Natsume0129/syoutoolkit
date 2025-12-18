import os
import math
import subprocess
from shutil import which

# =========================
# 配置区（只改这里）
# =========================
INPUT_FILE = r"E:\temp\12-15.mp3"   # 输入 mp3
N_PARTS = 3                        # 分成 N 份
OVERLAP_SEC = 10                   # 向后冗余秒数（除最后一段）
OUTPUT_DIR = None                  # None 表示输出到同目录；也可填 r"E:\temp\out"
BITRATE = "192k"                   # 重新编码输出的 mp3 码率
# =========================


def require_tool(name: str):
    if which(name) is None:
        raise FileNotFoundError(f"{name} not found in PATH. Please install and add to PATH: {name}")


def get_duration_sec(path: str) -> float:
    # 用 ffprobe 获取时长（秒）
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def format_idx(i: int, n: int) -> str:
    width = max(2, len(str(n)))
    return str(i).zfill(width)


def run_ffmpeg_cut(input_file: str, start: float, dur: float, out_file: str):
    # 为了分割边界准确，采用重新编码（最稳）
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{dur:.3f}",
        "-i", input_file,
        "-c:a", "libmp3lame",
        "-b:a", BITRATE,
        out_file
    ]
    subprocess.run(cmd, check=True)


def main():
    require_tool("ffmpeg")
    require_tool("ffprobe")

    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    if N_PARTS <= 0:
        raise ValueError("N_PARTS must be > 0")
    if OVERLAP_SEC < 0:
        raise ValueError("OVERLAP_SEC must be >= 0")

    total = get_duration_sec(INPUT_FILE)
    base = total / N_PARTS

    in_dir = os.path.dirname(INPUT_FILE)
    out_dir = OUTPUT_DIR or in_dir
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]

    for i in range(N_PARTS):
        start = i * base
        if i == N_PARTS - 1:
            end = total
        else:
            end = min((i + 1) * base + OVERLAP_SEC, total)

        dur = max(0.0, end - start)

        idx = format_idx(i + 1, N_PARTS)
        out_file = os.path.join(out_dir, f"{base_name}_part{idx}.mp3")

        run_ffmpeg_cut(INPUT_FILE, start, dur, out_file)
        print(f"[{i+1}/{N_PARTS}] {start:.2f}s -> {end:.2f}s  =>  {out_file}")

    print("Done.")


if __name__ == "__main__":
    main()
