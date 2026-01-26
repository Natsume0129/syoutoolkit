# video_clip_from_dat.py
# 依赖：本机已安装 ffmpeg，并且 ffmpeg 在 PATH 中可直接调用
# 用法：python video_clip_from_dat.py

import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple


# =========================
# CONFIG（只改这里）
# =========================
@dataclass
class Config:
    input_video: str = r"E:\Matsuda_data\20251029\20251029.mp4"

    # dat/csv 文件：第一行是表头（start_time,end_time）
    intervals_dat: str = r"E:\Matsuda_data\手动标注\10-29.dat"

    output_dir: str = r"E:\Matsuda_data\手动标注\10-29"

    # 输出在区间前后各扩展多少秒（防止边界帧缺失）
    pad_seconds: float = 1.0

    # True: 尽量无损快速裁剪（关键帧限制，仍可能有边界误差）
    # False: 重新编码，更精确（更慢）
    stream_copy: bool = True

    # 仅在 stream_copy=False 时生效
    vcodec: str = "libx264"
    acodec: str = "aac"
    crf: int = 18
    preset: str = "veryfast"


CFG = Config()
# =========================


def parse_time_to_seconds(t: str) -> float:
    t = t.strip()
    if not t:
        raise ValueError("空时间字符串")

    parts = t.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)

    raise ValueError(f"不支持的时间格式: {t}")


def seconds_to_hhmmssms(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    sec -= h * 3600
    m = int(sec // 60)
    sec -= m * 60
    s = sec
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def run_ffmpeg(cmd: List[str]) -> None:
    print("Running:")
    print(" ".join([f'"{c}"' if " " in c else c for c in cmd]))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise RuntimeError(f"ffmpeg 失败，退出码={p.returncode}")
    lines = p.stdout.strip().splitlines()
    if lines:
        print(lines[-1])


def read_intervals_from_dat(path: str) -> List[Tuple[str, str]]:
    """
    读取形式：
    start_time, end_time
    9:12,9:21
    ...
    - 允许表头
    - 允许空行
    - 允许行内空格
    """
    if not os.path.isabs(path):
        raise ValueError("intervals_dat 必须是绝对路径")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到区间文件: {path}")

    intervals: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            # 跳过表头（容错：只要包含 start_time 就认为是表头）
            low = line.lower().replace(" ", "")
            if "start_time" in low and "end_time" in low:
                continue

            # 支持 "a,b"
            if "," not in line:
                raise ValueError(f"第 {line_no} 行格式错误（缺少逗号）: {raw.rstrip()}")

            a, b = line.split(",", 1)
            start_raw = a.strip()
            end_raw = b.strip()

            start_sec = parse_time_to_seconds(start_raw)
            end_sec = parse_time_to_seconds(end_raw)
            if end_sec <= start_sec:
                raise ValueError(f"第 {line_no} 行结束时间必须大于开始时间: {raw.rstrip()}")

            intervals.append((start_raw, end_raw))

    if not intervals:
        raise ValueError("区间文件为空或未读到有效区间")
    return intervals


def build_output_path(input_video: str, output_dir: str, start_raw: str, end_raw: str) -> str:
    """
    命名方式：mp4文件名 + 时间区间
    例：20250926_09-30-09-41.mp4
    """
    base = os.path.splitext(os.path.basename(input_video))[0]
    safe_start = sanitize_for_filename(start_raw.replace(":", "-"))
    safe_end = sanitize_for_filename(end_raw.replace(":", "-"))
    name = f"{base}_{safe_start}-{safe_end}.mp4"
    return os.path.join(output_dir, name)


def main():
    if not os.path.isabs(CFG.input_video):
        raise ValueError("input_video 必须是绝对路径")
    if not os.path.isfile(CFG.input_video):
        raise FileNotFoundError(f"找不到输入视频: {CFG.input_video}")

    ensure_dir(CFG.output_dir)

    intervals = read_intervals_from_dat(CFG.intervals_dat)

    # 按你的要求：数组 + while 非空循环
    while intervals:
        start_raw, end_raw = intervals.pop(0)

        start_sec = parse_time_to_seconds(start_raw)
        end_sec = parse_time_to_seconds(end_raw)

        out_start_sec = max(0.0, start_sec - CFG.pad_seconds)
        out_end_sec = end_sec + CFG.pad_seconds

        start_ff = seconds_to_hhmmssms(out_start_sec)
        end_ff = seconds_to_hhmmssms(out_end_sec)

        out_path = build_output_path(CFG.input_video, CFG.output_dir, start_raw, end_raw)

        if CFG.stream_copy:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i", CFG.input_video,
                "-ss", start_ff,
                "-to", end_ff,
                "-c", "copy",
                out_path,
            ]
        else:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i", CFG.input_video,
                "-ss", start_ff,
                "-to", end_ff,
                "-c:v", CFG.vcodec,
                "-preset", CFG.preset,
                "-crf", str(CFG.crf),
                "-c:a", CFG.acodec,
                out_path,
            ]

        run_ffmpeg(cmd)
        print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
