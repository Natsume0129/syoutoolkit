import os
import subprocess

# =========================
# 配置区（只改这里）
# =========================
INPUT_FILE = r"E:\temp\0107.m4a"
BITRATE = "192k"
FFMPEG_EXE = r"C:\ffmpeg\bin\ffmpeg.exe"  # 改成你自己的 ffmpeg.exe 实际路径

# =========================


def main():
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    base, _ = os.path.splitext(INPUT_FILE)
    output_file = base + ".mp3"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", INPUT_FILE,
        "-b:a", BITRATE,
        output_file
    ]

    subprocess.run(cmd, check=True)
    print(f"Converted:\n{INPUT_FILE}\n-> {output_file}")


if __name__ == "__main__":
    main()
