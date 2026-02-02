import os
import subprocess

# =========================
# 配置区
# =========================
INPUT_DIR = r"E:\temp\videos"          # 包含 mp4 的文件夹路径
BITRATE = "192k"
FFMPEG_EXE = r"C:\ffmpeg\bin\ffmpeg.exe" 
# =========================

def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"文件夹不存在: {INPUT_DIR}")
        return

    # 遍历文件夹
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.mp4')]
    
    if not files:
        print("该文件夹下没有找到 .mp4 文件。")
        return

    print(f"找到 {len(files)} 个视频文件，准备开始转换...\n")

    for filename in files:
        full_input_path = os.path.join(INPUT_DIR, filename)
        
        # 生成同名的 mp3 路径
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(INPUT_DIR, base_name + ".mp3")

        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i", full_input_path,
            "-vn",
            "-b:a", BITRATE,
            "-loglevel", "error", # 减少日志输出，只显示错误
            output_path
        ]

        print(f"正在处理: {filename} -> mp3")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print("\n所有任务完成！")

if __name__ == "__main__":
    main()