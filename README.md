# syoutoolkit
my toolkit for daily usage


## extractor
从视频中提取一段区间。

## m4atomp3
音频文件格式转换

## splitmp3
切分mp3

## greenbackgournd 视频抠图workflow
- 1st：调整输入输出目录，原视频运行batch_rvm_extract.py
- 2nd: 对于运行出来的结果，调整reencoding.py的输入输出，运行reencoding
- 3rd: 对于2nd的结果（视频），这是一个视频文件夹，
- 打开face tracking的目录
- 运行：
Get-ChildItem -Path "E:\Matsuda_data\手动标注( vgg-face分析)\10-29_output_mp4" -Recurse -Filter *.mp4 |
  ForEach-Object {
    python .\FaceTracking\CUI-pyplot\face_detection.py --movie_file "$($_.FullName)"
  }
- 记得修改目录

## make_still_image.py
用来制作连续帧的图片，用法：

'''powershell
python make_stillimage.py [-h] --input INPUT --out OUT [--step STEP] [--pattern PATTERN] [--max_count MAX_COUNT] [--tile_w TILE_W] [--gap GAP] [--pad PAD] [--font_size FONT_SIZE] [--label_h LABEL_H] [--bg BG] [--text TEXT] [--target_aspect TARGET_ASPECT]
make_stillimage.py: error: the following arguments are required: --input, --out



python make_stillimage.py --input "E:\path\to\frames" --step 10 --out "still.png"
'''