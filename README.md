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