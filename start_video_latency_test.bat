@echo off
setlocal
cd /d "%~dp0"
set "VISION_PY=D:\app\miniconda\envs\python312\python.exe"
if not exist "%VISION_PY%" set "VISION_PY=python"
"%VISION_PY%" tools\measure_video_latency.py --source 0 --backend dshow --width 640 --height 480 --pattern-hz 8 --seconds 45 %*
pause
