@echo off
setlocal
cd /d "%~dp0"
set "VISION_PY=D:\app\miniconda\envs\python312\python.exe"
if not exist "%VISION_PY%" set "VISION_PY=python"
"%VISION_PY%" run_realtime.py --source 1 --servo-angle-calibration configs/servo_to_optical_angle_foam_center_lut_20261008_combined_v1.json --hc13-port COM3 --device 0 --imgsz 640 --show-centers --new-track-confirm-frames 1 --min-age 1 --hc13-confirm-frames 1 --window-title "Fast FPV control - one frame - press A to send" %*
pause
