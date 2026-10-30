@echo off
setlocal
cd /d "%~dp0"
set "VISION_PY=D:\app\miniconda\envs\python312\python.exe"
if not exist "%VISION_PY%" set "VISION_PY=python"
"%VISION_PY%" run_realtime.py --source 1 --servo-angle-calibration configs/servo_to_optical_angle_foam_center_lut_20261008_combined_v1.json --no-hc13 --device cpu --imgsz 640 --show-centers --window-title "Analog FPV foam detection -77 to -32 (preview)"
echo.
pause
