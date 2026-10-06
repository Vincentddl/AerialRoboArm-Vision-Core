# 工具说明

这些命令都应在项目根目录运行。

## 一键独立验证

双击根目录 `start_independent_validation.bat`。程序检查相机、RTT 和 HX8 回读后，打开 `-22°→-77°→-22°` 的 23 档录制窗口。按 R 开始，G 发送当前引导角，停稳后空格记录 3 秒，空格进入下一档；完成后 S 保存、Q 退出。退出后自动核对素材并用冻结的命令角候选表做 CPU 评估，报告保存到 `outputs/independent_validation/`。模型 `.pt` 与候选 JSON 需要在本机存在。程序不会自动替换运行标定。

已保存的独立视频也可用 `python tools/run_independent_validation.py --video <视频.mkv>` 重新检查和评估；`--check-only` 仅检查设备，不录制。

## 手动录制 50 FPS 角度素材

```powershell
python tools/capture_ffmpeg_manual.py --angles=-30,-35,-40,-45,-50,-55,-60,-65,-70,-75,-80,-85,-90,-95,-100
```

- 按 `R` 手动开始录制。
- 舵机到达屏幕提示角度并稳定后按空格。
- 按 `B` 返回上一个角度。
- 按 `S` 手动停止并保存。
- 按 `Q` 或 `Esc` 退出。

默认保存到 `data/raw/new_angle_recordings/`。

## 使用现有棋盘照片重新标定鱼眼镜头

```powershell
python tools/calibrate.py --model fisheye --cols 9 --rows 6 --images data/camera_calibration/chessboard --out outputs/camera_calibration_check.json
```

## 采集新棋盘照片

```powershell
python tools/capture_chessboard.py --source 0 --width 640 --height 480
```

`build_foam_board_2p1mm_v8.py` 是 V8 数据整理过程的复核工具。它使用已整理的 V8 数据作为基底，并把原始视频中的审核帧重建到 `outputs/foam_board_v8_rebuild/`，不会覆盖当前训练集。

## 构建和训练V10旋转框候选模型

```powershell
python tools/build_foam_center_v10_obb.py
python tools/train_foam_center_v10_obb.py
```

构建工具读取V9轮廓并生成最小面积旋转框，不修改V9。输出目录已存在时会拒绝覆盖。中心评估示例：

```powershell
python tools/evaluate_foam_center_obb.py --model models/foam_center_v10_obb.pt --conf 0.30
```
