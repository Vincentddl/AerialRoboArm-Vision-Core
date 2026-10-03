# Vision-Core 复现指南（2026-10-02）

## 结论与范围

本机已复现 **无红点泡沫中心 → HX8 等效目标角度**的离线验证。真实来源是 2026-08-16 的录像、引导角事件和 MCU RTT 编码器回读。使用 CPU 推理时，本次重算的 21 个保持点、230 个采样帧与原报告一致，平均绝对角度误差为 `0.8483°`（原报告 `0.8484°`），P95 为 `1.1983°`，最大值 `1.5144°`。62 个单元测试通过。

当前 `run_realtime.py` 还在 3 秒真实录像片段上以 `--no-hc13` 跑通，保存了 149 条目标记录。**实时摄像头 → HC-13 → MCU → HX8 自动动作**尚无本次复现的真机结论，需要在硬件接入后单独验收。

## 1. 使用已验证的环境

在 PowerShell 中进入 `D:\Study_data\AerialRoboArm-Vision-Core`。本次成功使用：

```powershell
$visionPy = 'D:\app\miniconda\envs\python312\python.exe'
& $visionPy -c "import sys,cv2,numpy,torch,ultralytics,serial; print(sys.version.split()[0], cv2.__version__, numpy.__version__, torch.__version__, ultralytics.__version__, serial.__version__)"
```

本机版本为 Python `3.12.9`、OpenCV `4.13.0`、NumPy `2.1.2`、PyTorch `2.6.0+cu118`、Ultralytics `8.4.33`、pyserial `3.5`。`requirements.txt` 钉住 Ultralytics `8.3.218`；本次已验证的 CPU 路径使用的是现有 `8.4.33` 环境。GPU 路径重算平均误差为 `1.0706°`，未达到当前评估脚本的 `1.0°` MAE 门槛，因此核对原报告时应固定 `--device cpu`。

## 2. 文件与单元测试

```powershell
& $visionPy tools/project_doctor.py
& $visionPy -m unittest discover -s tests -v
```

预期：项目自检输出 `AerialRoboArm Vision Core: OK`；单元测试 `Ran 62 tests`、`OK`。自检只证明文件和配置可读取，不代替角度评估。

## 3. 重跑独立录像角度验证

```powershell
$sessionBase = 'data/raw/foam_no_red_servo_relation_validation_session2_20260816/foam_no_red_servo_relation_validation_independent_20260816_221106_443784'
& $visionPy tools/evaluate_no_red_servo_relation.py `
  --video "${sessionBase}.mkv" `
  --events "${sessionBase}.angles.jsonl" `
  --rtt "${sessionBase}.rtt.jsonl" `
  --servo-calibration 'configs/servo_to_optical_angle_foam_center_lut_20260816_v1.json' `
  --device cpu `
  --output 'outputs/repro_no_red_servo_relation_cpu.json'
```

预期：`completed_holds=21`、`total_sampled_frames=230`、`total_detected_frames=230`、`acceptance.overall_pass=true`。平均绝对误差应接近 `0.8484°`，P95 约 `1.1983°`，最大约 `1.5144°`。原报告在 `outputs/evaluation/no_red_servo_relation_v1_independent_validation_20260816.json`；本次实测报告在 `outputs/repro_20261001_no_red_servo_relation_cpu.json`。

## 4. 当前入口的离线冒烟测试

`run_realtime.py` 的**代码实际默认**是模型 `models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt` 和角度表 `configs/servo_to_optical_angle_foam_center_lut_20260816_v1.json`。根目录 README 和 `ASSET_CATALOG.md` 仍描述较早的 V9 通用模型加红点 V3 表；复现时以启动脚本中的默认值为准。

可从上述录像裁出一个 3 秒片段，再在不发送 HC-13 的条件下运行：

```powershell
ffmpeg -hide_banner -loglevel error -ss 3 -i "${sessionBase}.mkv" -t 3 -an -c:v mjpeg -q:v 3 -y 'outputs/repro_short_hold.avi'
& $visionPy run_realtime.py `
  --source 'outputs/repro_short_hold.avi' `
  --device cpu --no-hc13 --no-window `
  --save-jsonl 'outputs/repro_current_runtime_short_hold.jsonl'
```

预期：启动时打印候选模型和 marker-free V1 角度表；输出 JSONL 非空，包含 `servo_command_deg`、`servo_calibration_valid`。本次片段得到 149 条有效目标记录。单张图片不足以通过默认的两帧目标确认，故用短视频片段测试入口。

## 5. 实时真机验收顺序

1. 接入 640×480、2.1 mm 鱼眼相机，以 `& $visionPy run_realtime.py --no-hc13 --print-targets` 只看检测。确认分割中心、鱼眼角度、`servo_command_deg` 与目标位置一致；先在实测 `g=-97°..-47°` 范围内检查，边界容差内的值也应谨慎核对。
2. 接入 PC 侧 HC-13（脚本默认 `COM5`、`230400 8N1`），保持 MCU 遥控器在 MANUAL，先验证串口 ACK 和 MCU 的 `hc13/vf` 计数，不让 AUTO 驱动主臂。
3. 支撑机械臂、保留急停操作空间后，在预览窗口按 `A` 解锁发送，再将 RC 切到 AUTO。观察 MCU `tgt/pos/rd/status`，确认目标角、编码器角和动作方向一致。
4. 对目标丢失、置信度不足、角度超出标定范围、HC-13 断链和急停逐项验收；任一条件失效时应停止发送并由 MCU 保持或停机。

前三节已在当前电脑离线完成。第 5 节依赖相机、HC-13、MCU、HX8 与机械结构同时到场，不能由离线报告替代。
