# AerialRoboArm Vision Core

精简后的单轴机械臂视觉项目。日常入口是 `run_realtime.py`；红点舵机角度验证入口是
`run_servo_angle_monitor.py`。

模型、标定文件和全部素材的用途及状态见 [`ASSET_CATALOG.md`](ASSET_CATALOG.md)。不确定一个文件能否删除或是否参与当前实时程序时，先查这张表。

## 当前推荐配置

三段 2026-10-06 录像已合并重新拟合，新的命令角候选表为 `configs/servo_to_optical_angle_foam_center_lut_20261006_merged_v1.json`。筛选后使用 53 个保持段、576 帧完整目标，仍存在跨录像方向相关偏差；训练帧对编码器的 MAE 为 0.90°、P95 为 3.17°，尚未验收。`start_independent_validation.bat` 已切到这张表，等待第四段新录像测试；`start_restricted_vision_preview.bat` 只预览新表，不发送 HC-13。三段旧录像不再作为独立测试。

2026-10-06 新装夹的候选视觉范围为 `-77°～-27°`，配置文件是 `configs/servo_to_optical_angle_foam_center_lut_20261006_restricted_v1.json`。严格端点复算的范围覆盖率为 90.35%、P95 角误差为 2.20°，尚未通过验收。双击 `start_restricted_vision_preview.bat` 可查看此候选角度，HC-13 输出关闭；普通 `run_realtime.py` 仍加载下面的历史运行表，不适用于新装夹的自动控制。独立验证入口改为范围内的 `-29°→-74°→-29°` 19 档，端点仍需单独核对。

- 目标模型：`models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt`
- 旧版基线：`models/foam_center_v9_seg.pt`、`models/foam_board_2p1mm_v8.pt`（保留，不覆盖）
- 相机标定：`configs/camera_2p1mm_640x480_fisheye.json`
- 泡沫中心—舵机标定：`configs/servo_to_optical_angle_foam_center_lut_20260816_v1.json`
- 默认置信度：0.50
- 当前模式：使用最新检测位置，不预测 0.4 秒后的目标位置

V9首先分割泡沫目标轮廓，再计算中心：

```text
正常目标尺寸（检测框占画面 1%～13%） → 可见泡沫轮廓的面积质心
很小目标或近距离超大目标                 → 检测框中心（更稳健）
```

该中心经过鱼眼模型换算为机械臂运动平面内的真实视线偏移角，再通过舵机标定查找表反算目标舵机角度。

## 启动

```powershell
python -m pip install -r requirements.txt
python run_realtime.py
```

显示终端目标数据：

```powershell
python run_realtime.py --print-targets
```

带红点检查视觉反推的舵机等效角度：

```powershell
python run_servo_angle_monitor.py
```

红点只用于建立和验证“舵机角度—夹爪中心视线”关系。最终检测外部泡沫目标时可以拆除红点；夹爪当前角度应优先读取 HX8 绝对编码器。

## V9基础模型训练与结果

V9使用2026-08-06录制的五段无红点素材。完整视频场次不会同时进入训练和验证：四段用于训练，浅色背景整段只用于验证。

- 训练正样本：522张轮廓标注
- 验证正样本：76张轮廓标注
- 人工确认负样本：训练197张，验证79张
- 最佳轮次：19
- 验证分割：P=0.973，R=1.000，mAP50=0.990，mAP50-95=0.864
- RTX 4070 Laptop GPU单帧模型推理：约4.4 ms

在独立浅色背景验证集上，V8框中心与V9混合中心对比如下：

| 指标 | V8框中心 | V9轮廓/框混合中心 |
| --- | ---: | ---: |
| 正样本召回 | 97.37% | 100% |
| 负样本误检 | 6/79 | 2/79（conf=0.30） |
| 平均中心误差 | 5.24 px | 2.69 px |
| 中位中心误差 | 3.87 px | 1.24 px |
| P95中心误差 | 15.91 px | 10.76 px |
| 平均机械臂平面角误差 | 0.406° | 0.196° |
| P95机械臂平面角误差 | 1.156° | 0.729° |

这些误差以经过审查的图像轮廓质心为参考，不等同于外部量角器或三维测量系统的绝对角度精度。

上表是保留的 V9 基础模型结果。当前启动脚本使用 2026-08-16 的 gripper-axis 候选模型与无红点泡沫中心 V1 查找表；在独立实拍录像和 MCU RTT 回读上的 CPU 复算平均舵机角误差为 `0.8483°`（原报告 `0.8484°`）。复现命令和限制见 [`REPRODUCTION_GUIDE_20261002.md`](REPRODUCTION_GUIDE_20261002.md)。

详细过程见 `docs/foam_center_v9_training.md`。

## 主要目录

- `run_realtime.py`：V9日常实时入口
- `run_servo_angle_monitor.py`：红点舵机角度验证入口
- `vision/`：检测、跟踪、中心选择、鱼眼角度和舵机换算
- `models/`：当前 gripper-axis 模型及保留的 V9/V8 基线
- `configs/`：相机和舵机—视觉标定
- `tools/`：录像、数据构建、训练、中心评估和项目检查
- `data/raw/`：原始录像
- `data/training/foam_center_v9/`：V9分割训练集和审查图
- `data/training/foam_board_v8/`：保留的V8检测训练集

## 角度定义

`camera_plane_angle_deg` 是目标中心射线投影到单轴机械臂运动平面后，相对镜头光轴的有符号角度；画面向下为正。

镜头相对机械水平向下安装30°时：

```text
legacy_mechanical_angle_deg = camera_plane_angle_deg + 30°
```

当前 `run_realtime.py` 使用 2026-08-16 的 11 节点无红点泡沫中心查找表：

```text
目标中心 → 鱼眼角度 → 查表反算 servo_command_deg
有效标定范围：g=-97°～-47°
```

该实测关系已经包含40:48外啮合齿轮的影响，运行时不要再次乘齿轮比。
MCU 当前软件限位为 `-90°～+85°`；HC-13 自动发送只接受两者交集 `-90°～-47°`，不会把超出限位的视觉目标钳位后发送。

`servo_to_optical_angle_red_marker_lut_20260815_v3.json` 仍用于专门的红点角度监视器和历史对照，不是当前无红点实时入口的默认表；更早的 V2/V1 表继续保留作回退。

## 素材与Git

`data/`默认不提交普通Git，避免大量视频和训练图像进入仓库历史。模型和素材应使用Git LFS或独立数据存储。

本代码仓库保存程序、测试、标定 JSON、模型元数据和说明文档；`*.pt` 权重、`data/` 录像与训练图像、`outputs/` 运行结果以及本机备份不随代码上传。克隆后可运行 `python -m unittest discover -s tests -v`；运行 `run_realtime.py` 还需将 `models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt` 放回 `models/`。完整离线评估和 `tools/project_doctor.py` 还依赖本机保存的素材，不能仅凭代码仓库完成。
