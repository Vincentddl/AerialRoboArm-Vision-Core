# 模型与素材总索引

更新时间：2026-10-02。

本页用于回答三个问题：哪个文件正在使用、每段素材是做什么的、哪些内容只是历史记录。详细说明还放在各目录自己的 `README.md` 中。

## 状态标记

- **当前使用**：默认实时程序直接加载。
- **训练来源**：用于生成当前模型的训练样本。
- **标定来源**：用于建立相机或舵机与视线角的关系，不用于识别外部目标。
- **历史保留**：用于复现实验或比较新旧结果；不建议删除，但不是默认运行依赖。
- **验证专用**：只用于独立评估，不能混入对应模型的训练集。

## 当前实时链路

```text
USB Video（640×480）
  → models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt
  → 泡沫轮廓质心（异常尺寸时退回检测框中心）
  → configs/camera_2p1mm_640x480_fisheye.json
  → 相对镜头光轴的真实视线偏移角
  → configs/servo_to_optical_angle_foam_center_lut_20260816_v1.json
  → 目标对应的舵机 g 角度
```

入口：`run_realtime.py`。当前不做抛接，也不预测 0.4 秒后的目标位置。

## 模型

| 文件 | 状态 | 作用 |
| --- | --- | --- |
| `models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt` | **当前使用** | 当前 `run_realtime.py` 的泡沫目标分割权重；默认置信度 0.50。文件名保留训练时的 candidate 标记。 |
| `models/foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.json` | **当前使用** | 当前权重的来源、校验值和验证记录。 |
| `models/foam_center_v9_seg.pt`、`models/foam_center_v9_seg.json` | **历史保留** | V9 基础模型及报告，用于旧结果复现和比较。 |
| `models/foam_center_v10_obb.pt` | **候选、尚未部署** | YOLO11s-obb旋转框模型；可输出几何中心和朝向，验证集误检更少，但困难样本尾部误差尚未优于V9。 |
| `models/foam_center_v10_obb.json` | **候选说明** | V10训练环境、哈希、训练指标和同基准中心评估结果。 |
| `models/foam_board_2p1mm_v8.pt` | **历史保留** | V8 检测框模型；用于 V9 数据构建时提供粗 ROI，也用于 V8/V9 对比。默认实时程序不加载它。 |
| 根目录 `yolo11n.pt` | 工具依赖/备用 | 通用 YOLO 基础权重，不是本项目泡沫目标成品模型。 |
| 根目录 `sam2.1_t.pt` | 工具依赖/备用 | 通用分割权重；当前 V9 实时推理不加载它。 |

模型细节见 [`models/README.md`](models/README.md)。

## 标定文件

| 文件 | 状态 | 作用 |
| --- | --- | --- |
| `configs/servo_to_optical_angle_foam_center_lut_20261006_restricted_v1.json` | **新装夹预览候选** | 2026-10-06 训练表限制到 `-77°～-27°`，边界容差 0°；严格范围验证未通过，使用专用预览入口，不启用自动发送。 |
| `configs/servo_to_optical_angle_foam_center_lut_20261006_merged_v1.json` | **三录像合并候选** | 53 段稳定保持、576 帧完整目标拟合；三段来源都已成为训练素材，等待第四段独立测试，仅预览。 |
| `configs/camera_2p1mm_640x480_fisheye.json` | **当前使用** | 2.1 mm 鱼眼镜头内参和畸变参数，把像素射线换算为视线角。仅适用于同一镜头、焦距和 640×480 分辨率。 |
| `configs/servo_to_optical_angle_foam_center_lut_20260816_v1.json` | **当前使用** | 无红点泡沫中心到舵机角度的 11 节点实测关系，范围 `g=-97°～-47°`。 |
| `configs/servo_to_optical_angle_red_marker_lut_20260815_v3.json` | **红点监视器/历史保留** | 夹爪红点视线标定，供 `run_servo_angle_monitor.py` 和历史验证使用；不是当前无红点入口默认表。 |
| `configs/servo_to_optical_angle_red_marker_lut_20260813_v2.json` | **机械滑移前备份** | 文件原样保留，默认实时程序不再加载；仅用于历史比较或显式回退。 |
| `configs/servo_to_optical_angle_red_marker_lut_20260806_v1.json` | **历史保留、回退备份** | 越限前使用的红点查找表；不再由默认实时程序加载，保留用于比较和快速回退。 |
| `configs/servo_to_optical_angle_yolo_v8_20260804_v1.json` | **历史保留** | V8 检测框中心与舵机角的早期线性关系；不供 V9 默认实时链路使用。 |
| `configs/servo_to_optical_angle_20260804_v1.json` | **历史保留** | 2026-08-04 泡沫块中心的早期线性拟合；保留用于复现和比较。 |

配置细节见 [`configs/README.md`](configs/README.md)。

## 相机棋盘素材

| 目录 | 状态 | 内容与用途 |
| --- | --- | --- |
| `data/camera_calibration/chessboard/` | **当前相机标定来源** | 53 张有效棋盘图片；9×6 内角点、20 mm 方格，用于生成当前鱼眼相机 JSON。 |

棋盘图片只负责求相机内参、光心和畸变，不直接给出物体世界坐标、舵机角度或夹爪机械零位。详见 [`data/camera_calibration/README.md`](data/camera_calibration/README.md)。

## 原始视频素材

| 目录 | 状态 | 内容与用途 |
| --- | --- | --- |
| `data/raw/v8_source_20260803/` | **历史保留、V8来源** | V8 原始构建录像，187.36 秒、9368 帧；用于复现 V8 数据构建。 |
| `data/raw/servo_angle_20260804/` | **历史标定来源** | 泡沫块随舵机按 `-30°～-100°` 引导角运动；生成两份早期线性标定。 |
| `data/raw/servo_accuracy_validation_20260805/` | **历史验证** | 第一版红点准确性验证，`-30°～-100°` 单向 15 步；被后续大红点双向素材取代。 |
| `data/raw/servo_accuracy_large_marker_20260805/` | **历史标定来源** | 大红点双向录像，`-30°→-100°→-30°` 共 29 步；用于生成越限前 v1 LUT。 |
| `data/raw/servo_accuracy_large_marker_repeat2_20260805/` | **历史标定来源** | 大红点独立重复录像，`-35°→-100°→-35°` 共 27 步；用于生成越限前 v1 LUT。 |
| `data/raw/servo_recalibration_after_limit_20260813_session2/` | **当前标定来源** | 机械越限恢复后的第一组大红点双向录像，`-35°→-100°→-35°` 共 27 步。 |
| `data/raw/servo_recalibration_after_limit_20260813_session3/` | **当前标定来源/独立验证** | 相同机械状态下的第二组独立双向录像；与 session2 共同生成 v2 LUT。 |
| `data/raw/servo_red_encoder_validation_after_slip_20260815_session1/` | **当前 v3 标定来源** | 机械滑移后的红点整数节点双向素材；使用有效的 `-45°～-100°` 节点生成 v3，排除出画的 `-40°/-35°`。 |
| `data/raw/servo_optical_relation_validation_20260815_session2/` | **当前 v3 独立验证** | 使用未参与拟合的 `-47°、-52°…-97°` 中间整数角双向验证 v3 插值和回差。 |
| `data/raw/target_detection_no_red_normal_slow_20260806/` | **V9训练来源** | 无红点、正常光、慢速、独立手持泡沫目标；V9 训练场景。 |
| `data/raw/target_detection_no_red_normal_medium_20260806/` | **V9训练来源** | 无红点、正常光、中速目标；V9 训练场景。 |
| `data/raw/target_detection_no_red_normal_fast_20260806/` | **V9训练来源** | 无红点、正常光、快速目标和运动模糊；V9 训练场景及实时回放测试素材。 |
| `data/raw/target_detection_no_red_dim_light_20260806/` | **V9训练来源** | 无红点、暗光目标；V9 训练场景。 |
| `data/raw/target_detection_no_red_alt_background_20260806/` | **V9验证专用** | 无红点、不同背景；整段只进入验证集，避免同一录像泄漏到训练集。 |

原始素材的时长、帧数和配套文件说明见 [`data/raw/README.md`](data/raw/README.md)。

## 训练数据集

| 目录 | 状态 | 内容 |
| --- | --- | --- |
| `data/training/foam_center_v9/` | **当前训练集** | 分割标签；训练 719 张（522 正样本、197 负样本），验证 155 张（76 正样本、79 负样本）。 |
| `data/training/foam_center_v10_obb/` | **候选训练集** | 从V9轮廓转换的旋转框标签；保持相同训练/验证场景划分。 |
| `data/training/foam_board_v8/` | **历史训练集** | 检测框标签；保留用于复现 V8 和新旧模型对照。 |

数据目录的结构和标签含义见 [`data/training/README.md`](data/training/README.md)。

## 实验输出

`outputs/` 保存训练曲线、最佳/末轮权重、红点分析表、中心误差评估和调试图。它们不是新的原始素材；完整说明见 [`outputs/README.md`](outputs/README.md)。其中：

- `outputs/training/foam_center_v9_seg/weights/best.pt` 与保留的 V9 基础权重 `models/foam_center_v9_seg.pt` 内容相同。
- `last.pt` 是最后训练轮次，不等于验证表现最好的权重，默认实时程序不使用。
- `center_eval_*.json` 是比较 V8/V9 与不同置信度的评估记录，不是运行配置。

## 删除与修改规则

1. 不要删除标为“当前使用”的权重或配置，否则实时入口会失效。
2. 不要把 `alternate_background` 验证视频抽帧加入 V9 训练集，否则当前验证成绩会失去独立性。
3. 红点素材只标定夹爪中心射线，不训练外部泡沫目标检测器；完成标定后，真实运行可拆除红点。
4. `.mkv` 是原始证据，`.angles.jsonl` 是角度事件，`.timestamps.jsonl` 是逐帧时刻，`.ffmpeg.log` 是录制日志，四者应作为一组保存。
5. 若更换镜头、分辨率、相机安装角度、夹爪结构或齿轮传动，应重新检查相机/舵机标定，不能直接沿用旧文件。
