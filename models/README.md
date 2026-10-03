# 模型说明

## 当前模型：`foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt`

- 状态：**当前 `run_realtime.py` 默认使用**；文件名保留训练时的 candidate 标记。
- 类型：YOLO11s-seg 实例分割，类别只有 `foam_board`。
- 输入：项目相机的 640×480 画面；推理时可由框架缩放。
- 当前入口置信度：0.50。
- 中心定义：检测框占画面 1%～13% 时使用可见泡沫轮廓的面积质心；超出该范围时退回检测框中心，防止极小或贴近镜头的掩膜不稳定。
- 用途：识别无红点泡沫目标；配合 `../configs/servo_to_optical_angle_foam_center_lut_20260816_v1.json` 生成舵机目标角。
- 完整校验信息：`foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.json`。
- SHA-256：`04595F4F5D92AFC5EC219D1A7A3E20D2B1A8A6AD7D01230813A954D019B7A8C3`。

独立实拍录像与 MCU RTT 回读的无红点角度验证：21 个保持点、230/230 帧检出；CPU 复算平均误差 `0.8483°`，P95 `1.1983°`。这是视觉和编码器的对应关系验证，不是外部量角器意义上的绝对角度精度。

## 保留的 V9 基础模型：`foam_center_v9_seg.pt`

- 状态：历史结果复现与比较，不是当前实时入口默认权重。
- 配套说明：`foam_center_v9_seg.json`。
- SHA-256：`EAD54F6F05AF3329C729F58EF4F7A21881C5EA91DB7125699A578047181C3288`。

## 历史模型：`foam_board_2p1mm_v8.pt`

- 状态：**保留，不是默认模型**。
- 类型：基于 YOLO11s 微调的目标检测模型，只输出检测框，没有分割轮廓。
- 用途：复现旧实时方案；为 V9 自动标注提供粗略候选区域；与 V9 做基线比较。
- 中心定义：检测框中心，因此容易受到框大小和遮挡变化影响。
- 不要用它覆盖 V9，也不要因为 V9 已启用就删除它；重新构建 V9 数据时仍可能需要。

## 候选模型：`foam_center_v10_obb.pt`

- 状态：**已训练，尚未部署**；`run_realtime.py`使用 gripper-axis V9 模型。
- 类型：YOLO11s-obb旋转框检测。
- 中心定义：预测旋转框的中心，同时可输出四角位置和目标朝向。
- 最佳轮次：16；验证Recall 1.000、mAP50 0.995、mAP50-95 0.884。
- 同一OBB中心参考下：平均中心误差6.04 px，79张负样本误检0张。
- 限制：P95中心误差26.18 px，困难样本尾部误差尚未优于V9，所以没有自动切换默认实时模型。
- 完整说明：`foam_center_v10_obb.json`和`../docs/foam_center_v10_obb_training.md`。

## 根目录通用权重

- `../yolo11n.pt`：Ultralytics 通用检测基础权重/备用工具输入，不识别本项目专用泡沫类别。
- `../sam2.1_t.pt`：通用分割辅助权重/备用工具输入，不参与当前 `run_realtime.py` 的默认推理。

运行时应确认终端显示加载的是 `foam_center_v9_gripper_axis_normal_bg03_20260816_candidate.pt`。模型的训练素材、训练集和验证集不得只凭文件名混用，参见 `../ASSET_CATALOG.md`。
