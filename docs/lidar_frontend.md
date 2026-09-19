# BEVFusion LiDAR 前端接入 MapTracker

本阶段将原始 LiDAR 点云编码成 MapTracker 所需的 BEV 特征，复用现有向量建图头、
定位 neck、地图编码器和 SE(2) 匹配器。本文三个配置只使用单帧、单 sweep；不使用相机图像，
不执行历史 BEV 融合或向量 query 传播。接口跑通不代表定位或建图精度已验证。

新增的独立 [LiDAR 纯建图配置](lidar_online_mapping.md) 关闭定位，并在单 sweep
前端之外启用跨帧向量 query 推理，并新增
[两帧/五帧历史 query 联合训练](lidar_history_queries.md)；验证范围与本文的三个配置分别记录。

## 对照源码与复用边界

上游采用 [MIT-HAN-Lab/BEVFusion](https://github.com/mit-han-lab/bevfusion)，
固定对照 commit `326653dc06e0938edf1aae7d01efcd158ba83de5`。
本机源码位于 `E:\source_code\bevfusion`（WSL：`/mnt/e/source_code/bevfusion`），
是独立 checkout，不是 MapTracker 的运行时依赖，也没有安装它的整套 `mmdet3d`。

主要对照路径：

| 上游路径 | 本项目对应实现 |
|---|---|
| `configs/nuscenes/seg/lidar-centerpoint-bev128.yaml` | 稀疏编码器、SECOND、SECONDFPN 的层数和通道数 |
| `configs/nuscenes/default.yaml` | 原生范围、体素尺寸与点云输入约定 |
| `mmdet3d/models/fusion_models/bevfusion.py` | 硬体素化、体素内均值、batch 坐标拼接 |
| `mmdet3d/models/backbones/sparse_encoder.py` | XYZ 稀疏布局、高度卷积和高度压缩 |
| `mmdet3d/models/backbones/second.py`、`necks/second.py` | 二维 BEV 特征提取与多尺度融合 |
| `mmdet3d/models/heads/segm/vanilla.py` | 按物理范围构造 `align_corners=False` 采样坐标 |

前端代码见 [bevfusion_lidar.py](../plugin/models/backbones/bevfusion_lidar.py)。
它复用当前 MMDetection3D/OpenMMLab 的稀疏模块和 SECOND/FPN，显式适配上游布局。
不包含 BEVFusion 相机分支、融合器、检测头或其语义头。
上游与相关 OpenMMLab 实现使用 Apache-2.0，见
[上游许可](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/LICENSE)；
本项目原有 MapTracker 许可声明仍保留。

## 编码过程与坐标约定

```text
原始 nuScenes LiDAR：N × 5 (x, y, z, intensity, ring)
  → 保留 xyz/intensity，末列改为 time_lag=0
  → 硬体素化（每体素最多 10 点）+ 体素内均值
  → SparseEncoder，四个 stage，XY 总下采样 8 倍
  → 沿 Z 压缩：128 × 2 → 256 通道
  → SECOND：128/256 两个尺度，每层段 layer_nums=5
  → SECONDFPN：两支各 256 通道，拼接成 512 通道
  → 按物理坐标采样 + 1×1 Conv / GroupNorm / SiLU
  → [B, 256, 50, 100]，交给原有建图和定位后端
```

原始第五列是 ring，不是时间差；不能将它直接输入预期 `time_lag` 的网络。
单 sweep 明确设为零，多 sweep 加载和运动补偿本阶段尚未实现。
点云保留原生 LiDAR 坐标，不在加载阶段旋转或修改 intensity。

上游体素 CUDA/CPU 实际输出 XYZ 坐标，虽然部分 Python 注释写成了 ZYX。
当前 MMCV 算子输出 ZYX，因此编码前显式重排为 `[batch,x,y,z]`。
上游 `conv_out` 的 kernel/stride 为 `(1,1,3)/(1,1,2)`，高度在最后一维；
不能直接使用默认 ZYX SparseEncoder 的 `(3,1,1)/(2,1,1)`。
稠密张量 `[B,C,X,Y,Z]` 重排为 `[B,C,Z,X,Y]` 后，将 C 与 Z 合并。

地图提取器沿用 −90° 平面旋转，因此 `x_map=y_lidar, y_map=-x_lidar`。
前端输出遵循 MapTracker 的 x 从左到右、y 从上到下递减约定，覆盖 60×30 m。
输入特征采用 BEVFusion 的范围/单元中心采样约定，输出采用当前 matcher 的端点网格。
这一步是明确的物理坐标转换，不是对非等比例矩形做简单 resize。
当前仍为平面地图任务，不处理复杂地形下的完整 SE(3) 建图。

单帧 LiDAR pipeline 同时提供与地图一致的全局平面位姿元数据。
初始位姿扰动只改变先验地图查询，不改变 LiDAR 观测。
真实地图裁剪、标签 `C=P^-1 G` 和两个辅助语义监督保持原来的定义。

## 三个配置

| 配置 | 用途 |
|---|---|
| [nuscenes_lidar_localization_mini.py](../plugin/configs/bev_localization/nuscenes_lidar_localization_mini.py) | 本机单帧定位 pipeline，较小体素网格 |
| [nuscenes_lidar_localization_bevfusion.py](../plugin/configs/bev_localization/nuscenes_lidar_localization_bevfusion.py) | 上游原生范围与体素分辨率，优先用于服务器 |
| [nuscenes_lidar_joint_mini.py](../plugin/configs/bev_localization/nuscenes_lidar_joint_mini.py) | 小网格上恢复原语义及向量损失，与定位联合反传 |

本机小网格为原生 LiDAR 范围 `[-19.2,-32,-5,19.2,32,3]` m，现已对齐体素 `(0.1,0.1,0.2)` m，
稀疏尺寸 `[384,640,41]`，高度压缩后 BEV 为 `[B,256,48,80]`。
先前 `(0.2,0.2,0.2)` m、`[192,320,41]` 的结果属于历史小网格实验，
复现时需使用 work-dir 保存的配置。当前本机范围仍比上游小，不是完整范围精度复现。

原生配置范围为 `[-51.2,-51.2,-5,51.2,51.2,3]` m，体素 `(0.1,0.1,0.2)` m，
稀疏尺寸 `[1024,1024,41]`，高度压缩后 BEV 为 `[B,256,128,128]`。
两者都输出统一的 `[B,256,50,100]`。原生配置仍默认读取 mini PKL，不自动切换完整数据集。

三个配置默认 `load_from=None`，从随机初始化开始，不误加载相机权重。
`freeze_bev=False`、`detach_bev=False`，定位梯度能更新 LiDAR 编码器与 BEV 适配层。
官方 `lidar-only-seg.pth` 已完成 210 个张量的映射与全覆盖加载检查；
完整数据配置使用转换后的前端权重。下载、映射和多卡流程见
[完整数据训练指南](lidar_full_training.md)，不直接用原始 checkpoint 的键名加载 MapTracker。
`freeze_encoder=True` 可冻结 LiDAR 编码器与 SECOND/FPN，并固定其 BN 统计，
但随机初始化时不建议使用；适配层仍可训练。

## 验证与训练命令

在 WSL 的现有 `maptracker` 环境运行，不需要安装同级 BEVFusion 包：

```bash
conda activate maptracker
cd /mnt/e/source_code/maptracker
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 几何、体素化、原有定位回归测试；有 CUDA/上游源码时执行数值对照
export BEVFUSION_SOURCE=/mnt/e/source_code/bevfusion
python -m unittest discover -s tools/localization -p 'test_*.py' -v

# 真实 mini 样本：反传、保存/重载、无 GT 推理一致性
# 输出目录必须为空，每次复验换一个目录
python tools/localization/smoke_lidar.py \
  --steps 3 --out-dir work_dirs/lidar_smoke_manual

# 同时检查原向量头、语义头与定位分支的梯度
python tools/localization/smoke_lidar.py \
  --config plugin/configs/bev_localization/nuscenes_lidar_joint_mini.py \
  --steps 3 --out-dir work_dirs/lidar_joint_smoke_manual
```

定位短训练，batch size 1：

```bash
python tools/train.py \
  plugin/configs/bev_localization/nuscenes_lidar_localization_mini.py \
  --work-dir work_dirs/lidar_localization_mini_500iter \
  --no-validate --seed 0 \
  --cfg-options log_config.interval=10
```

这里 `--no-validate` 避免误用原建图验证入口解释定位性能。完成训练后，使用统一的
固定扰动评估脚本；正确点云、跨场景错配点云和零修正基线共享相同的先验地图与目标：

```bash
python tools/localization/evaluate_fixed.py \
  plugin/configs/bev_localization/nuscenes_lidar_localization_mini.py \
  work_dirs/lidar_localization_mini_500iter/iter_500.pth \
  --out-dir work_dirs/lidar_localization_mini_val_500iter \
  --map-source real --repeats 5 --seed 20260916
```

评估仍由完整供体观测生成描述子后再交换；LiDAR 模式不加载图像。
报告增加 `observation_modality=lidar`，防止混淆相机历史实验。
联合配置可用同一训练命令替换 config，但向量质量需要额外用原建图评估工具测量，
定位评估里的语义诊断不能代替向量 AP。

## 验证边界和后续顺序

2026-09-19 在本机 WSL / PyTorch 1.13.1+cu117 / MMCV 1.7.0 完成：

| 验证 | 结果 |
|---|---|
| 定位原有测试 + LiDAR 新增测试 | 17 项通过，包括上游 SparseEncoder、SECOND/FPN 输出与梯度数值对照 |
| 小网格定位，真实 mini，batch 1 | 3 步反传；编码器、BEV 适配和定位 neck 均有有限非零梯度 |
| 小网格单帧联合配置 | 3 步反传；原向量头与语义头也收到梯度，完整推理通过 |
| 上游原生范围/分辨率 | 1 步真实点云反传与推理通过 |
| checkpoint 与 GT 隔离 | 上述三组均通过保存重载；移除 GT 地图及修正标签后定位输出一致 |
| 正式 `tools/train.py` 入口 | 小网格定位 3 iter，通过 runner、优化器、日志及 checkpoint 保存 |
| 原相机配置回归 | 真实裁剪配置加载原定位权重完成 2 iter 训练与保存 |
| 固定 mini-val 评估入口 | 81 帧 × 1 固定扰动，正确/跨场景错配点云/零修正三组完成 |

成功记录位于 `work_dirs/lidar_smoke_20260919_v2`、
`work_dirs/lidar_joint_smoke_20260919_v2`、`work_dirs/lidar_native_smoke_20260919`、
`work_dirs/lidar_runner_smoke_20260919` 和 `work_dirs/lidar_fixed_eval_smoke_20260919`。
相机回归记录在 `work_dirs/camera_lidar_interface_regression_20260919`。
小网格定位、联合配置及原生配置 smoke 的 PyTorch allocated 显存峰值分别约为
437 / 749 / 749 MiB。这不是整卡占用，也不代表更密点云、多 sweep 或长期训练的峰值。

mini-val 这里使用的是仅训练 3 iter 的随机初始化权重：正确/错配 MAP 预测一致率为
100%，平移均值约 5.175 m，差于零修正的 2.991 m。此结果只验证评估链路完整，
模型尚未学到有效点云与地图对应关系，不能作为性能结果。

上游数值对照使用同一组权重，比较稀疏编码器与 SECOND/FPN 的输出和输入梯度。
测试在隔离命名空间执行拉下来的上游 Python 类，仅替换稀疏后端导入并移除注册装饰器，
避免另一个 `mmdet3d` 覆盖现有环境。它证明对应层计算一致，不是上游整套 CUDA
编译环境、预训练精度或训练协议的完整复现。

先固定 4～8 帧验证可学习性，再做完整 mini-train/mini-val。随机初始化下的几步
smoke 只能说明接口和梯度通，不应用其误差推断 LiDAR 定位效果。
小网格会损失细线信息；随后比较原生网格与预训练初始化，再决定正式训练设置。
当前三类语义仍为人行横道、分隔线和边界，地图输入仍是官方语义地图，
不是自主累计的 LiDAR 点云地图。

历史 query 已完成短训练与反传检查；sweep 拼接、在线向量关联精度、预测位姿反馈和矿区数据泛化仍待验证。
本文的单帧 LiDAR 配置推理时每帧重置状态；纯建图在线配置则明确维护跨帧状态。
