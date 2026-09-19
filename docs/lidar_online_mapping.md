# LiDAR 纯建图：训练与在线推理管线验证

本阶段关闭定位，专门验证 LiDAR → BEV → 语义/向量地图，以及相邻帧之间的
向量 query 传播。`localization_cfg=None`，不构建定位头、不生成先验裁剪、不计算
定位损失。推理只接收点云、位姿元数据和场景顺序；GT 地图只用于训练和离线评估。

## 当前在线模式的含义

```text
当前 LiDAR sweep → BEVFusion 编码器 → 当前 BEV
                                      ↓
上一帧 query + 折线 → 相对位姿变换 → 原向量 decoder → 当前折线、类别与分数
                                      ↓
                             筛选、维护实例 ID
                                      ↓
                          下一帧传播 / 全局坐标导出
```

采用 nuScenes 的 GT ego pose 和 LiDAR 外参生成相对位姿，不估计自身运动，
不是无位姿输入的 SLAM。定位模块完全不参与这次验证。

LiDAR 编码器仍独立处理每一帧。`history_steps=1` 与 `test_time_history_steps=1`
在此模式下保留前一帧的位姿信息，供向量变换使用；不把历史特征输入 LiDAR 编码器。
`use_memory=False`，尚未启用原 MapTracker 的长时向量记忆库和多帧 BEV 融合。

通过 `lidar_online_mapping=True` 开启跨帧推理。序列必须从场景第 0 帧开始且按顺序
处理；跳帧索引、乱序或未从头开始会报错。新场景第 0 帧清空旧 query 状态。
已有单帧 LiDAR 配置继续按每帧独立方式推理。

本文的 `nuscenes_lidar_mapping_mini.py` 仍是单帧建图：原分类、折线回归、语义分割和 Dice 损失参与反传。
这会训练 LiDAR 前端和建图头，但**不会训练跨帧 query 传播中的 MotionMLP**，
也不包含跨帧实例匹配损失。新增的两帧/五帧配置已接通原版跨帧监督，见
[历史 query 联合训练](lidar_history_queries.md)。本文下方的五步结果属于此前单帧管线检查，
不能当成多帧训练或在线精度验证的结果。

## 配置与入口

- [本机小网格配置](../plugin/configs/lidar_mapping/nuscenes_lidar_mapping_mini.py)：batch size 1，500 iter，默认随机初始化。
- [BEVFusion 原生网格配置](../plugin/configs/lidar_mapping/nuscenes_lidar_mapping_bevfusion.py)：恢复上游范围和分辨率，默认数据仍为 mini。
- [验证脚本](../tools/mapping/verify_lidar.py)：短训练、非空 query 检查、完整 mini-val 顺序推理、AP/IoU、导出与可视化。

前端的坐标、体素化和源码对照见 [LiDAR 前端说明](lidar_frontend.md)。
在线模式修复了原向量实例 ID 使用 CPU 张量而筛选 mask 位于 CUDA 的索引问题。

## 本机命令

WSL 中执行：

```bash
conda activate maptracker
cd /mnt/e/source_code/maptracker
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 纯建图配置、相对位姿、新生 ID 与全局导出坐标测试
python tools/mapping/test_online_mapping.py -v

# 五步建图训练，再逐场景验证 mini-val；输出目录须为空
python tools/mapping/verify_lidar.py \
  --train-steps 5 --out-dir work_dirs/lidar_mapping_verify_manual
```

使用完整 mini-train 从随机初始化训练，batch size 1（此命令不会接续上面的五步权重）：

```bash
python tools/train.py \
  plugin/configs/lidar_mapping/nuscenes_lidar_mapping_mini.py \
  --work-dir work_dirs/lidar_mapping_mini_500iter \
  --no-validate --seed 0 \
  --cfg-options log_config.interval=10
```

训练后单独执行在线评估：

```bash
python tools/mapping/verify_lidar.py \
  --config plugin/configs/lidar_mapping/nuscenes_lidar_mapping_mini.py \
  --checkpoint work_dirs/lidar_mapping_mini_500iter/iter_500.pth \
  --train-steps 0 --out-dir work_dirs/lidar_mapping_mini_val_500iter
```

训练日志看 `cls`、`reg`、`d*.cls/reg`、`seg`、`seg_dice` 和 `grad_norm`。
这里不应出现 `loc_nll`、`loc_reg` 等定位指标。单次 loss 不代表建图质量，
需结合验证集向量 AP、语义 IoU 和可视化判断。

评估复用原 Chamfer AP 实现，阈值为 0.5/1.0/1.5 m，保留全部候选参与 AP。
GT 从当前验证配置重新读取并核对 token，绕过原评估器不能区分 mini/full 的全局
缓存。使用单进程计算 CPU 距离，避免 GPU 推理后 fork PyTorch 线程池产生锁等待。
语义指标采用原分割输出的最大类别与 0.4 阈值，并对齐 GT 高度轴翻转。

## 非空传播检查与正式评估的区别

未训练模型可能在默认阈值下没有任何可传播 query，因此验证脚本先做一个独立
的受控检查：首帧全部接收，后两帧保留已有实例并禁止新生。检查 100 个 query
确实进入 MotionMLP、ID 连续、重复场景首帧的输出一致、切换场景重置及乱序拒绝。
单元测试另外检查新生实例的 ID 分配和 CPU/CUDA mask 一致性。

正式 mini-val 评估恢复默认阈值：首帧检测 0.4，后续新生 0.6，已有轨迹 0.5。
受控检查的低阈值结果不进入 AP、IoU 或全局地图导出。
默认阈值下是否存在实际传播，应查看 `frames.json` 的 `propagated` 和 `active_ids`，
不能只因为受控检查通过就认为模型学会了跨帧关联。

## 本机实测结果（2026-09-19）

使用小网格配置、随机初始化，在 WSL 的 `maptracker` 环境完成以下验证：

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 4 项单元测试 | 全部通过，含 CUDA ID 筛选 | 配置隔离、相对位姿、实例 ID 与全局坐标导出正确 |
| 5 步真实点云训练 | loss 为 191.864、162.237、132.510、127.974、116.240 | 编码器、BEV 解码器、适配层、向量头和分割头均有有限非零梯度 |
| 权重保存与严格重载 | 通过 | 建图 checkpoint 可保存并恢复 |
| 强制非空 query 传播 | 相邻两次各传播 100 个 query，ID 保持一致 | 传播接口可执行；场景重置、首帧重放和乱序拒绝通过 |
| 完整 mini-val | 2 个场景、81 帧顺序推理完成 | 推理输入无 GT 地图，评估和导出流程可执行 |
| 原训练入口 | `tools/train.py` 完成 3 iter 并保存 `iter_3.pth` | 数据整理、优化器和 checkpoint 接口可用 |

**建图效果尚未验证有效**：这次五步权重的各类向量 AP 和语义 IoU 均为 0；
默认阈值下接受的实例数及实际传播 query 数均为 0，因此两个场景的全局地图均为空。
可视化展示的是未经阈值筛选的 top-20 原始候选，不代表已有有效地图。
五步 loss 下降只能作为反传检查，不能证明泛化或跨帧关联已经学会。

完整结果在 `work_dirs/lidar_mapping_verify_20260919_v2/`，重点查看
`summary.json`、`controlled_tracking.json`、`frames.json` 和 `mapping_preview.png`。
标准训练入口日志在 `work_dirs/lidar_mapping_runner_20260919/`。
验证脚本记录的 PyTorch 峰值已分配显存约 699 MiB，标准训练入口记录约 813 MiB；
这不是整张显卡的总占用，也不代表 BEVFusion 原生大网格的显存需求。
本次未验证原生大网格配置，未运行定位，也未进行跨帧传播训练。

## 输出内容

| 文件 | 内容 |
|---|---|
| `mapping.pth` | 脚本执行短训练后保存的纯建图权重 |
| `training.json` | 逐步 loss 与模块梯度；只评估模式为空列表 |
| `controlled_tracking.json` | 强制非空传播的接口检查结果 |
| `frames.json` | 逐帧 scene、索引、筛选后检测数、传播数、活跃 ID 与耗时 |
| `submission_vector.json` | 原向量评估格式，含语义预测 |
| `pos_predictions.pkl` | 原工具格式的筛选后实例与场景 ID |
| `global_map_latest.json` | 每个场景、每个实例 ID 最近一次折线，变换到全局坐标 |
| `mapping_preview.png` | 两个场景首帧 GT 与 top-20 预测折线对照 |
| `summary.json` | 验证范围、向量 AP、语义 IoU、显存与位姿来源 |

`global_map_latest.json` 是可核查的最新观测累积结果，不包含多次观测几何优化、
回环或重复实例融合；相同数值 ID 在不同场景中彼此独立。
可视化只显示 top-20 便于阅读，AP 计算不做这个截断。

后续使用历史 query 配置验证小样本跨帧建图可学习性，再扩展 mini-train；
结合无历史对照评估建图与关联质量，之后再考虑历史 BEV 融合和长时记忆库。
