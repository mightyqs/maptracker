# 完整 nuScenes：LiDAR 建图 → 冻结 BEV 定位

推荐路线为官方 LiDAR 地图分割预训练初始化、五帧向量建图、验证选择建图权重，
最后冻结整个观测 BEV 路径训练局部定位。这里的雷达指激光 LiDAR，不是毫米波 radar。
本页给出可执行配置与多卡命令；完整 trainval 收敛效果、多 GPU/多机吞吐尚未在本机验证。

## 1. 环境和数据

使用 Linux/WSL2。当前验证环境为 Python 3.8、PyTorch 1.13.1+cu117、
MMCV-full 1.7.0、MMDetection 2.28.2、MMDetection3D 1.0.0rc6、MMSeg 0.30.0。
安装步骤见 [installation.md](installation.md)。不要直接安装 BEVFusion 的整套
MMDetection3D 覆盖当前环境；前端已适配到本仓库依赖。

服务器准备 `v1.0-trainval` 元数据、完整 LiDAR samples、地图扩展 v1.3，目录例如：

```text
/data/nuscenes/
  v1.0-trainval/
  samples/LIDAR_TOP/
  maps/expansion/*.json
  maps/...
```

当前使用单 sweep，不读取 `sweeps/`。仍需保留元数据中的相机标定/记录供原转换器读取，
纯 LiDAR pipeline 不打开相机图像文件。原版六相机实验需要另行准备图像。
mini 与完整数据使用独立目录，避免覆盖同名 PKL；先在服务器生成绝对路径标注：

```bash
conda activate maptracker
cd /path/to/maptracker
export NUSCENES_ROOT=/data/nuscenes
export NUSCENES_ANN_ROOT=/data/nuscenes

# WSL 需要；普通 Linux 不需要额外加入 /usr/lib/wsl/lib。
# export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python tools/data_converter/nuscenes_converter.py \
  --data-root "$NUSCENES_ROOT" --version v1.0-trainval

export NUSCENES_TRAIN_FRAMES=$(python -c \
  'import os,mmcv; print(len(mmcv.load(os.path.join(os.environ["NUSCENES_ANN_ROOT"],"nuscenes_map_infos_train.pkl"))))')

python tools/mapping/prepare_lidar_tracks.py \
  --config plugin/configs/lidar_mapping/nuscenes_lidar_mapping_full.py
```

转换器使用原版 old split，并排除源码列出的四个问题场景；不加 `--newsplit`。
通常训练帧数为 27968，但训练预算以实际导出的 PKL 长度为准。
GT 轨迹生成是单进程几何匹配，完整数据需等待；不占用 GPU，也不加载预训练权重。
输出 `nuscenes_map_infos_train_lidar_gt_tracks.pkl`，不会覆盖原相机 `_gt_tracks.pkl`。
文件已存在时拒绝覆盖；修改标注集、裁剪或坐标配置后应显式备份并重新生成。

## 2. 推荐预训练模型及加载方式

优先使用 MIT-HAN-Lab BEVFusion 的 **`lidar-only-seg.pth`**。
它的纯 LiDAR BEV 地图分割任务更接近当前建图任务，配置为
`configs/nuscenes/seg/lidar-centerpoint-bev128.yaml`。
不要将 `bevfusion-seg.pth` 的融合 decoder 当作纯 LiDAR decoder 直接加载。

```bash
mkdir -p work_dirs/pretrained_ckpts
curl -L --fail --retry 3 \
  'https://www.dropbox.com/scl/fi/mi3w6uxvytdre9i42r9k7/lidar-only-seg.pth?rlkey=rve7hx80u3en1gfoi7tjucl72&dl=1' \
  -o work_dirs/pretrained_ckpts/lidar-only-seg.pth

python tools/mapping/convert_bevfusion_checkpoint.py \
  work_dirs/pretrained_ckpts/lidar-only-seg.pth \
  --output work_dirs/pretrained_ckpts/lidar-only-seg-maptracker.pth
```

本机已下载并验证的源文件 SHA256：
`b647f36a40d816789c6b448c6838c7790f26fd0ae00686e515c9902fa93ad3c8`。
转换器核对并逐项加载了 210 个参数/缓冲区张量、7279443 个元素，覆盖所选前端 100%。
它只迁移 SparseEncoder、SECOND、SECONDFPN；adapter、向量头、MotionMLP 和本项目
语义头仍需训练。原 BEVFusion 地图头类别/输出不同，故不迁移。
映射详情写入 `.report.json`；缺失键、形状不符或非有限值会报错，不自动猜测稀疏卷积轴排列。

官方权重的输入通常为当前扫描 + 9 个历史 sweep，本项目目前每个关键帧只用一次扫描。
五帧 query 传播不等于十个 sweep 点云累积。可以微调，但本配置不是上游完整输入协议的复现。

## 3. 建图训练：可选适配预热，再联合微调

完整配置为 [nuscenes_lidar_mapping_full.py](../plugin/configs/lidar_mapping/nuscenes_lidar_mapping_full.py)：
体素 `(0.1,0.1,0.2)` m、范围 `[-51.2,-51.2,-5,51.2,51.2,3]` m、
每体素最多 10 点、最多 90000/120000 个训练/测试体素。
输出仍为 `[B,256,50,100]`，地图 ROI 为 60×30 m。
历史 query 使用 five-frame/span-ten；历史 BEV 融合、长时记忆及定位保持关闭。

以下以单机 8 GPU、每卡 batch 1 为例。有效 batch 为 8 个五帧窗口，
不是 40 个相互独立的训练样本。采用 FP32；前端尚未验证 AMP。

可选先预热 2 个等效 epoch：冻结已加载的 sparse encoder 和 SECOND/FPN（含 BN），
训练新 adapter、向量头、语义头和有监督信号时的 MotionMLP：

```bash
export LIDAR_TRAIN_EPOCHS=2
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/train.py plugin/configs/lidar_mapping/nuscenes_lidar_mapping_full.py \
  --launcher pytorch --no-validate --seed 0 \
  --work-dir work_dirs/lidar_mapping_warmup \
  --cfg-options model.backbone_cfg.freeze_encoder=True log_config.interval=20
```

27968 帧、8 卡、batch 1 时每 epoch 为 3496 iter，2 epoch 的权重为 `iter_6992.pth`。
不同数据量/卡数请使用对应实际 checkpoint；不生成 `latest.pth` 符号链接。
若分类分数和匹配数量仍低，可延长预热或先做固定窗口过拟合，不以 2 epoch 作为保证。

随后以五帧联合微调 24 个等效 epoch 作为起点：

```bash
export LIDAR_TRAIN_EPOCHS=24
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/train.py plugin/configs/lidar_mapping/nuscenes_lidar_mapping_full.py \
  --launcher pytorch --no-validate --seed 0 \
  --work-dir work_dirs/lidar_mapping_full \
  --cfg-options load_from=work_dirs/lidar_mapping_warmup/iter_6992.pth \
  log_config.interval=20
```

微调阶段 `freeze_encoder=False`，前端学习率为 `1e-5`，新模块为 `1e-4`；
warmup 500 iter，随后余弦衰减。历史帧 sparse encoder 的无梯度策略仍遵循原 query 训练设计。
如果跳过预热，删除 `load_from=...` 覆盖即可使用配置中的转换后预训练权重。
不要同时加 `--autoscale-lr`；改变有效 batch 后另行明确学习率。

配置从 torchrun 的 `WORLD_SIZE` 计算迭代数，batch size 的默认值为每卡 1。
如果通过 `--cfg-options data.samples_per_gpu=2` 改 batch，必须同时重算
`runner.max_iters`、`checkpoint_config.interval` 和 warmup；Python 配置里的派生值不会因 CLI 覆盖自动重算。
断点恢复使用同一阶段、同一预算的 `--resume-from /path/to/iter_N.pth`；
跨预热/微调/定位阶段使用 `load_from`，不恢复旧优化器和迭代计数。

多机示例：两机各 4 GPU，在两台机器分别设置 `NODE_RANK=0` 和 `NODE_RANK=1`，
指定可达的主节点 IP；代码、环境、数据和权重路径须一致：

```bash
export MASTER_ADDR=10.0.0.10
export NODE_RANK=0  # 第二台设为 1
python -m torch.distributed.run --nnodes=2 --nproc_per_node=4 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port=29500 \
  tools/train.py plugin/configs/lidar_mapping/nuscenes_lidar_mapping_full.py \
  --launcher pytorch --no-validate --seed 0 \
  --work-dir work_dirs/lidar_mapping_full \
  --cfg-options load_from=work_dirs/lidar_mapping_warmup/iter_6992.pth
```

本机只有一张 GPU，仅验证过 `nproc_per_node=1` 的 NCCL/DDP 入口；多卡/多机需要在服务器验收。
建议服务器先覆盖 `runner.max_iters=20 checkpoint_config.interval=20 lr_config.warmup_iters=5`
完成所有 rank 的短训练，再启动正式预算。

## 4. 验证并选择建图权重

使用顺序、单 GPU 评估，确保每个场景从第 0 帧开始。不要把帧随机拆给多个推理进程：

```bash
python tools/mapping/verify_lidar.py \
  --config plugin/configs/lidar_mapping/nuscenes_lidar_mapping_full.py \
  --checkpoint work_dirs/lidar_mapping_full/iter_83904.pth \
  --train-steps 0 --out-dir work_dirs/lidar_full_val_24e
```

上例 checkpoint 对应 27968 帧、8 卡、batch 1、24 epoch；请替换为实际文件。
`--no-validate` 关闭训练 runner 的默认在线评估，使用独立脚本规避分布式乱序和旧 GT 缓存。
脚本支持完整 val，但 AP 单进程计算和全量结果导出会较慢，需预留主机内存与磁盘。
仅预览两个场景，不影响全量指标。

检查向量 AP、语义 IoU、`frames.json` 中的检测/传播数量，以及地图可视化。
训练日志中同时看各帧建图损失、`track_queries_t*`、`track_matches_t*` 和 `f_trans/b_trans`。
持续没有正例传播时，不能因总 loss 下降就认为历史关联已学会。
不要只按训练 loss 选择权重；应在相同 val 协议下选择效果更好的 checkpoint。
当前导出的全局地图是按实例 ID 保留最新折线，不含回环或几何融合优化。

## 5. 冻结 BEV 后训练定位

选定建图 checkpoint 后启用
[nuscenes_lidar_localization_frozen_full.py](../plugin/configs/bev_localization/nuscenes_lidar_localization_frozen_full.py)。
沿用完全相同的 LiDAR 网格和 BEV adapter。冻结 backbone、neck、向量头、语义头及
MotionMLP 的参数，并将它们设为 eval，确保 BN 统计不更新；仅训练 `localization_head.*`。
定位仍使用单帧点云和带误差位姿下真实裁剪的官方先验地图，不启用历史 query。

```bash
export MAPPING_CKPT=/path/to/selected_mapping_checkpoint.pth
export LIDAR_LOC_EPOCHS=12
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/train.py plugin/configs/bev_localization/nuscenes_lidar_localization_frozen_full.py \
  --launcher pytorch --no-validate --seed 0 \
  --work-dir work_dirs/lidar_localization_frozen_full \
  --cfg-options load_from="$MAPPING_CKPT" log_config.interval=20
```

初次从建图权重加载时，缺失 `localization_head.*` 是预期现象；
如果 backbone/adapter 等建图权重缺失或形状不符，应停止检查配置。
12 epoch 同样是建议起点，不是已验证的最佳预算。

本机可先检查冻结是否正确：

```bash
python tools/localization/smoke_lidar.py \
  --config plugin/configs/bev_localization/nuscenes_lidar_localization_frozen_full.py \
  --mapping-checkpoint "$MAPPING_CKPT" --steps 2 \
  --out-dir work_dirs/lidar_frozen_check
```

脚本逐项检查建图参数及 BN 缓冲区在优化前后完全相同，同时定位 neck 有有限非零梯度。
定位日志看 `loc_nll/loc_reg`、x/y/yaw 误差、候选命中率和辅助语义指标。

固定扰动评估包含正确点云、跨场景错配点云、零位姿修正三种条件：

```bash
python tools/localization/evaluate_fixed.py \
  plugin/configs/bev_localization/nuscenes_lidar_localization_frozen_full.py \
  work_dirs/lidar_localization_frozen_full/iter_41952.pth \
  --out-dir work_dirs/lidar_localization_full_val \
  --map-source real --repeats 5 --seed 20260916
```

仍需将 checkpoint 改为实际文件。评估会逐样本编码点云并生成地图裁剪，完整 val 比 mini 显著更慢。
应看到正确点云优于错配点云和零修正，不能只看训练命中率。
这里验证的是官方地图先验上的局部 SE(2) 定位；预测地图接入定位、矿区地图泛化、
解冻后的双任务联合训练均属于后续工作。

## 6. 已验证与未验证

- 官方 `lidar-only-seg.pth` 下载、前端参数映射及 100% 覆盖检查通过。
- 新 0.1 m 小范围五帧反传、非空历史 query、保存/重载通过。
- 官方完整范围 + 官方前端权重，在 mini 上的单卡 DDP 五帧短训练通过。
- 从建图权重启动冻结定位短训练通过，建图参数/BN 不变，去掉 GT 后推理一致。
- 完整 trainval 长期收敛、多 GPU/多机通信、预训练收益及最终定位精度尚未验证。

本机验证输出（未提交到 Git）及规模：

| 检查 | 输出目录 | 结果 |
|---|---|---|
| 新体素小范围五帧 | `work_dirs/lidar_history_0p1_verify_20260919` | 2 步反传及受控传播通过，峰值 allocated 约 2893 MiB |
| 官方权重、完整范围、五帧 DDP | `work_dirs/lidar_pretrained_native_ddp1_20260919` | 单进程 NCCL 2 iter，通过并保存；日志显存约 3084 MiB |
| 官方范围顺序 mini-val | `work_dirs/lidar_pretrained_native_val_20260919` | 81 帧、2 场景推理及 AP/IoU 评估完成 |
| 冻结状态逐项检查 | `work_dirs/lidar_frozen_0p1_verify_20260919` | 2 步定位反传、参数与 BN 不变、checkpoint 重载、去 GT 推理一致 |
| 官方范围定位 DDP | `work_dirs/lidar_frozen_native_ddp1_20260919` | 单进程 NCCL 2 iter，通过并保存；日志显存约 454 MiB |

这里的显存是 PyTorch 记录，不是整卡总占用。12 项相关单元测试通过，包括上游源码数值对照。
两步建图权重在 mini-val 的 mAP 约 0.000103（0～1 标度），语义 mIoU 为 0；
新地图头基本未训练，这只证明评估管线可运行，不能作为预训练收益或实用精度结论。

过去 500 iter 的小网格随机初始化实验使用 0.2 m 体素，不能直接与现在的 0.1 m
结果混为同一实验；复现旧结果请使用各 work-dir 内保存的配置。
