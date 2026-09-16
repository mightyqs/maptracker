# 带误差位姿下的真实地图裁剪

本阶段在 nuScenes 全局矢量地图中按带误差的先验位姿重新查询几何、裁剪、
栅格化，然后编码地图。初始位姿误差仍由实验合成，但输入地图不再来自对 GT
地图特征的变换。旧的 `nuscenes_raster_localization_quick.py` 保留作为对照。

## 坐标与数据路径

设 `G` 是观测坐标系到全局地图的 SE(2)，`P` 是先验地图裁剪坐标系到全局地图
的 SE(2)。定位标签定义为 `C = inverse(P) @ G`，也就是观测坐标到先验地图坐标
的变换，与匹配器 `grid_sample` 的 output-to-input 方向一致。

实验先采样修正 `C`，构造 `P = G @ inverse(C)`，再到全局地图中查询 `P` 对应
窗口；预测后可以通过 `P @ C_pred` 恢复全局位姿。平移与旋转必须组合，不能
仅将 yaw 和全局 x/y 分别取负。

现有 nuScenes 提取器在 LiDAR 局部几何上额外执行 −90° 旋转，因此这里的
`G.yaw = lidar_global_yaw + 90°`。查询提取器时再减回 90°，保持与旧 GT 栅格
一致。图像投影及其元数据保持原有观测几何；不使用带误差的地图先验去更改
相机外参，也不在此阶段改变原有 ego/LiDAR 轴约定。

每帧提供三个独立监督/输入量：

| 字段 | 用途 |
|---|---|
| `semantic_mask` | 真值位姿下地图，监督图像 BEV 语义及原有建图任务 |
| `localization_map` | 先验位姿下重新查询的地图，输入定位分支并作为地图重建目标 |
| `localization_target_pose` | `C` 的 `[x, y, yaw]`，外部提供的定位监督 |

地图查询使用 `NuscDataset.get_localization_prior`，复用已加载的全局地图。
因此先验窗口可以包含 GT 窗口之外新进入的几何。两个地图以相同方式栅格化并
翻转 y 轴。定位核心使用外部标签，关闭特征级二次扰动。

新配置开启 `require_prior_map`：若忘记传先验地图则显式报错，不会静默退回 GT
地图。训练还要求提供独立的 BEV 语义目标。推理可只提供先验地图与相机输入，
不需要 GT 地图和目标修正；生成真实在线先验地图的上游系统仍需提供全局先验。

## 训练

新配置默认从旧特征级训练的 `iter_500.pth` 初始化，保持冻结 BEV、batch size 4，
用完整 mini-train 训练 500 次，每 100 次保存一个普通 checkpoint 文件。

```bash
conda activate maptracker
cd /mnt/e/source_code/maptracker
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python tools/train.py \
  plugin/configs/bev_localization/nuscenes_raster_localization_real_crop.py \
  --work-dir work_dirs/localization_mini_real_crop_500iter \
  --no-validate --seed 0 \
  --cfg-options log_config.interval=10
```

这是新协议下的微调，使用 `load_from` 加载权重，重新开始优化器和学习率计划，
不使用 `--resume-from` 恢复旧实验的迭代状态。不要把新协议指标下降直接解释为
代码回归，需要先看几何检查和同协议下的验证对照。

本机在 GPU 空闲时使用 `samples_per_gpu=4, workers_per_gpu=2`。此前 batch 4
诊断时游戏同时占用显卡，导致吞吐下降；用户已确认这一外部占用，因此那次
运行不作为新裁剪路径的性能基准。不要把 torch 的显存日志等同于整卡占用。
GPU 空闲后重新完成了 batch 4 的 5 步训练及保存，常规迭代约 0.9 秒，日志峰值
显存 2305 MB，loss 和梯度均有限；记录在 `work_dirs/localization_mini_real_crop_idle_gpu_20260916`。
如需与其他程序共享 GPU，可在命令末尾覆盖
`data.samples_per_gpu=1 data.workers_per_gpu=0`；该设置已完成 5 步反传和保存，
常规迭代约 0.8～0.9 秒，日志显存约 950 MB。
batch 1 的 500 次只约等于遍历 mini-train 1.55 遍，与 batch 4 的 500 次
（约 6.2 遍）并非相同样本预算。

默认范围和网格不变：平移 ±3.6 m / 1.2 m，yaw ±4° / 2°，245 个候选。
训练每次取样重新抽扰动；数据集普通验证入口按 token 固定扰动；下面的独立
评估入口覆盖为每帧 5 个固定目标，与前一轮实验使用完全相同的清单生成方法。

外部目标也支持连续值，训练可配置 `data.train.localization_prior.sampling=continuous`。
但当前标准对照仍为网格目标，先完成这一阶段再单独评估连续目标；连续训练时
`loc_exact_acc` 表示最近网格分类正确率，不代表精确连续位姿命中。

## 固定 mini-val 对照

```bash
python tools/localization/evaluate_fixed.py \
  plugin/configs/bev_localization/nuscenes_raster_localization_real_crop.py \
  work_dirs/localization_mini_500iter/iter_500.pth \
  work_dirs/localization_mini_real_crop_500iter/iter_500.pth \
  --out-dir work_dirs/localization_mini_val_real_crop_compare \
  --map-source real --repeats 5 --seed 20260916
```

第二个 checkpoint 需要等真实裁剪训练完成后再加入命令；只传旧 checkpoint 即可
测量跨协议的初始性能。每次使用新的空输出目录。

`--map-source auto` 默认按配置选择真实裁剪或旧特征扰动。真实裁剪模式先生成
全部先验地图，保存为 `prior_rasters.npz`；多个 checkpoint 共用这些地图。
`manifest.json` 额外保存每组 `prior_global_pose` 和 `observation_global_pose`。
正确配对、跨场景图像错配、零修正基线仍共享相同地图和扰动。

语义诊断保持使用真值位置的地图，便于与旧实验比较；定位指标才使用带误差的
先验裁剪。输出中的 `semantic_protocol` 对这一点做了明确标注。

## 验证方法与边界

```bash
python -m unittest discover -s tools/localization -p 'test_*.py' -v
python tools/localization/smoke_test.py --device cuda
```

协议与几何测试覆盖：SE(2) 组合、−90° 提取器约定、零扰动、GT 窗外几何进入
先验窗口、确定性验证扰动、外部/旧合成标签一致性、连续标签反传，以及两条
语义监督的隔离。

这仍是官方语义矢量地图上的局部 SE(2) 定位，不是独立会话 LiDAR 建图，也没有
验证在线建图联合训练。当前地图窗口与观测窗口同为 60×30 m，仍存在有限覆盖、
裁剪边界和卷积边界效应；未来可用更大先验地图窗口及内区评分做进一步对照。

## 2026-09-16 初始跨协议结果

同一旧 `localization_mini_500iter/iter_500.pth`，相同 mini-val 81 帧 × 5 个目标，
对比特征级扰动与真实裁剪。这里尚未使用真实裁剪训练后的权重。

| 条件 | 精确候选命中率 | 平移均值 / m | yaw 均值 / ° |
|---|---:|---:|---:|
| 零修正基线 | 0.25% | 3.187 | 2.331 |
| 特征扰动，正确图像 | 73.83% | 0.201 | 0.321 |
| 真实裁剪，正确图像 | 67.41% | 0.275 | 0.425 |
| 真实裁剪，跨场景错配 | 0.25% | 4.615 | 3.032 |

真实裁剪的正确配对在 `scene-0103` 命中率为 87.50%，`scene-0916` 为 47.80%；
后者是优先分析的失败场景。正确配对概率均值位姿误差仍为 1.115 m / 1.482°。
旧权重能迁移到新裁剪，但连续定位精度与拒绝更新机制仍未验证。

原始结果：`work_dirs/localization_mini_val_real_crop_20260916/summary.json`。
同目录保存全局先验位姿、固定标签、实际裁剪栅格和逐组结果。该评估没有更新权重。

实际数据的几何检查：mini-val 首帧零扰动裁剪与原 GT 栅格完全相同；目标修正为
`[2.4 m, -1.2 m, 4°]` 时，去掉四周 10 像素的内区 raster IoU 从 0.177 提升
到按真值修正后的 0.853。栅格化、插值、裁剪后几何合并差异使其不要求达到 1。

完整 `MapTracker.forward_test` 也已验证：删除 GT `semantic_mask` 和
`localization_target_pose` 后，仅用图像与先验地图仍能输出相同 MAP 位姿。
