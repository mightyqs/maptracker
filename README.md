# 基于 MapTracker 的地图定位与联合建图探索

本仓库在 [MapTracker](https://github.com/woodfrog/maptracker) 上增加了图像 BEV 与
先验语义地图之间的局部定位分支。输入当前帧六相机图像、已有地图和带误差的
初始位姿，预测平面位置及航向修正。最终目标是让定位与在线建图共享 BEV
表征并联合训练。

**当前重点是单帧局部 SE(2) 定位。** 已跑通 nuScenes mini 的训练、固定扰动
验证、图像错配对照，以及按带误差位姿重新查询全局地图的裁剪路径。当前定位
配置冻结 BEV 主干、关闭历史与向量头训练；尚未验证在线建图联合训练。

原版论文介绍、作者及使用说明保留在 [README_UPSTREAM.md](README_UPSTREAM.md)。

## 1. 相比原版 MapTracker，改了什么

| 部分 | 原版 MapTracker | 当前定位分支 |
|---|---|---|
| 主要任务 | 多帧一致性向量 HD 建图 | 增加图像 BEV 与先验地图的局部 x/y/yaw 定位 |
| 观测特征 | 图像主干、BEV 编码、时序记忆 | 复用 BEV 编码，新增 `LocalizationNeck` 输出匹配描述子 |
| 先验地图 | 地图用于建图监督 | 新增 `RasterMapEncoder`，编码三通道语义地图作为定位输入 |
| 位姿估计 | 原建图任务不提供此地图匹配分支 | `SE2TemplateMatcher` 枚举候选，输出 MAP/均值位姿、协方差、熵等 |
| 定位监督 | 无此定位损失 | 软标签候选交叉熵及位姿回归损失 |
| 辅助监督 | 原语义/向量建图头 | 新增描述子地图重建、图像 BEV 语义解码两项辅助任务 |
| 初始验证 | 原版建图评估 | 新增特征级合成扰动 smoke、短训练与固定 mini-val 对照 |
| 地图裁剪 | 按真值位置提取地图监督 | 增加按带误差先验查询全局地图、重新裁剪和栅格化 |
| 数据与工程 | nuScenes/AV2 原版流程 | mini 数据支持、AV2 可选导入、定位专用配置与评估脚本 |

新增定位模块是可选分支，原版建图配置和工具仍保留。当前使用官方 nuScenes
语义地图，尚未接入独立采集会话累计的 LiDAR 地图。

| 主要代码入口 | 用途 |
|---|---|
| [core.py](plugin/models/localization/core.py) | 描述子、SE(2) 匹配、损失、概率解码 |
| [head.py](plugin/models/localization/head.py) | 定位头与两条辅助语义监督 |
| [MapTracker.py](plugin/models/mapers/MapTracker.py) | 接入训练与推理 |
| [nusc_dataset.py](plugin/datasets/nusc_dataset.py) | nuScenes 样本与全局地图查询 |
| [localization_prior.py](plugin/datasets/map_utils/localization_prior.py) | 先验位姿组合及扰动采样 |
| [evaluate_fixed.py](tools/localization/evaluate_fixed.py) | 固定扰动、正确/错配图像、零修正基线评估 |

## 2. 已验证内容与当前边界

截至 2026-09-16：

- mini-train：323 帧、8 个场景；mini-val：81 帧、2 个场景。
- 特征级扰动配置完成 500 次训练，batch size 4，无 NaN/Inf 或 OOM。
- 100/300/500 次 checkpoint 均完成 mini-val 固定扰动对照。
- 真实裁剪路径完成全量 mini-val 初始评估；batch size 1 和 4 均完成 5 步训练、反传及保存。
- 实际地图零扰动裁剪与原 GT 栅格一致，SE(2) 方向经过几何测试。
- 完整推理删除 GT 地图和位姿标签后，仍能由图像与先验地图输出相同定位结果。
- 12 项定位评估/几何/监督测试通过。

同一个旧 `iter_500.pth`，81 帧 × 5 个固定扰动的初始对照如下。
平移为平均 L2 误差，yaw 为平均绝对误差，使用 MAP 候选输出：

| 验证条件 | 精确候选命中率 | 平移误差 | yaw 误差 |
|---|---:|---:|---:|
| 不修正初始位姿 | 0.25% | 3.187 m | 2.331° |
| 特征级扰动，正确图像 | 73.83% | 0.201 m | 0.321° |
| 特征级扰动，跨场景错配图像 | 1.73% | 4.379 m | 3.091° |
| 真实地图裁剪，正确图像 | 67.41% | 0.275 m | 0.425° |
| 真实地图裁剪，跨场景错配图像 | 0.25% | 4.615 m | 3.032° |

真实裁剪这组结果使用的是旧特征任务权重，**不是完成真实裁剪长期微调后的成绩**。
新裁剪输入仍然有效，但需要针对新协议训练。

这些成绩只覆盖局部、离散候选任务。候选间隔为 1.2 m / 2°，目标也从网格抽取，
所以 0.201 m 的平均误差不能解释为连续定位已达到 20 cm 精度。初始位姿误差仍
由实验合成；“真实裁剪”指在全局地图上重新查询几何，而非变换 GT 地图特征。
当前 mini-train/mini-val 场景互不重叠，但未核验上游预训练权重是否见过相同场景。
405 组包含重复帧及相关场景，不是 405 个独立场景。

## 3. 环境

本机已验证环境是 Windows + WSL2 Ubuntu 22.04，RTX 4060 Laptop 8 GB。
以下命令均在 **WSL Ubuntu 终端**执行，使用独立的 `maptracker` conda 环境，
不要与 CALIB 环境混用。

| 依赖 | 本机验证版本 |
|---|---|
| Python | 3.8 |
| PyTorch | 1.13.1+cu117 |
| PyTorch CUDA runtime | 11.7 |
| MMCV-full | 1.7.0 |
| MMDetection | 2.28.2 |
| MMSegmentation | 0.30.0 |
| MMDetection3D | 1.0.0rc6 |

已有环境的启动与检查：

```bash
conda activate maptracker
cd /mnt/e/source_code/maptracker
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python tools/localization/smoke_test.py --device cuda
python -m unittest discover -s tools/localization -p 'test_*.py' -v
```

`/usr/lib/wsl/lib` 用于让 cuDNN 找到 WSL 提供的 `libcuda.so`。驱动支持的 CUDA
版本与 PyTorch 自带 runtime 版本不是同一概念，不要求版本号相同。
AV2 在 Python 3.8 上不可用的可选导入提示不影响当前 nuScenes 实验。

从空环境安装时可参考 [原版安装说明](docs/installation.md)，但其中 PyTorch
1.9/MMCV 1.6 是原版组合，与上表本机已验证组合不同；不要直接覆盖已跑通的环境。
原 `requirements.txt` 包括 AV2 等上游依赖，并不是当前 WSL 环境的完整锁定文件。

## 4. 数据和权重

本机数据位于 Windows `D:\nuscenes\mini`，WSL 路径为 `/mnt/d/nuscenes/mini`。
项目中 `datasets/nuscenes` 已链接到该目录。换机器时需建立自己的数据路径。

```text
datasets/nuscenes/
├── samples/
├── sweeps/
├── v1.0-mini/
├── maps/
│   └── expansion/                 # 官方地图拓展 1.3 的四个 JSON
├── nuscenes_map_infos_train.pkl
└── nuscenes_map_infos_val.pkl
```

上述 PKL 已存在时不必重建。新数据目录可执行：

```bash
python tools/data_converter/nuscenes_converter.py \
  --data-root ./datasets/nuscenes --version v1.0-mini
```

定位 quick/real-crop 配置使用 `multi_frame=False, matching=False`，无需生成
`*_gt_tracks.pkl`；原版时序建图训练仍按 [数据准备说明](docs/data_preparation.md) 操作。

第一次训练定位分支所需的原版 stage-3 权重：

```text
work_dirs/pretrained_ckpts/
└── maptracker_nusc_oldsplit_5frame_span10_stage3_joint_finetune/
    └── latest.pth
```

下载入口见 [数据与权重说明](docs/data_preparation.md#checkpoints)。这个预训练
`latest.pth` 应是实际 checkpoint 文件；不要使用以前在 NTFS 上生成失败的空链接。

本机已完成的定位权重位于 `work_dirs/localization_mini_500iter/iter_500.pth`。
新的真实裁剪配置默认从它初始化；`work_dirs` 中的数据和权重不由源码提供，
新机器必须复制该权重，或先完成下面的阶段 A。

## 5. 训练

### 阶段 A：特征级扰动基线

用于复现已经完成的基线。现有本机权重可直接进入阶段 B。

```bash
python tools/train.py \
  plugin/configs/bev_localization/nuscenes_raster_localization_quick.py \
  --work-dir work_dirs/localization_feature_500iter_run2 \
  --no-validate --seed 0 \
  --cfg-options \
  data.samples_per_gpu=4 data.workers_per_gpu=2 \
  runner.max_iters=500 lr_config.warmup_iters=50 \
  log_config.interval=10 \
  checkpoint_config.interval=100 checkpoint_config.create_symlink=False
```

### 阶段 B：带误差位姿下的真实地图裁剪

这是当前建议推进的训练入口。默认冻结 BEV，batch size 4、500 次迭代，
学习率从新计划开始，每 100 次保存权重。

```bash
python tools/train.py \
  plugin/configs/bev_localization/nuscenes_raster_localization_real_crop.py \
  --work-dir work_dirs/localization_mini_real_crop_500iter \
  --no-validate --seed 0 \
  --cfg-options log_config.interval=10
```

若阶段 A 的权重位于其他目录，再加
`load_from=work_dirs/localization_feature_500iter_run2/iter_500.pth`。
`load_from` 用于初始化权重；只有中断后继续**同一实验**时，才用
`--resume-from 路径/iter_XXX.pth` 恢复优化器与迭代状态。

只检查 pipeline 时，将阶段 B 命令的 `--work-dir` 换为新目录，并覆盖
`runner.max_iters=20 lr_config.warmup_iters=5 checkpoint_config.interval=20`。

游戏等程序会占用同一张显卡，应在 GPU 空闲时比较吞吐和显存。共享 GPU 时可
覆盖 `data.samples_per_gpu=1 data.workers_per_gpu=0`。batch 1 的 500 次约遍历
mini-train 1.55 遍；batch 4 的 500 次约 6.2 遍，二者不是相同训练样本预算。
GPU 空闲时，本机 batch 4 真实裁剪短训练的常规迭代约 0.9 秒，日志显存约 2.3 GB；
这不是整卡总占用，也不是不同设备上的性能保证。

日志位于输出目录的 `*.log`、`*.log.json`，关键指标包括：

- `loc_nll`、`loc_reg`：定位损失；NLL 使用软标签，不以降到零为目标。
- `loc_exact_acc`：网格目标下的精确命中率，245 候选的均匀随机水平约 0.41%。
- `loc_err_x_m`、`loc_err_y_m`、`loc_err_yaw_deg`：训练 MAP 候选误差。
- `loc_map_recon_miou`、`loc_bev_sem_miou`：地图重建及图像 BEV 语义指标。
- `grad_norm`：梯度范数，检查是否有 NaN/Inf。

新裁剪训练中，地图重建使用带误差的先验地图作目标；图像 BEV 语义仍使用
真值位置地图作目标，两者不能混用。更详细的坐标定义见
[真实地图裁剪说明](docs/localization_real_crop.md)。

## 6. mini-val 验证与对照

定位训练暂用 `--no-validate`，因为原版默认评估器汇总的是地图指标，不能替代
定位评估。下面脚本不更新权重，使用全部 mini-val，并保存固定扰动清单。

复现旧特征级扰动对照：

```bash
python tools/localization/evaluate_fixed.py \
  plugin/configs/bev_localization/nuscenes_raster_localization_quick.py \
  work_dirs/localization_mini_500iter/iter_100.pth \
  work_dirs/localization_mini_500iter/iter_300.pth \
  work_dirs/localization_mini_500iter/iter_500.pth \
  --out-dir work_dirs/localization_mini_val_feature_run2 \
  --map-source feature --repeats 5 --seed 20260916
```

真实裁剪训练完成后，在相同裁剪上比较旧权重与新权重：

```bash
python tools/localization/evaluate_fixed.py \
  plugin/configs/bev_localization/nuscenes_raster_localization_real_crop.py \
  work_dirs/localization_mini_500iter/iter_500.pth \
  work_dirs/localization_mini_real_crop_500iter/iter_500.pth \
  --out-dir work_dirs/localization_mini_val_real_crop_compare \
  --map-source real --repeats 5 --seed 20260916
```

新权重尚不存在时，删除第二个 checkpoint 参数即可评估旧权重的初始性能。
每次使用新的空输出目录，以免覆盖已有实验。

三种条件共用目标帧、地图与扰动：`correct` 使用正确图像，`mismatched` 使用
另一验证场景的整组六相机观测，`identity` 预测零修正。错配图像先使用供体自身
相机几何编码，再替换观测描述子，不混用两帧的相机外参。

| 输出 | 内容 |
|---|---|
| `manifest.json` | sample token、固定目标、错配供体；真实裁剪还记录全局位姿 |
| `prior_rasters.npz` | 真实裁剪模式生成的实际先验地图 |
| `*_records.json` | 每组 MAP/均值预测、目标、熵及置信度 |
| `*_summary.json`、`summary.json` | 全局与分场景误差、命中率、成功率、语义诊断 |

默认成功阈值为平移 ≤1 m 且 yaw ≤1°。在当前 1.2 m / 2° 网格目标上，
MAP 成功率与精确命中率相同。概率均值输出也单独统计，避免只看最高分候选。
confidence 目前由熵计算，尚不是校准后的成功概率。

## 7. 下一步工作

1. 完成真实裁剪协议下的 mini-train 微调，并与旧权重在相同固定 mini-val 对比。
2. 可视化 `scene-0916` 的失败帧，增加同场景不同位置的困难错配及边界影响对照。
3. 验证非网格连续扰动、局部位姿细化和可信的拒绝更新机制。
4. 定位验证稳定后，再解冻共享 BEV、恢复建图损失，验证联合训练。

现有接口允许定位与建图使用同一 BEV，但仅修改 `freeze_bev` / `detach_bev`
并不等于完成联合训练；还需要恢复建图分支、配置损失并进行双任务评估。

## 8. 文档与原版功能

- [定位 quickstart 与固定评估协议](docs/localization_quickstart.md)
- [特征级扰动实验记录](docs/localization_mini_val_20260916.md)
- [真实地图裁剪、坐标约定与训练命令](docs/localization_real_crop.md)
- [原版训练、推理、评估与可视化](docs/getting_started.md)
- [原版环境安装](docs/installation.md) · [原版数据准备](docs/data_preparation.md)

## 9. 来源、引用与许可

本项目基于 MapTracker（ECCV 2024 Oral）及其所使用的 BEVFormer、StreamMapNet、
MapTR 等开源工作扩展。本分支的定位实验结果不代表原版 MapTracker 的论文结果。
上游作者信息、项目链接与致谢保留在 [原 README](README_UPSTREAM.md)。

```bibtex
@inproceedings{chen2024maptrakcer,
  author = {Chen, Jiacheng and Wu, Yuefan and Tan, Jiaqi and Ma, Hang and Furukawa, Yasutaka},
  title = {MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping},
  journal = {arXiv preprint arXiv:2403.15951},
  year = {2024}
}
```

保留原仓库 [LICENSE](LICENSE) 与 [LICENSE_GPL](LICENSE_GPL)：原声明限制商业使用，
研究用途遵循其中列出的 GPLv3 条款。本分支没有变更该许可声明。
