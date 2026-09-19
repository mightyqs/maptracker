# LiDAR 历史 query 联合训练

本阶段复用原版 MapTracker 的历史 query 训练代码，将历史图像输入替换为历史
LiDAR 点云。定位头、先验地图查询和定位损失保持关闭。

## 与原版对齐的部分

1. 每帧独立提取 LiDAR BEV，用原向量头预测地图并执行 GT 匹配。
2. 从 GT 地图跨帧轨迹得到 `gt_cur2prev/gt_prev2cur`，为传播 query 构造下一帧目标，
   保留新生、消失、误检及 padding 的原版处理。
3. 按相对位姿变换折线参考点，将位姿编码与历史 query 输入原 `MotionMLP`。
   保留原版训练时的位姿噪声增强。
4. 更新后的历史 query 与当前帧的 100 个新 query 拼接，交给原向量 decoder。
5. 汇总所有帧的分类、折线回归、语义分割、Dice 损失，以及每次传播的
   `f_trans`、`b_trans`（原权重 0.1）。

保留原版 `detach`：送入下一帧 decoder 的更新 query 截断梯度，MotionMLP 通过
未截断分支的双向变换损失训练。没有改成跨整段序列的全量反向传播。
训练选取历史 query 的阈值保持 0.4，`track_fp_aug=False`，与原 nuScenes 配置一致。

历史帧的 LiDAR sparse encoder 不记录梯度，当前帧正常反传；BEV 的 SECOND/FPN、
adapter 和地图头在各帧参与训练。这对应原版对历史帧图像特征提取器限制梯度、
但保留后续 BEV 模块训练的策略，不代表两个不同前端的网络结构完全相同。
这里的 `no_grad` 不等于将历史帧编码器设为 eval，BN 行为随训练模式保持。

目前每帧的 BEV 仍独立提取，没有添加原 BEVFormer 的历史 BEV 融合，
`use_memory=False`，没有长时向量记忆库。**对齐范围是历史 query 联合训练机制，
不是原版全部时序结构或三阶段训练日程。** 位姿来自 nuScenes GT，不进行自身定位。

## GT 轨迹与采样

`tools/mapping/prepare_lidar_tracks.py` 直接复用原
`tools/tracking/prepare_gt_tracks.py` 的几何变换、栅格 IoU 匹配和全局实例 ID 分配。
训练与生成标签使用相同的地图裁剪、向量顺序和 LiDAR 平面地图坐标系；生成时不枚举折线排列。
这些 ID 是从 GT 地图几何关联得到的监督，不是模型预测的轨迹。

输出独立的 `nuscenes_map_infos_train_lidar_gt_tracks.pkl`，不覆盖原相机轨迹文件。
加载时核对样本 token；LiDAR 多帧数据拒绝没有 LiDAR 坐标系标记的旧轨迹。
标注集或几何配置变化后需要重新生成，不能只因为文件名相同就复用缓存。

| 配置 | 历史采样 |
|---|---|
| `nuscenes_lidar_mapping_2frame_mini.py` | 当前帧 + 前两帧范围内随机一帧 |
| `nuscenes_lidar_mapping_5frame_mini.py` | 当前帧 + 前十帧范围内随机四帧，与原版 five-frame/span-ten 一致 |

沿用原采样器：按帧索引升序处理，不跨场景，场景开头用第 0 帧补齐。
因此它们是历史窗口采样，不保证每次都是严格连续帧。
五帧配置的 `history_steps=4` 保存位姿信息；在线推理只需保留前一帧状态。

## 训练与验证命令

在 WSL 中执行：

```bash
conda activate maptracker
cd /mnt/e/source_code/maptracker
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 仅首次生成；本次已在本机生成 323 帧 / 8 场景的训练标签。
# 文件已存在时脚本拒绝覆盖。
python tools/mapping/prepare_lidar_tracks.py

# 原版五帧窗口训练：batch size 1、小网格、500 iter、随机初始化。
python tools/train.py \
  plugin/configs/lidar_mapping/nuscenes_lidar_mapping_5frame_mini.py \
  --work-dir work_dirs/lidar_mapping_5frame_mini_500iter \
  --no-validate --seed 0 \
  --cfg-options log_config.interval=10
```

若希望先训练两帧，换成 `nuscenes_lidar_mapping_2frame_mini.py`，并换一个 work-dir。
若从现有 LiDAR 建图权重继续初始化，在 `--cfg-options` 后增加
`load_from=work_dirs/lidar_history_train40_20260919/mapping.pth`。
这是权重初始化，不恢复优化器或迭代计数；也不是 BEVFusion 预训练权重。

小样本梯度检查：

```bash
python tools/mapping/test_online_mapping.py -v
python tools/mapping/verify_history_queries.py \
  --config plugin/configs/lidar_mapping/nuscenes_lidar_mapping_5frame_mini.py \
  --pairs 4 --steps 5 --out-dir work_dirs/lidar_history_check_manual
```

该检查固定样本及历史采样，验证实际拼接到 decoder 的 query 数、梯度截断、
前端梯度策略、逐帧损失与场景首帧补齐。另做 `pos_th=0` 的独立诊断，确保随机初始化
没有有效预测时仍能检查正例传播和 MotionMLP 反传；该诊断不更新优化器，
不写入保存的 checkpoint，也不代表默认阈值下的学习效果。

完整 mini-val 顺序评估仍使用纯观测输入：

```bash
python tools/mapping/verify_lidar.py \
  --config plugin/configs/lidar_mapping/nuscenes_lidar_mapping_5frame_mini.py \
  --checkpoint work_dirs/lidar_mapping_5frame_mini_500iter/iter_500.pth \
  --train-steps 0 --out-dir work_dirs/lidar_mapping_5frame_mini_val
```

`verify_lidar.py` 的短训练模式只用于单帧配置；多帧训练使用标准训练入口或
`verify_history_queries.py`。评估阶段不读取 GT 实例关联作为模型输入。

## 日志与判读

除了当前帧的 `cls/reg/seg/seg_dice`，历史帧有 `cls_t0/reg_t0/seg_t0` 等指标；
五帧训练有四组 `f_trans_t0…t3`、`b_trans_t0…t3`。

- `track_queries_t*`：该阶段选出的传播 query 数，按 batch 平均，可能包含误检。
- `track_matches_t*`：其中在下一帧有 GT 实例对应的数量，按 batch 平均。
- `f_trans_t* / b_trans_t*`：正向/反向传播监督；没有选中 query 时可能为 0。

初始化时分类分数可能都低于 0.4，因此“loss 有限”不足以说明传播模块得到训练。
应同时看到真实匹配、非零双向损失和 MotionMLP 梯度；不能把 query 数当作关联精度。
最终需在 scene-held-out mini-val 比较建图 AP、语义 IoU、实例关联与稳定性。

## 实测记录（2026-09-19）

两帧检查先完成 5 步，再用该权重在固定 4 组样本上训练 40 步。
后 40 步 loss 从 212.85 降至 75.48；第 34、39、40 步在原版 0.4 阈值下各选中
1 个真实匹配 query，双向损失及 MotionMLP 梯度均非零。
例如第 34 步 `f_trans=1.138`、`b_trans=1.404`。
这是小样本训练链路验证，没有得到 held-out 精度提升结论。

结果在 `work_dirs/lidar_history_train40_20260919/summary.json`；权重只包含默认阈值训练。
独立的正例诊断验证了 100 个历史 query 的传播和反传。
两帧完整检查的 PyTorch 峰值已分配显存约 1049 MiB，不是整卡占用。

五帧配置加载上述权重完成 3 步检查。第一步默认阈值下四次传播分别选出
`1/2/2/1` 个 query，四组双向变换损失全部非零；decoder 实际 query 数为
`100/101/102/102/101`。独立受控检查的 decoder query 数为
`100/200/300/400/500`，验证了逐帧传播累积与新 query 拼接，四次均有 GT 正例。
报告在 `work_dirs/lidar_history_5frame_20260919/summary.json`。
包括受控大量 query 的峰值已分配显存约 2670 MiB。

标准 `tools/train.py` 五帧入口也完成 3 iter 并保存 `iter_3.pth`，
日志在 `work_dirs/lidar_history_runner_20260919/`，记录的峰值显存约 1620 MiB。
5 项单元测试通过，包括 GT 实例重排、新生、消失、不同类别 ID 隔离和 CUDA ID 筛选。
以上均为本机小网格测试，不代表原生大网格显存，也未在本轮重新评估 mini-val 精度。
