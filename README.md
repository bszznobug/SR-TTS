# SR-TTS：超分辨传感器定位与增量学习

> 基于多通道时序信号（TDMS）的超分辨触觉/压力传感器定位系统：从原始信号预处理、HDF5 数据集构建，到「通道注意力 + Transformer + CNN」分类/回归模型训练，并支持在曲面数据上的增量学习（EWC + 知识蒸馏 + 记忆重放），解决接触点在传感器阵列上的精确定位问题。

---

## 1. 项目概述

本项目的核心任务是 **超分辨传感器定位**：一块 32 通道的压力/触觉传感器阵列，其物理分辨率有限，但需要推断接触点位于更细网格（当前配置为 `51 × 51`）上的具体位置。

- **输入**：`[B, 32, 100]` 的时序信号 —— 32 个通道、100 个时间点（降采样到 100 Hz 后 1 s 的数据）。
- **输出**：接触点的网格坐标 `(row, col)`，支持两种预测方式：
  - **独立分类**（默认，`joint_mode: False`）：分别预测行号和列号；
  - **联合分类**：预测所有 `row × col` 组合中的单一网格。
  - 另提供 **回归** 变体，直接输出连续 `(x, y)` 坐标。

系统按「平面 → 曲面」两阶段训练：先在平面（基准）数据上预训练，再在曲面数据上做 **增量学习**，通过三种持续学习技术抑制灾难性遗忘。

---

## 2. 目录结构

```
SR-TTS_code/
├── data/                        # 数据管线（预处理 + 合并 + 加载）
│   ├── raw_data/                # 原始输入
│   │   ├── origin_TDMS/         #   原始 .tdms 文件（NI 格式）
│   │   └── label/               #   标签 CSV（每个文件一个，含 x_mm/y_mm）
│   ├── processed_data/          # 处理产物
│   │   ├── segmented_output/    #   分割后的 CSV 信号段（按文件名/组名分层）
│   │   └── training_data*.h5    #   合并后的 HDF5 数据集
│   ├── config_data.yaml         # 数据管线配置
│   ├── tdms_reader.py           # TDMS 读取 → 滤波 → 分割 → CSV
│   ├── data_preprocess.py       # 预处理主流程（批量 TDMS + 合并 H5）
│   ├── data_merger.py           # SmartDataMerger：CSV → HDF5 合并器
│   └── loader.py                # PyTorch Dataset / DataLoader
│
├── main/                        # 训练 / 评估 / 增量学习入口
│   ├── config.yaml              # 主配置（模型、训练、增量学习、路径）
│   ├── train.py                 # 平面数据分类训练
│   ├── qianyi_learn.py          # 曲面增量学习（EWC + KD + Replay）
│   ├── test.py                  # 单测试集评估（混淆矩阵 / ROC）
│   └── test_weitiao.py          # 批量测试集评估
│
├── src/                         # 模型与损失函数
│   ├── model.py                 # 模型定义（分类 / 回归）
│   └── Loss.py                  # 损失函数
│
├── utils/                       # 可视化工具
│   ├── visualization.py         # 训练曲线 / 混淆矩阵 / ROC
│   └── feature_visualizer.py    # 特征提取 + t-SNE 可视化
│
└── output/                      # 输出（模型权重、训练历史、评估结果）
```

---

## 3. 系统架构与数据流

```
                     ┌───────────────────────────────────────────┐
                     │             数据预处理阶段                 │
                     └───────────────────────────────────────────┘
  raw .tdms  ──► tdms_reader.py ──► data_preprocess.py ──► SmartDataMerger
  (32 通道)       滤波/去DC/         批量调度                 (CSV → HDF5)
                   找峰/分割 CSV
                                                        │
                                                        ▼
                                            training_data.h5
                                            (sample_* 组，含 signal、
                                             row/col、channel_names)
                                                        │
                     ┌───────────────────────────────────┼──────────────────────┐
                     │                                   ▼                       │
                     │                     data/loader.py                         │
                     │              HighPerformanceSensorDataset                 │
                     │              collate_padded_fast（映射到 32 通道 + mask）  │
                     │                                   │                       │
                     │             ┌─────────────────────┴───────────────┐       │
                     │             ▼                                     ▼       │
                     │     train_loader (平面)                  curve/plane loader │
                     │             │                                     │       │
                     │             ▼                                     ▼       │
                     │      main/train.py                       main/qianyi_learn.py
                     │    （平面预训练分类）                     （曲面增量学习）
                     │             │                                     │       │
                     │             ▼                                     ▼       │
                     │    src/model.py（通道注意力+Transformer+CNN）       │       │
                     │    src/Loss.py（多任务损失）                        │       │
                     │             │                                     │       │
                     │             └──────────────► best_model.pth ──────┘       │
                     │                          （作为增量学习预训练权重）        │
                     └──────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                                     main/test.py / test_weitiao.py
                                     （评估：准确率、混淆矩阵、ROC）
```

---

## 4. 模型架构

`SuperResolutionSensorModel` 采用「**通道注意力 + Transformer + 1D-CNN**」的混合结构：

| 阶段 | 模块 | 说明 |
|------|------|------|
| ① 输入处理 | `channel_embed` + `PositionalEncoding` | 单通道信号映射到 `embed_dim//2` 维并注入正弦位置编码 |
| ② 通道注意力 | `ChannelAttention` | 全局平均池化 → 压缩-激活 → Softmax，按通道重要度加权 |
| ③ 特征融合 | `projection` | 拼接嵌入特征与注意力特征，投影到统一 `embed_dim=64` |
| ④ 时序建模 | `TransformerEncoder` | 3 层、4 头、`dim_ff=256`，建模任意时间点依赖 |
| ⑤ 空间压缩 | `spatial_cnn`（1D-CNN + 全局池化） | `64→128→256`，输出全局特征 `[B, 256]` |
| ⑥ 分类头 | `joint_head` / `row_head`+`col_head` | 联合分类输出 `[B, R, C]`；独立分类输出两个 logits |

- **独立分类模式**（默认）：`row_head` 与 `col_head` 分别输出 `num_rows=51` 和 `num_cols=51` 类的 logits。
- **联合分类模式**：`joint_head` 输出 `num_rows × num_cols = 2601` 维，再 reshape 为 `[B, 51, 51]`，通过 `max(dim=2)` / `max(dim=1)` 边缘化得到行/列预测。
- **回归模式**：`SuperResolutionSensorModelRegression` 将分类头替换为 `regression_head`（`256 → 2`），直接输出连续 `(x, y)` 坐标，配合 `RegressionLoss`（MSE）。

---

## 5. 关键特性

### 5.1 训练技巧（`train.py`）

- **学习率预热 + 平缓衰减**：`WarmupReduceLROnPlateau` 前 5 轮线性升温，之后按验证损失 `ReduceLROnPlateau`（factor=0.7，patience=10）。
- **标签平滑**（`label_smoothing=0.1`）：缓解高维分类（51×51）过拟合。
- **类别权重**：按逆频率计算行/列类别权重，缓解样本不均衡。
- **梯度裁剪**（`max_grad_norm=1.0`）防止梯度爆炸。
- **早停**：以平滑后的验证准确率为核心指标，支持保存最佳模型与断点。

### 5.2 增量学习（`qianyi_learn.py`）

在曲面数据上增量微调时，联合使用三种持续学习技术抑制对平面任务的灾难性遗忘：

| 技术 | 核心思想 | 关键超参 |
|------|----------|----------|
| **EWC**（弹性权重巩固） | Fisher 信息矩阵度量参数对旧任务重要度，惩罚重要参数漂移 | `ewc_lambda=500` |
| **知识蒸馏 KD** | 冻结的教师模型（平面预训练）软化输出作为软目标 | `kd_temperature=4`, `kd_weight=0.5` |
| **记忆重放 Replay** | 环形缓冲区采样平面样本，训练时复习 | `replay_buffer_size=1000`, `replay_weight=0.3` |

另支持 **分阶段解冻**（前 `freeze_backbone_epochs` 轮只训分类头，之后按 `unfreeze_layers` 解冻 `head+cnn+transformer`）与 **分层学习率**（主干 `backbone_lr_scale=0.1`）。

---

## 6. 环境依赖

| 依赖 | 用途 |
|------|------|
| `torch` | 模型、训练、推理 |
| `numpy` / `pandas` | 数值计算与数据处理 |
| `h5py` | HDF5 数据集读写 |
| `PyYAML` | 配置文件解析 |
| `nptdms` | 读取 NI `.tdms` 原始文件 |
| `scipy` | 滤波、降采样、信号处理 |
| `scikit-learn` | 混淆矩阵、ROC、t-SNE |
| `matplotlib` / `seaborn` | 可视化 |
| `tqdm` | 进度条 |
| `psutil` | 合并器内存自适应 |

安装示例：

```bash
pip install torch numpy pandas h5py PyYAML nptdms scipy scikit-learn matplotlib seaborn tqdm psutil
```

---

## 7. 快速上手

### 7.1 数据预处理（TDMS → HDF5）

1. 将原始 `.tdms` 文件放入 `data/raw_data/origin_TDMS/`；
2. 将坐标标签 CSV（含 `x_mm`、`y_mm` 列，文件名与 TDMS 同名）放入 `data/raw_data/label/`；
3. 按需修改 `data/config_data.yaml`（采样率、滤波、分割窗口、坐标步长、`mode` 等）；
4. 执行：

```bash
python data/data_preprocess.py
```

产物：`data/processed_data/segmented_output/`（CSV 段）与 `data/processed_data/training_data.h5`（合并数据集）。

### 7.2 平面数据训练

确认 `main/config.yaml` 中 `raw_h5_path` 指向预处理产物，然后：

```bash
python main/train.py
```

产物：`output/results_temp/best_model.pth` 与 `training_history_with_row_col.csv`（含行/列准确率）。

### 7.3 曲面增量学习

配置 `pretrain_weights_path`（平面权重）与 `curve_h5_path`（曲面数据），然后：

```bash
python main/qianyi_learn.py
```

产物：`output/results_temp/best_incremental_model.pth`。

### 7.4 评估

- 单测试集：`python main/test.py`
- 批量测试集：`python main/test_weitiao.py`（扫描 `test_h5_dir` 下所有 `.h5`）

产物：每数据集独立的 `results.csv`、混淆矩阵、ROC 曲线与汇总报告。

### 7.5 特征可视化（t-SNE）

```bash
python utils/feature_visualizer.py
```

提取模型各层（投影层 / Transformer / CNN / 分类头）特征并做 t-SNE 降维可视化。

---

## 8. 配置说明（`main/config.yaml`）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `num_rows` / `num_cols` | `51` / `51` | 超分辨网格行/列数 |
| `num_channels` / `seq_len` | `32` / `100` | 输入通道数 / 时间步 |
| `joint_mode` | `False` | `False`=独立分类，`True`=联合分类 |
| `mode` | `classification` | `classification` 或 `regression` |
| `learning_rate` | `1e-4` | AdamW 基础学习率 |
| `num_epochs` | `200` | 训练轮数 |
| `label_smoothing` | `0.1` | 标签平滑系数 |
| `class_weight` | `True` | 是否启用类别权重 |
| `lr_warmup_epochs` | `5` | 学习率预热轮数 |
| `max_grad_norm` | `1.0` | 梯度裁剪范数 |
| `patience` | `20` | 早停耐心 |
| `unfreeze_layers` | `head+cnn+transformer` | 增量学习解冻策略 |
| `use_ewc` / `ewc_lambda` | `True` / `500` | EWC 开关与强度 |
| `use_kd` / `kd_temperature` / `kd_weight` | `True` / `4` / `0.5` | 知识蒸馏配置 |
| `use_replay` / `replay_buffer_size` / `replay_weight` | `True` / `1000` / `0.3` | 记忆重放配置 |

数据管线参数见 `data/config_data.yaml`（滤波、分割窗口、坐标步长 `step_um` 等）。

---

## 9. 注意事项

- **路径配置**：`config.yaml` 中的路径为绝对路径，部署到新环境时需同步修改。
- **标签来源**：`data_merger` 支持三种 `mode` —— `origin`（从文件名解析行列）、`classification`（从标签 CSV 按步长离散化）、`regression`（保留连续坐标）。
- **可复现性**：训练固定随机种子（`seed=42`），数据集分割使用固定生成器。
- **内存策略**：加载器支持 `full_load`（全内存）与动态加载两种模式，内存不足时自动回退到 `fallback_batch_size`。
