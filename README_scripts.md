# SR-TTS Project Structure Overview

A super-resolution sensor localization system: from raw TDMS signal preprocessing and HDF5 dataset construction, to channel-attention + Transformer + CNN model training, and incremental learning on curved-surface data (EWC + Knowledge Distillation + Memory Replay).

---

## 1. Folder Architecture

```
SR-TTS_code/
├── data/                      # Data pipeline: preprocessing + merging + loading
│   ├── raw_data/              # Raw inputs (origin_TDMS/ raw signals, label/ coordinate labels)
│   ├── processed_data/        # Processed outputs (segmented_output/ segmented CSVs, training_data*.h5)
│   ├── config_data.yaml       # Data pipeline configuration
│   ├── tdms_reader.py         # Read and segment TDMS files
│   ├── data_preprocess.py     # Preprocessing main entry
│   ├── data_merger.py         # Merge CSVs into an HDF5 dataset
│   └── loader.py              # PyTorch data loaders
│
├── main/                      # Training / evaluation / incremental learning entry points
│   ├── config.yaml            # Main config (model, training, incremental learning, paths)
│   ├── train.py               # Flat-surface classification training
│   ├── qianyi_learn.py        # Curved-surface incremental learning
│   ├── test.py                # Single test-set evaluation
│   └── test_weitiao.py        # Batch test-set evaluation
│
├── src/                       # Model and loss functions
│   ├── model.py               # Model definitions (classification / regression)
│   └── Loss.py                # Loss functions
│
├── utils/                     # Visualization utilities
│   ├── visualization.py       # Training curves / confusion matrix / ROC
│   └── feature_visualizer.py  # Feature extraction + t-SNE visualization
│
└── output/                    # Outputs (model weights, training history, evaluation results)
```

---

## 2. Script Responsibilities

### 2.1 `data/` — Data Pipeline

| Script | Responsibility |
|--------|----------------|
| `tdms_reader.py` | Reads raw NI `.tdms` files; applies low-pass filtering and DC removal (de-mean) to the 32-channel signals; selects valid channels via 1 Hz energy ratio; segments the continuous signal into 1 s windows based on the reference channel's negative/positive peaks, saving each segment as CSV. |
| `data_preprocess.py` | Preprocessing main entry: batch-processes all TDMS files under `origin_TDMS/` (calling `process_single_tdms`), then invokes `SmartDataMerger` to merge the segmented CSVs into a single HDF5 dataset. |
| `data_merger.py` | `SmartDataMerger`: collects all segmented CSVs → parallel loading + anti-aliasing downsampling → parses row/col labels (supports `origin` for filename-based parsing, `classification`/`regression` for reading from label CSVs) → writes to HDF5; supports batched processing, checkpoint/resume, memory adaptation, and checkpoints. |
| `loader.py` | PyTorch data-access layer: `HighPerformanceSensorDataset` (full-memory / on-demand loading modes); `collate_padded_fast` maps each sample's valid channels to the global 32-channel layout and generates a mask; provides train/val/test splits, curved-surface finetune loaders, flat-surface full loaders, and a streaming loader. |
| `config_data.yaml` | Data pipeline configuration: paths, filter parameters, segmentation window, coordinate step, merge mode, sampling rate, etc. |

### 2.2 `src/` — Model & Loss

| Script | Responsibility |
|--------|----------------|
| `model.py` | Defines the models: `PositionalEncoding` (sinusoidal position encoding), `ChannelAttention` (channel attention), `SuperResolutionSensorModel` (classification model: channel embedding + position encoding + channel attention + Transformer + 1D-CNN + classification heads, supporting independent/joint classification), and `SuperResolutionSensorModelRegression` (regression model that outputs continuous `(x, y)` coordinates). |
| `Loss.py` | Defines the losses: `MultiTaskLoss` (multi-task classification loss, supporting independent/joint modes) and `RegressionLoss` (MSE regression loss). |

### 2.3 `main/` — Training & Evaluation Entry Points

| Script | Responsibility |
|--------|----------------|
| `train.py` | Flat-surface classification training: loads config → builds data loaders → initializes model → trains. Includes learning-rate warmup scheduling, label smoothing, class weights, gradient clipping, early stopping, and training-curve plotting. |
| `qianyi_learn.py` | Curved-surface incremental learning: based on flat-surface pretrained weights, jointly applies EWC (Elastic Weight Consolidation) + Knowledge Distillation + Memory Replay to suppress catastrophic forgetting, and supports staged unfreezing and layer-wise learning rates. |
| `test.py` | Single test-set evaluation: loads the model → predicts → computes row/col/overall accuracy → outputs results CSV, confusion matrix, and ROC curves. |
| `test_weitiao.py` | Batch test-set evaluation: scans all HDF5 files under `test_h5_dir`, evaluates each one, and generates a summary report. |
| `config.yaml` | Main configuration: paths, model structure, training hyperparameters, incremental-learning hyperparameters, hardware optimization, etc. |

### 2.4 `utils/` — Visualization

| Script | Responsibility |
|--------|----------------|
| `visualization.py` | Plots training curves (loss/acc), confusion matrices (seaborn heatmap), and ROC curves (multi-class + AUC). |
| `feature_visualizer.py` | Extracts features from each model layer (projection / Transformer / CNN / classification heads) via forward hooks and visualizes clustering with t-SNE dimensionality reduction. |

---

## 3. Quick Start

### 3.1 Data Preprocessing (TDMS → HDF5)

1. Place the raw `.tdms` files into `data/raw_data/origin_TDMS/`;
2. Place the coordinate label CSVs (containing `x_mm`, `y_mm` columns, filenames matching the TDMS files) into `data/raw_data/label/`;
3. Adjust `data/config_data.yaml` as needed;
4. Run:

```bash
python data/data_preprocess.py
```

Outputs: `data/processed_data/segmented_output/` (CSV segments) and `data/processed_data/training_data.h5` (merged dataset).

### 3.2 Flat-Surface Training

Make sure `raw_h5_path` in `main/config.yaml` points to the preprocessed output, then:

```bash
python main/train.py
```

Outputs: `output/results_temp/best_model.pth` and the training-history CSV.

### 3.3 Curved-Surface Incremental Learning

Configure `pretrain_weights_path` (flat-surface weights) and `curve_h5_path` (curved-surface data), then:

```bash
python main/qianyi_learn.py
```

Outputs: `output/results_temp/best_incremental_model.pth`.

### 3.4 Evaluation

```bash
python main/test.py            # Single test-set evaluation
python main/test_weitiao.py    # Batch test-set evaluation
```

Outputs: per-dataset `results.csv`, confusion matrices, ROC curves, and a summary report.

### 3.5 Feature Visualization (t-SNE)

```bash
python utils/feature_visualizer.py
```
