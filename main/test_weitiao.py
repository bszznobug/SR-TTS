import os
import glob
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
from data.loader import create_high_performance_test_loaders
from src.model import SuperResolutionSensorModel
from sklearn.preprocessing import label_binarize
from utils.visualization import plot_confusion_matrix
from utils.visualization import plot_roc_curve


# 读取yaml配置
def load_config(config_path="config.yaml"):
    """加载评估配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 自动适配设备（优先使用GPU）
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config


# 加载配置
config = load_config()


def load_model():
    """加载训练好的模型（使用配置文件参数）"""
    model = SuperResolutionSensorModel(
        num_rows=config['num_rows'],
        num_cols=config['num_cols'],
        joint_classification=config['joint_mode']
    )

    # 处理CPU/GPU兼容问题
    if config['device'] == 'cuda':
        model.load_state_dict(torch.load(config['finetune_path']))
    else:
        model.load_state_dict(torch.load(config['finetune_path'], map_location=torch.device('cpu')))

    print(f"模型路径: {config['finetune_path']}")
    model = model.to(config['device'])
    model.eval()
    return model


def find_test_h5_files(config):
    """
    获取所有待测试的 HDF5 文件列表。
    优先使用 test_h5_dir（目录），其次 test_h5_path（单文件），
    支持 test_h5_path 为目录的兼容情况。
    """
    # 方式1：配置了 test_h5_dir，扫描目录下所有 .h5/.hdf5 文件
    test_dir = config.get('test_h5_dir', None)
    if test_dir and os.path.isdir(test_dir):
        files = []
        for ext in ['*.h5', '*.hdf5']:
            files.extend(glob.glob(os.path.join(test_dir, '**', ext), recursive=True))
        files = sorted(set(files))
        if files:
            print(f"从 test_h5_dir 扫描到 {len(files)} 个测试集文件")
            return files

    # 方式2：单文件 test_h5_path
    test_path = config.get('test_h5_path', None)
    if test_path:
        # 兼容：test_h5_path 可能是一个目录
        if os.path.isdir(test_path):
            files = []
            for ext in ['*.h5', '*.hdf5']:
                files.extend(glob.glob(os.path.join(test_path, '**', ext), recursive=True))
            files = sorted(set(files))
            print(f"test_h5_path 为目录，扫描到 {len(files)} 个测试集文件")
            return files
        elif os.path.isfile(test_path):
            print(f"使用单文件测试集: {os.path.basename(test_path)}")
            return [test_path]
        else:
            raise FileNotFoundError(f"测试集路径不存在: {test_path}")

    raise FileNotFoundError("未配置 test_h5_dir 或 test_h5_path，无法找到测试集文件")


def evaluate_model(model, data_loader, dataset_name, output_root):
    """
    评估模型并保存结果。

    参数:
      model:        已加载的模型
      data_loader:  单个测试集的 DataLoader
      dataset_name: 数据集名称（用于文件夹命名和输出前缀）
      output_root:  总输出根目录，该数据集的结果放在 output_root/dataset_name/ 下
    """
    # ---- 为该数据集创建独立的输出目录 ----
    dataset_output_dir = os.path.join(output_root, dataset_name)
    cm_dir = os.path.join(dataset_output_dir, 'confusion_matrices')
    roc_dir = os.path.join(dataset_output_dir, 'roc_curves')
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)

    results = []
    all_row_true = []
    all_row_pred = []
    all_col_true = []
    all_col_pred = []
    all_row_probs = []
    all_col_probs = []

    with torch.no_grad():
        for signals, masks, labels, metadata in tqdm(data_loader, desc=f"评估 {dataset_name}"):
            signals = signals.to(config['device'])
            masks = masks.to(config['device'])
            row_labels = labels[:, 0].cpu().numpy()
            col_labels = labels[:, 1].cpu().numpy()

            # 预测
            outputs = model(signals, masks)

            if model.joint_classification:
                # 联合分类模式
                row_probs = torch.softmax(outputs.sum(dim=2), dim=1)
                col_probs = torch.softmax(outputs.sum(dim=1), dim=1)
                _, row_preds = torch.max(row_probs, 1)
                _, col_preds = torch.max(col_probs, 1)
            else:
                # 独立分类模式
                row_logits, col_logits = outputs
                row_probs = torch.softmax(row_logits, 1)
                col_probs = torch.softmax(col_logits, 1)
                _, row_preds = torch.max(row_probs, 1)
                _, col_preds = torch.max(col_probs, 1)

            row_preds = row_preds.cpu().numpy()
            col_preds = col_preds.cpu().numpy()
            row_probs = row_probs.cpu().numpy()
            col_probs = col_probs.cpu().numpy()

            # 收集结果
            for i in range(len(labels)):
                results.append({
                    'sample_id': metadata[i]['signal_name'],
                    'true_row': row_labels[i],
                    'true_col': col_labels[i],
                    'pred_row': row_preds[i],
                    'pred_col': col_preds[i],
                    'row_probs': row_probs[i].tolist(),
                    'col_probs': col_probs[i].tolist(),
                    'correct': (row_preds[i] == row_labels[i]) and (col_preds[i] == col_labels[i])
                })

            all_row_true.extend(row_labels)
            all_row_pred.extend(row_preds)
            all_col_true.extend(col_labels)
            all_col_pred.extend(col_preds)
            all_row_probs.extend(row_probs)
            all_col_probs.extend(col_probs)

    # ---- 保存 CSV 结果 ----
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(dataset_output_dir, f'{dataset_name}_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  📄 结果已保存: {csv_path}")

    # ---- 计算准确率 ----
    row_accuracy = np.mean(np.array(all_row_true) == np.array(all_row_pred))
    col_accuracy = np.mean(np.array(all_col_true) == np.array(all_col_pred))
    overall_accuracy = results_df['correct'].mean()

    print(f"  📊 {dataset_name} — "
          f"行准确率: {row_accuracy:.4f}, "
          f"列准确率: {col_accuracy:.4f}, "
          f"整体准确率: {overall_accuracy:.4f}")

    # ---- 混淆矩阵 ----
    max_classes = config.get('max_confusion_classes', 11)
    num_rows = model.num_rows
    num_cols = model.num_cols
    row_classes = [str(i) for i in range(min(num_rows, max_classes))]
    col_classes = [str(i) for i in range(min(num_cols, max_classes))]

    # 行混淆矩阵
    row_cm_path = os.path.join(cm_dir, f'{dataset_name}_row_confusion.png')
    plot_confusion_matrix(all_row_true, all_row_pred, row_classes,
                          f'{dataset_name} Row Confusion (Acc: {row_accuracy:.2f})',
                          row_cm_path)

    # 列混淆矩阵
    col_cm_path = os.path.join(cm_dir, f'{dataset_name}_col_confusion.png')
    plot_confusion_matrix(all_col_true, all_col_pred, col_classes,
                          f'{dataset_name} Column Confusion (Acc: {col_accuracy:.2f})',
                          col_cm_path)

    # ---- ROC 曲线（根据实际检测到的类别绘制） ----
    try:
        unique_rows = sorted(set(all_row_true))
        unique_cols = sorted(set(all_col_true))
        all_row_probs_arr = np.array(all_row_probs)  # [N, num_rows] 模型全类别概率
        all_col_probs_arr = np.array(all_col_probs)  # [N, num_cols]

        # 行 ROC：按实际出现类别筛选概率列，列顺序与 unique_rows 一致
        if len(unique_rows) > 1:
            if all_row_probs_arr.shape[1] > max(unique_rows):
                row_probs_sub = all_row_probs_arr[:, unique_rows]
                plot_roc_curve(np.array(all_row_true), row_probs_sub, unique_rows,
                               f'{dataset_name} Row ROC',
                               os.path.join(roc_dir, f'{dataset_name}_row_roc.png'))
            else:
                print(f"  ⚠️ 行 ROC 跳过：概率维度({all_row_probs_arr.shape[1]})小于最大行类别({max(unique_rows)})")
        else:
            print(f"  ⚠️ 行 ROC 跳过：测试集行类别少于2个")

        # 列 ROC
        if len(unique_cols) > 1:
            if all_col_probs_arr.shape[1] > max(unique_cols):
                col_probs_sub = all_col_probs_arr[:, unique_cols]
                plot_roc_curve(np.array(all_col_true), col_probs_sub, unique_cols,
                               f'{dataset_name} Column ROC',
                               os.path.join(roc_dir, f'{dataset_name}_col_roc.png'))
            else:
                print(f"  ⚠️ 列 ROC 跳过：概率维度({all_col_probs_arr.shape[1]})小于最大列类别({max(unique_cols)})")
        else:
            print(f"  ⚠️ 列 ROC 跳过：测试集列类别少于2个")
    except Exception as e:
        print(f"  ⚠️ ROC 曲线绘制失败: {e}")

    return results_df, {
        'row_accuracy': row_accuracy,
        'col_accuracy': col_accuracy,
        'overall_accuracy': overall_accuracy,
        'total_samples': len(results_df),
    }


def create_loader_for_file(h5_path):
    """为单个 HDF5 文件创建 DataLoader，失败时自动回退"""
    try:
        loader = create_high_performance_test_loaders(
            h5_path,
            batch_size=config.get("batch_size", 128),
            full_load=config.get("full_load", False)
        )
        print(f"    ✅ 全内存/标准加载成功")
        return loader
    except Exception as e:
        print(f"    ⚠️ 标准加载失败 ({e})，回退到动态加载")
        loader = create_high_performance_test_loaders(
            h5_path,
            batch_size=config.get("fallback_batch_size", 128),
            full_load=False
        )
        return loader


def main():
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # ---------- 1. 加载模型 ----------
    print("=" * 60)
    print("  批量测试集评估")
    print("=" * 60)
    print("\n🔧 加载模型...")
    model = load_model()
    print(f"  ✅ 模型加载成功")

    # ---------- 2. 发现所有测试集文件 ----------
    print("\n📁 扫描测试集文件...")
    test_files = find_test_h5_files(config)
    print(f"  共 {len(test_files)} 个测试集:")
    for f in test_files:
        print(f"    - {os.path.basename(f)}")

    # ---------- 3. 逐个测试集评估 ----------
    output_root = config['output_dir']
    os.makedirs(output_root, exist_ok=True)

    all_summaries = {}

    for i, h5_path in enumerate(test_files):
        dataset_stem = os.path.splitext(os.path.basename(h5_path))[0]
        print(f"\n{'=' * 60}")
        print(f"  [{i+1}/{len(test_files)}] 评估: {dataset_stem}")
        print(f"{'=' * 60}")

        # 创建 DataLoader
        print(f"  📦 加载数据: {h5_path}")
        test_loader = create_loader_for_file(h5_path)

        # 评估
        print(f"  🚀 开始评估...")
        _, summary = evaluate_model(model, test_loader, dataset_stem, output_root)
        all_summaries[dataset_stem] = summary

    # ---------- 4. 汇总报告 ----------
    print(f"\n{'=' * 60}")
    print(f"  📊 批量评估汇总")
    print(f"{'=' * 60}")
    print(f"  {'数据集':<40} {'样本数':>8} {'行Acc':>8} {'列Acc':>8} {'整体Acc':>8}")
    print(f"  {'-' * 72}")
    for name, s in all_summaries.items():
        print(f"  {name:<40} {s['total_samples']:>8} "
              f"{s['row_accuracy']:>8.4f} {s['col_accuracy']:>8.4f} {s['overall_accuracy']:>8.4f}")

    # 保存汇总 CSV
    summary_df = pd.DataFrame(all_summaries).T
    summary_df.index.name = 'dataset'
    summary_path = os.path.join(output_root, '_batch_evaluation_summary.csv')
    summary_df.to_csv(summary_path)
    print(f"\n📝 汇总报告已保存: {summary_path}")

    print(f"\n{'=' * 60}")
    print(f"  🎉 全部评估完成！")
    print(f"  结果根目录: {output_root}")
    print(f"  各数据集结果在对应子文件夹中")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
