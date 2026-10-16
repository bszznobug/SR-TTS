import os
import yaml
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
from data.loader import create_high_performance_test_loaders
from torch.utils.data import DataLoader
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
        model.load_state_dict(torch.load(config['model_path']))
    else:
        model.load_state_dict(torch.load(config['model_path'], map_location=torch.device('cpu')))

    model = model.to(config['device'])
    model.eval()
    return model


def evaluate_model(model, data_loader, dataset_name):
    """评估模型并保存结果（使用配置文件参数）"""
    # 创建输出目录
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'confusion_matrices'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'roc_curves'), exist_ok=True)

    results = []
    all_row_true = []
    all_row_pred = []
    all_col_true = []
    all_col_pred = []
    all_row_probs = []
    all_col_probs = []

    with torch.no_grad():
        for signals, masks, labels, metadata in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
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
                    'row_probs': row_probs[i].tolist(),  # 保存所有类别的概率
                    'col_probs': col_probs[i].tolist(),  # 保存所有类别的概率
                    'correct': (row_preds[i] == row_labels[i]) and (col_preds[i] == col_labels[i])
                })

            # 收集混淆矩阵和ROC数据
            all_row_true.extend(row_labels)
            all_row_pred.extend(row_preds)
            all_col_true.extend(col_labels)
            all_col_pred.extend(col_preds)
            all_row_probs.extend(row_probs)
            all_col_probs.extend(col_probs)

    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f'{dataset_name}_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # 计算准确率
    row_accuracy = np.mean(np.array(all_row_true) == np.array(all_row_pred))
    col_accuracy = np.mean(np.array(all_col_true) == np.array(all_col_pred))
    overall_accuracy = results_df['correct'].mean()

    print(f"\n{dataset_name} Evaluation Results:")
    print(f"Row Accuracy: {row_accuracy:.4f}")
    print(f"Column Accuracy: {col_accuracy:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # 绘制混淆矩阵（使用配置中的最大显示类别数）
    max_classes = config['max_confusion_classes']
    row_classes = [str(i) for i in range(min(model.num_rows, max_classes))]
    col_classes = [str(i) for i in range(min(model.num_cols, max_classes))]

    # 行混淆矩阵
    row_cm_path = os.path.join(output_dir, 'confusion_matrices', f'{dataset_name}_row_confusion.png')
    plot_confusion_matrix(all_row_true, all_row_pred, row_classes,
                          f'{dataset_name} Row Confusion Matrix (Acc: {row_accuracy:.2f})',
                          row_cm_path)

    # 列混淆矩阵
    col_cm_path = os.path.join(output_dir, 'confusion_matrices', f'{dataset_name}_col_confusion.png')
    plot_confusion_matrix(all_col_true, all_col_pred, col_classes,
                          f'{dataset_name} Column Confusion Matrix (Acc: {col_accuracy:.2f})',
                          col_cm_path)

    # 绘制 ROC 曲线（根据实际检测到的类别绘制）
    roc_dir = os.path.join(output_dir, 'roc_curves')
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

    return results_df


def main():
    # 设置随机种子（使用配置文件参数）
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # 加载模型
    print("Loading model...")
    model = load_model()
    print(f"Model loaded successfully from {config['model_path']}")
    H5_PATH = config["test_h5_path"]

    # 创建数据加载器（使用配置文件参数）
    print("\nCreating data loaders...")

    # 根据内存情况选择加载策略
    try:
        # 尝试全内存加载（参数从配置读取）
        test_loader = create_high_performance_test_loaders(
            H5_PATH,
            batch_size=config["batch_size"],
            # num_workers=config["num_workers"],
            full_load=config["full_load"]
        )
        print("✅ 使用全内存加载模式")
    except Exception as e:
        print(f"全内存加载失败: {e}")
        print("🔄 回退到动态加载模式（使用配置的回退参数）")
        test_loader = create_high_performance_test_loaders(
            H5_PATH,
            batch_size=config["fallback_batch_size"],
            # num_workers=config["fallback_num_workers"],
            full_load=False
        )

    # 评估测试集
    print("\nEvaluating test set...")
    test_results = evaluate_model(model, test_loader, 'test')

    print(f"\nEvaluation completed! All results saved to {config['output_dir']}")


if __name__ == "__main__":
    main()