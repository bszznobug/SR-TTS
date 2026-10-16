"""
可视化工具模块
用于绘制训练曲线、混淆矩阵、ROC曲线等
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc
import yaml
import torch

# 项目根目录（utils 的上一级），用于定位 main/config.yaml
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 读取yaml配置
def load_config(config_path=os.path.join(base_dir, "main", "config.yaml")):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 自动适配num_workers和device
    config['num_workers'] = min(8, os.cpu_count() // 2) if os.cpu_count() else 0
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config


# 加载配置
config = load_config()

# 默认训练曲线输出文件名（未指定时保存到 result_save_dir，与训练历史 CSV 同目录）
DEFAULT_CURVE_FILENAME = "training_curves.png"

def plot_training_curves(history_df, save_path=None, csv_path=None):
    """绘制训练和验证的 loss 和 accuracy 曲线

    :param history_df: 训练历史 DataFrame（需包含 epoch/train_loss/val_loss/train_acc/val_acc 列）
    :param save_path:  图片保存路径。若为 None，则自动推导：
                       - 提供 csv_path 时，保存到与 CSV 同目录、同名的 .png；
                       - 未提供时，默认保存到 config 的 result_save_dir 下 training_curves.png
    :param csv_path:  训练历史 CSV 文件路径，用于推导默认保存目录
    :return: 实际保存的图片路径
    """
    if save_path is None:
        if csv_path:
            # 与 CSV 同目录、同名（后缀改为 .png）
            save_path = os.path.splitext(os.path.abspath(csv_path))[0] + ".png"
        else:
            save_dir = config.get("result_save_dir", os.getcwd())
            save_path = os.path.join(save_dir, DEFAULT_CURVE_FILENAME)

    # 确保输出目录存在
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')
    print(f"训练曲线已保存至: {save_path}")
    return save_path


def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    """绘制混淆矩阵并保存"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)
    plt.close()



def plot_roc_curve(y_true, y_score, n_classes, title, filename):
    """绘制ROC曲线并保存

    :param y_true: 真实标签，一维数组
    :param y_score: 每个类别的预测概率，二维数组 [N, C]，列顺序需与 classes 一致
    :param n_classes: 类别数(int，按 0..n-1) 或实际检测到的类别列表(list/tuple/ndarray)
    :param title: 图标题
    :param filename: 保存路径
    """
    # 兼容两种调用：int 表示类别数 0..n-1；list 表示实际检测到的类别
    if isinstance(n_classes, (list, tuple, np.ndarray)):
        classes = list(n_classes)
    else:
        classes = list(range(int(n_classes)))
    n_class = len(classes)

    # 按实际类别二值化标签（列顺序与 classes 一致）
    y_true_bin = label_binarize(y_true, classes=classes)

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, cls in enumerate(classes):
        # 该类别在样本中只有单一标签时(全正或全负) roc_curve 无法计算，跳过
        if len(np.unique(y_true_bin[:, i])) < 2:
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    if not roc_auc:
        plt.close('all')
        print(f"⚠️ 没有可绘制的ROC类别，跳过 {filename}")
        return

    # 绘制所有类别的ROC曲线
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_class))
    for i, cls in enumerate(classes):
        if i not in roc_auc:
            continue
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label='Class {0} (AUC = {1:0.2f})'.format(cls, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    # 优化图例显示
    if n_class > 15:
        # 方案2：将图例分成两列显示
        plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0), ncol=3, fontsize=6)
        plt.subplots_adjust(right=0.6)
    else:
        plt.legend(loc='lower right', fontsize=6)
    plt.savefig(filename)
    plt.close()
    print(f"ROC曲线已保存: {filename}")


if __name__ == "__main__":
    import sys
    import glob

    # 用法1：python visualization.py <csv路径>  直接指定训练历史 CSV 重新绘制
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.isfile(csv_path):
            print(f"❌ 文件不存在: {csv_path}")
            sys.exit(1)
        try:
            df = pd.read_csv(csv_path)
            plot_training_curves(df, csv_path=csv_path)
        except Exception as e:
            print(f"❌ 绘制失败: {e}")
            sys.exit(1)
        sys.exit(0)

    # 用法2：python visualization.py  自动检测训练历史 CSV 并绘制
    save_dir = config.get("result_save_dir", os.getcwd())
    csv_files = []

    # 优先检测 training_history_with_row_col.csv，其次扫描 result_save_dir 下所有 CSV
    prefer = os.path.join(save_dir, "training_history_with_row_col.csv")
    if os.path.isfile(prefer):
        csv_files = [prefer]
    elif os.path.isdir(save_dir):
        csv_files = sorted(glob.glob(os.path.join(save_dir, "*.csv")))

    if not csv_files:
        print(f"❌ 在 {save_dir} 下未找到训练历史 CSV 文件，请先运行训练脚本。")
        sys.exit(1)

    for csv_path in csv_files:
        print(f"📊 正在根据 {os.path.basename(csv_path)} 重新绘制训练曲线 ...")
        try:
            df = pd.read_csv(csv_path)
            out = plot_training_curves(df, csv_path=csv_path)
            print(f"   ✅ 完成: {out}")
        except Exception as e:
            print(f"   ❌ 绘制 {csv_path} 失败: {e}")
