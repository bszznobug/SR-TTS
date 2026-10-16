import os
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import SuperResolutionSensorModel
import seaborn as sns
import yaml
from data.loader import OptimizedSensorDataset, collate_padded_fast, worker_init_fn

# 读取yaml配置
def load_config(config_path=r"D:\super\main\config.yaml"):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 自动适配num_workers和device
    config['num_workers'] = min(8, os.cpu_count() // 2) if os.cpu_count() else 0
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config


# 加载配置
config = load_config()

class FeatureExtractor:
    """用于提取模型各层特征的辅助类"""

    def __init__(self, model_path, num_rows=501, num_cols=501, joint_classification=False, device='cuda'):
        """加载模型并设置钩子"""
        self.device = device
        self.joint_classification = joint_classification
        self.model = SuperResolutionSensorModel(
            num_rows=num_rows,
            num_cols=num_cols,
            joint_classification=joint_classification
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.model.eval()

        # 存储各层特征的字典
        self.features = {
            'input': [],
            'projected': [],
            'transformer': [],
            'cnn': [],
            'fc_row': [],  # 单独存储row的FC层特征
            'fc_col': [],  # 单独存储col的FC层特征
            'labels': []
        }

        # 注册钩子
        self._register_hooks()

    def _register_hooks(self):
        """注册钩子以捕获中间层输出"""
        # 先清除旧钩子
        for h in list(self.model._forward_hooks.values()):
            self.model._forward_hooks.clear()

        # 投影层后的钩子
        def projected_hook(module, input, output):
            self.features['projected'].append(output.detach().cpu().numpy())

        # Transformer后的钩子
        def transformer_hook(module, input, output):
            self.features['transformer'].append(output.detach().cpu().numpy())

        # CNN后的钩子
        def cnn_hook(module, input, output):
            self.features['cnn'].append(output.detach().cpu().numpy())

        # 注册通用钩子
        self.model.projection.register_forward_hook(
            lambda m, i, o: projected_hook(m, i, o))
        self.model.temporal_encoder.register_forward_hook(
            lambda m, i, o: transformer_hook(m, i, o))
        self.model.spatial_cnn.register_forward_hook(
            lambda m, i, o: cnn_hook(m, i, o))

        if self.joint_classification:
            # 联合分类模式下的FC层钩子
            def joint_fc_hook(module, input, output):
                self.features['fc_row'].append(output.detach().cpu().numpy())
                self.features['fc_col'].append(output.detach().cpu().numpy())

            self.model.joint_head.register_forward_hook(
                lambda m, i, o: joint_fc_hook(m, i, o))
        else:
            # 独立分类模式下的FC层钩子
            def row_fc_hook(module, input, output):
                self.features['fc_row'].append(output.detach().cpu().numpy())

            def col_fc_hook(module, input, output):
                self.features['fc_col'].append(output.detach().cpu().numpy())

            self.model.row_head.register_forward_hook(
                lambda m, i, o: row_fc_hook(m, i, o))
            self.model.col_head.register_forward_hook(
                lambda m, i, o: col_fc_hook(m, i, o))

    def extract_features(self, data_loader):
        """从数据加载器中提取特征"""
        # 清空
        for k in self.features:
            self.features[k] = []

        with torch.no_grad():
            for signals, masks, labels, _ in tqdm(data_loader, desc="Extracting features"):
                signals = signals.to(self.device)
                masks = masks.to(self.device)

                # 存储输入特征
                self.features['input'].append(signals.detach().cpu().numpy())

                # 前向传播(会触发钩子)
                _ = self.model(signals, masks)

                # 存储标签
                self.features['labels'].append(labels.numpy())

        # 合并所有batch的特征，并确保所有层样本数一致
        min_samples = min(
            len(np.concatenate(self.features['input'], axis=0)) if len(self.features['input']) > 0 else 0,
            len(np.concatenate(self.features['projected'], axis=0)) if len(self.features['projected']) > 0 else 0,
            len(np.concatenate(self.features['transformer'], axis=0)) if len(self.features['transformer']) > 0 else 0,
            len(np.concatenate(self.features['cnn'], axis=0)) if len(self.features['cnn']) > 0 else 0,
            len(np.concatenate(self.features['fc_row'], axis=0)) if len(self.features['fc_row']) > 0 else 0,
            len(np.concatenate(self.features['fc_col'], axis=0)) if len(self.features['fc_col']) > 0 else 0,
            len(np.concatenate(self.features['labels'], axis=0)) if len(self.features['labels']) > 0 else 0
        )

        for key in self.features:
            if len(self.features[key]) > 0:
                self.features[key] = np.concatenate(self.features[key], axis=0)[:min_samples]

        return self.features


def apply_tsne(features, n_components=2, perplexity=30, random_state=42):
    """应用t-SNE降维"""
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                n_jobs=-1)
    return tsne.fit_transform(features)


def plot_tsne_variations(tsne_results, labels, layer_name, output_dir):
    """绘制三种不同标签方式的 t-SNE 结果并以连续 colorbar 展示（低饱和度，兼容 Matplotlib ≥3.7）"""
    import matplotlib.colors as mcolors
    import matplotlib as mpl

    os.makedirs(output_dir, exist_ok=True)

    row_labels = labels[:, 0]
    col_labels = labels[:, 1]
    combined_labels = row_labels * (labels[:, 1].max() + 1) + col_labels

    plot_types = [
        ('color_only', 'Combined label (continuous)', combined_labels),
        ('row', 'Row label (continuous)', row_labels),
        ('col', 'Col label (continuous)', col_labels)
    ]

    n_points = len(tsne_results)
    s = max(1, min(30, 100000 / n_points))

    # 低饱和度色图：将 Viridis 的饱和度降为 60 %
    base_cmap = mpl.colormaps['gist_rainbow']          # 兼容 3.7+
    desat_cmap = mcolors.LinearSegmentedColormap.from_list(
        'desat_gist_rainbow',
        base_cmap(np.linspace(0, 1, 256)) * np.array([0.9, 0.9, 0.9, 1])
    )

    for plot_type, title, c_data in plot_types:
        fig, ax = plt.subplots(figsize=(8, 8))

        sc = ax.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=c_data,
            cmap=desat_cmap,
            s=s,
            alpha=0.8
        )

        ax.set_title(f'{title} - {layer_name} Layer')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_aspect('equal', adjustable='box')

        plt.colorbar(sc, ax=ax, shrink=0.8, label='Label value')

        plot_path = os.path.join(output_dir, f'tsne_{layer_name}_{plot_type}.png')
        fig.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close(fig)

    # 保存数据
    data_path = os.path.join(output_dir, f'tsne_{layer_name}_data.csv')
    pd.DataFrame({
        'tsne1': tsne_results[:, 0],
        'tsne2': tsne_results[:, 1],
        'row_label': row_labels,
        'col_label': col_labels,
        'combined_label': combined_labels
    }).to_csv(data_path, index=False)

def visualize_features(model_path, h5_path, output_dir, batch_size=32):
    """主函数：可视化模型各层特征"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型参数(需要与训练时一致)
    num_rows = 501
    num_cols = 501
    joint_classification = False  # 必须与训练时设置相同

    # 创建数据加载器
    dataset = OptimizedSensorDataset(h5_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_padded_fast
    )

    # 初始化特征提取器
    extractor = FeatureExtractor(model_path, num_rows, num_cols, joint_classification, device=device)
    features = extractor.extract_features(loader)

    # 对每层特征应用t-SNE并绘制图像
    for layer in ['input', 'projected', 'transformer', 'cnn', 'fc_row', 'fc_col']:
        if layer not in features or len(features[layer]) == 0:
            continue

        print(f"\nProcessing {layer} layer features...")

        # 展平特征(除fc层外)
        if not layer.startswith('fc_'):
            flat_features = features[layer].reshape(features[layer].shape[0], -1)
        else:
            flat_features = features[layer]

        # 应用t-SNE
        tsne_results = apply_tsne(flat_features)

        # 绘制可视化结果
        plot_tsne_variations(tsne_results, features['labels'], layer, output_dir)


if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = config["model_path"]  # 训练好的模型路径
    H5_PATH = config["raw_h5_path"]  # 测试集HDF5文件
    OUTPUT_DIR = os.path.join(config["result_save_dir"], 'feature_visualization_test')  # 输出目录

    print("Starting feature visualization...")
    visualize_features(MODEL_PATH, H5_PATH, OUTPUT_DIR)
    print("\nFeature visualization completed!")