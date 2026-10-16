import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.multiprocessing as mp
import yaml

# 解决pytorch多进程加载数据时的内存共享冲突
mp.set_sharing_strategy('file_system')

base_dir = os.path.dirname(os.path.abspath(__file__))

# 读取yaml配置
def load_config(config_path= os.path.join(base_dir, "..", "main", "config.yaml")):
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 自动适配num_workers和device
    config['num_workers'] = min(8, os.cpu_count() // 2) if os.cpu_count() else 0
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config


# 加载配置
config = load_config()

class OptimizedSensorDataset(Dataset):
    def __init__(self, h5_path, transform=None, cache_size=1000):
        """
        Args:
            h5_path: Path to the HDF5 file
            transform: Optional transform to be applied on a sample
            cache_size: Number of samples to keep in memory cache
        """
        self.h5_path = h5_path
        self.transform = transform
        self.cache_size = cache_size
        self._cache = {}

        # Get all sample keys from the HDF5 file
        with h5py.File(h5_path, 'r') as hf:
            self._sample_keys = [key for key in hf.keys() if key.startswith('sample_')]

    def __len__(self):
        return len(self._sample_keys)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        key = self._sample_keys[idx]
        with h5py.File(self.h5_path, 'r') as hf:
            group = hf[key]
            signal = torch.from_numpy(group['signal'][:]).float()

            # Ensure label extraction matches your HDF5 structure
            row = group.attrs.get("row", group.attrs.get("Data-Row", 0))  # Fallback to 0 if not found
            col = group.attrs.get("col", group.attrs.get("SETxCol", 0))  # Fallback to 0 if not found
            label = torch.tensor([row, col], dtype=torch.long)

            # Extract signal number from key (e.g., "signal_001" -> 1)
            signal_num = int(key.split('_')[-1]) if 'signal_' in key else 0

            metadata = {
                "file_name": group.attrs["file_name"],
                "group_name": group.attrs["group_name"],
                "signal_name": group.attrs["signal_name"],
                "signal_number": signal_num,
                "channel_names": group.attrs["channel_names"],
                "row": row,
                "col": col
            }

            # Validate channel names
            for name in metadata["channel_names"]:
                if not name.startswith('Pos_'):
                    raise ValueError(f"Invalid channel name format: {name}")

        if self.transform:
            signal = self.transform(signal)

        if len(self._cache) < self.cache_size:
            self._cache[idx] = (signal, label, metadata)
        return signal, label, metadata

class HighPerformanceSensorDataset(Dataset):
    def __init__(self, h5_path, transform=None, full_load_to_memory=True):
        """
        高性能数据加载器 - 全内存加载版本
        """
        self.h5_path = h5_path
        self.transform = transform
        self.full_load_to_memory = full_load_to_memory # 选择不同加载模式

        if full_load_to_memory:
            print("正在全量加载数据到内存...")
            self._load_all_to_memory()
        else:
            self._load_metadata_only()

    def _load_all_to_memory(self):
        """一次性加载所有数据到内存"""
        self.signals = []
        self.labels = []
        self.metadata_list = []

        with h5py.File(self.h5_path, 'r') as hf:
            # 筛选所有以"sample_"开头的样本键，并按数字序号排序（确保加载顺序一致）
            sample_keys = sorted([k for k in hf.keys() if k.startswith('sample_')],
                                 key=lambda x: int(x.split('_')[1]))
            # 遍历所有样本，带进度条显示
            for key in tqdm(sample_keys, desc="全量加载数据"):
                group = hf[key]

                # 加载信号数据，HDF5数组→numpy数组→转换为float32→torch张量
                signal = torch.from_numpy(group['signal'][:].astype(np.float32))
                self.signals.append(signal)

                # 加载标签，从HDF5组的属性中获取行、列坐标，转换为long型张量（分类任务常用）
                row = group.attrs.get("row", 0)
                col = group.attrs.get("col", 0)
                label = torch.tensor([row, col], dtype=torch.long)
                self.labels.append(label)

                # 加载元数据，存储信号相关的辅助信息（通道名、文件名等）
                metadata = {
                    "channel_names": group.attrs.get("channel_names", []),  # 信号通道名称列表
                    "file_name": group.attrs.get("file_name", ""),          # 原始文件名
                    "signal_name": group.attrs.get("signal_name", ""),      # 信号名称
                    "surface_id": group.attrs.get("surface_id", -1),        # 曲面类型ID (0=平面, 1-9=曲面, -1=未标注)
                }
                self.metadata_list.append(metadata)

        print(f"数据加载完成: {len(self.signals)} 个样本")

    def _load_metadata_only(self):
        """仅加载元数据，信号数据动态加载"""
        self.metadata_list = []
        self.sample_keys = []

        with h5py.File(self.h5_path, 'r') as hf:
            self.sample_keys = sorted([k for k in hf.keys() if k.startswith('sample_')],
                                      key=lambda x: int(x.split('_')[1]))

            for key in tqdm(self.sample_keys, desc="加载元数据"):
                group = hf[key]
                metadata = {
                    "row": group.attrs.get("row", 0),
                    "col": group.attrs.get("col", 0),
                    "channel_names": group.attrs.get("channel_names", []),
                    "file_name": group.attrs.get("file_name", ""),
                    "signal_name": group.attrs.get("signal_name", ""),
                    "surface_id": group.attrs.get("surface_id", -1),  # 曲面类型ID
                }
                self.metadata_list.append(metadata)

        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)

    def __len__(self):
        if self.full_load_to_memory:
            return len(self.signals)
        else:
            return len(self.sample_keys)

    def __getitem__(self, idx):
        if self.full_load_to_memory:
            signal = self.signals[idx]
            label = self.labels[idx]
            metadata = self.metadata_list[idx]
        else:
            # 动态加载模式
            key = self.sample_keys[idx]
            group = self.h5_file[key]
            signal = torch.from_numpy(group['signal'][:].astype(np.float32))
            metadata = self.metadata_list[idx]
            label = torch.tensor([metadata["row"], metadata["col"]], dtype=torch.long)

        if self.transform:
            signal = self.transform(signal)

        return signal, label, metadata

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

def worker_init_fn(worker_id):
    np.random.seed(66 + worker_id)

def collate_padded_fast(batch):
    """优化的collate函数"""
    batch_size = len(batch)

    # 预分配内存
    # 这里需要根据信号最后包含的数据点来修改---由于信号降采样到了200 Hz，因此1 s的数据有200个点
    if config["mode"] == "regression":
        padded_signals = torch.zeros(batch_size, 32, config["target_fs"], dtype=torch.float32)
    if config["mode"] == "classification":
        padded_signals = torch.zeros(batch_size, 32, config["target_fs"])
    masks = torch.zeros(batch_size, 32, dtype=torch.bool)
    labels = torch.zeros(batch_size, 2, dtype=torch.long)
    metadata_list = []

    for i, (signal, label, metadata) in enumerate(batch):
        channel_names = metadata["channel_names"]

        # 向量化操作
        for local_idx, name in enumerate(channel_names):
            global_idx = int(name.split('_')[-1]) - 1
            padded_signals[i, global_idx] = signal[local_idx]
            masks[i, global_idx] = True

        labels[i] = label
        metadata_list.append(metadata)

    return padded_signals, masks, labels, metadata_list


def create_high_performance_loaders(h5_path, batch_size=256, num_workers=0, full_load=True):
    """
    创建高性能数据加载器
    """
    print(f"创建数据加载器 - 批次大小: {batch_size}, Workers: {num_workers}, 全内存: {full_load}")

    dataset = HighPerformanceSensorDataset(h5_path, full_load_to_memory=full_load)

    # 数据集分割
    total_size = len(dataset)
    train_size = int(float(config["train_ratio"]) * total_size)
    val_size = int(float(config["val_ratio"]) * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 保存分割后的数据集到独立的 HDF5 文件
    save_split_datasets(h5_path, train_dataset, val_dataset, test_dataset)

    # 数据加载器配置
    loader_config = {
        'batch_size': batch_size,
        'pin_memory': True,
        'collate_fn': collate_padded_fast,
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_config)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_config)
    # test_loader = DataLoader(test_dataset, shuffle=False, **loader_config)

    return train_loader, val_loader

def save_split_datasets(h5_path, train_dataset, val_dataset, test_dataset):
    """将分割后的数据集保存到不同的HDF5文件中"""
    base_dir = os.path.dirname(h5_path)
    base_name = os.path.splitext(os.path.basename(h5_path))[0]

    # 创建输出文件名
    train_path = os.path.join(base_dir, f"{base_name}_train.h5")
    val_path = os.path.join(base_dir, f"{base_name}_val.h5")
    test_path = os.path.join(base_dir, f"{base_name}_test.h5")

    # 获取源文件中实际的 sample_* 键名（排序后与 Dataset.__getitem__ 的索引一致）
    with h5py.File(h5_path, 'r') as src:
        all_sample_keys = sorted(
            [k for k in src.keys() if k.startswith('sample_')],
            key=lambda x: int(x.split('_')[1])
        )

    def _copy_samples(src_path, dst_path, indices):
        """复制指定索引的样本到目标文件"""
        with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for idx in indices:
                real_key = all_sample_keys[idx]  # 用真实键名代替硬编码
                src.copy(src[real_key], dst, real_key)

    _copy_samples(h5_path, train_path, train_dataset.indices)
    _copy_samples(h5_path, val_path, val_dataset.indices)
    _copy_samples(h5_path, test_path, test_dataset.indices)

    print(f"数据集已分割保存到:\n- 训练集: {train_path}\n- 验证集: {val_path}\n- 测试集: {test_path}")



def create_high_performance_test_loaders(h5_path, batch_size=256, num_workers=0, full_load=True):
    """
    创建高性能数据加载器
    """
    print(f"创建数据加载器 - 批次大小: {batch_size}, Workers: {num_workers}, 全内存: {full_load}")

    dataset = HighPerformanceSensorDataset(h5_path, full_load_to_memory=full_load)

    # 数据集分割
    total_size = len(dataset)
    test_size = total_size

    test_dataset = torch.utils.data.random_split(
        dataset, [test_size],
        generator=torch.Generator().manual_seed(42)
    )[0]

    # 数据加载器配置
    loader_config = {
        'batch_size': batch_size,
        'pin_memory': True,
        'collate_fn': collate_padded_fast,
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
    }

    test_loader = DataLoader(test_dataset, shuffle=False, **loader_config)

    return test_loader

def create_curve_finetune_loaders(curve_h5_path, batch_size=128, num_workers=0, full_load=False):
    """加载曲面数据集，拆分微调训练/验证集"""
    print(f"加载曲面微调数据集: {curve_h5_path}")
    dataset = HighPerformanceSensorDataset(curve_h5_path, full_load_to_memory=full_load)
    total_size = len(dataset)
    # 优先使用曲面微调专属比例，否则回退到通用train_ratio
    curve_train_ratio = config.get("curve_train_ratio", config["train_ratio"])
    train_size = int(curve_train_ratio * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    loader_config = {
        'batch_size': batch_size,
        'pin_memory': True,
        'collate_fn': collate_padded_fast,
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
        'worker_init_fn': worker_init_fn
    }
    curve_train_loader = DataLoader(train_dataset, shuffle=True, **loader_config)
    curve_val_loader = DataLoader(val_dataset, shuffle=False, **loader_config)
    return curve_train_loader, curve_val_loader

def create_plane_full_loaders(h5_path, batch_size=256, num_workers=0, full_load=True):
    """
    加载全部平面数据，不分割训练/验证/测试集，不保存分割文件。
    返回两个 DataLoader 指向同一个全量数据集（train用于训练，val用于评估遗忘）。

    :return: (full_train_loader, full_val_loader) — 两者共享同一份全量数据
    """
    print(f"📦 加载全量平面数据（不分割）- 批次: {batch_size}, 全内存: {full_load}")

    dataset = HighPerformanceSensorDataset(h5_path, full_load_to_memory=full_load)
    total_size = len(dataset)
    print(f"  全量平面样本数: {total_size}")

    loader_config = {
        'batch_size': batch_size,
        'pin_memory': True,
        'collate_fn': collate_padded_fast,
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
    }

    # 训练和验证都使用全部数据（不分割）
    train_loader = DataLoader(dataset, shuffle=True, **loader_config)
    val_loader = DataLoader(dataset, shuffle=False, **loader_config)

    return train_loader, val_loader


# data/loader.py 补充流式平面加载函数（适配 sample_* 格式的 HDF5 文件）
def stream_plane_loader(h5_path, stream_batch=512, num_workers=0):
    """
    增量流式平面数据加载器，分块读取h5，不一次性载入全部数据
    生成器迭代返回 (signal_batch, mask_batch, batch_keys)
    适配 sample_* 组格式的 HDF5 文件，与 collate_padded_fast 保持一致的填充逻辑
    """
    import h5py
    f = h5py.File(h5_path, "r")
    # 筛选并排序所有 sample_* 键，确保加载顺序一致
    sample_keys = sorted(
        [k for k in f.keys() if k.startswith('sample_')],
        key=lambda x: int(x.split('_')[1])
    )
    total_num = len(sample_keys)

    # 从第一个样本推断信号的时间维度
    if total_num == 0:
        f.close()
        return
    first_signal = f[sample_keys[0]]['signal'][:]
    num_channels = config.get("num_channels", 32)
    # 信号形状 [C, T]，取T作为时间长度
    signal_T = first_signal.shape[1] if first_signal.ndim == 2 else first_signal.shape[0]

    for start in range(0, total_num, stream_batch):
        end = min(start + stream_batch, total_num)
        batch_keys = sample_keys[start:end]
        batch_size = end - start

        # 预分配批次张量，与 collate_padded_fast 格式一致
        sig_batch = torch.zeros(batch_size, num_channels, signal_T, dtype=torch.float32)
        mask_batch = torch.zeros(batch_size, num_channels, dtype=torch.bool)

        for i, key in enumerate(batch_keys):
            group = f[key]
            signal = group['signal'][:]  # numpy array [C, T] 或 [T,]
            channel_names = list(group.attrs.get("channel_names", []))

            if signal.ndim == 1:
                # 单通道信号，尝试从 channel_names 推断全局索引
                signal = signal.reshape(1, -1)

            for local_idx, name in enumerate(channel_names):
                # channel_names 格式: Pos_1, Pos_2, ... → 全局索引从1开始
                global_idx = int(name.split('_')[-1]) - 1
                if global_idx < num_channels and local_idx < signal.shape[0]:
                    sig_batch[i, global_idx, :min(signal.shape[1], signal_T)] = \
                        torch.from_numpy(signal[local_idx, :min(signal.shape[1], signal_T)]).float()
                    mask_batch[i, global_idx] = True

        yield sig_batch, mask_batch, batch_keys

    f.close()