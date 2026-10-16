# 执行超分辨的数据合并，可选从文件名中获取点的行列信息，也可以从标签读入（也分为回归任务和分类任务的读取）
import os
import glob
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.signal as signal


warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class SmartDataMerger:
    """
    智能大数据合并器，支持分批处理、断点续传、内存优化和并行处理
    """

    def __init__(self, input_root, output_h5_path, labels_root_dir, mode, batch_size=None, max_workers=4,
                 checkpoint_interval=5, compression="gzip", compression_opts=4, original_fs=1000, target_fs=50, step_um=100):
        """
        :param input_root: 分割后CSV文件的根目录
        :param output_h5_path: 输出的HDF5文件路径
        :param batch_size: 每批处理的样本数(自动计算如果为None)
        :param max_workers: 并行工作线程数
        :param checkpoint_interval: 检查点保存间隔(批次数)
        :param compression: HDF5压缩算法
        :param compression_opts: 压缩级别
        """
        self.input_root = input_root
        self.output_h5_path = output_h5_path
        self.labels_root_dir = labels_root_dir  # 保存为类属性，供其他方法调用
        self.max_workers = max_workers
        self.checkpoint_interval = checkpoint_interval
        self.compression = compression
        self.compression_opts = compression_opts
        self.mode = mode
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.step_um = step_um

        # 自动计算合适的批次大小
        self.batch_size = self._calculate_batch_size() if batch_size is None else batch_size

        self.checkpoint_file = output_h5_path + '.checkpoint'

    def _calculate_batch_size(self):
        """根据可用内存自动计算批次大小"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)

        # 保守估计: 每个样本约0.5MB (假设32通道, 1000采样点)
        sample_size_mb = 0.125
        safe_batch_size = int((available_gb * 1024 * 0.5) / sample_size_mb)  # 使用50%可用内存

        # 限制在合理范围内
        return max(500, min(safe_batch_size, 10000))

    def _collect_csv_paths(self):
        """收集并排序所有CSV文件路径"""
        search_pattern = os.path.join(self.input_root, "**", "segmented_signals", "*.csv")
        csv_paths = sorted(glob.glob(search_pattern, recursive=True))

        # 验证路径结构
        sample_path = csv_paths[0] if csv_paths else None
        if sample_path and len(sample_path.split(os.sep)) < 4:
            raise ValueError("CSV文件路径结构不符合预期，无法解析元数据")

        return csv_paths

    def _parse_metadata(self, csv_path):
        """从CSV路径解析元数据，并提取有效通道名"""
        parts = csv_path.split(os.sep)
        file_name = parts[-4]  # 原始文件名（如 Data-Row-1）
        group_name = parts[-3]  # 组名（如 SETxColx2）
        signal_name = os.path.splitext(parts[-1])[0]  # 信号名

        # 解析行列信息
        row = int(file_name.split('-')[-1])
        col = int(group_name.replace('SETxColx', ''))

        # 从CSV文件读取通道名（假设列名为 Time, Pos_01, Pos_05, ...）
        try:
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
            channel_names = [col for col in header if col.startswith('Pos_')]
        except Exception as e:
            raise ValueError(f"无法从CSV文件解析通道名: {csv_path}") from e

        return file_name, group_name, signal_name, row, col, channel_names

    def _parse_metadata_row_col(self, csv_path, labels_root_dir):
        """从CSV路径解析元数据，并提取有效通道名"""
        # 格式示例 E:\Project_YCJ\Super_Res_Figure\Super_Res_Data\segmented_output\ xue\SET001\segmented_signals\ xx.csv
        parts = csv_path.split(os.sep)
        file_name = parts[-4]  # 原始文件名（如 zhong）
        group_name = parts[-3]  # 组名（如 SET001）
        signal_name = os.path.splitext(parts[-1])[0]  # 信号名

        # 2. 自动查找对应的标签CSV文件
        label_path = os.path.join(labels_root_dir, f"{file_name}.csv")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"未找到与文件名匹配的标签CSV：{label_path}")
        # print(f"自动匹配标签文件：{label_path}")  # 可选：用于调试

        # 从组名提取行索引（SET001 -> 1，对应坐标CSV表头后第1行）
        try:
            # 提取SET后的数字（如"SET001" -> "001" -> 1）
            set_number = int(group_name.replace("SETxColx", ""))
            # 坐标CSV行索引（CSV表头占1行，数据从索引1开始，所以等于set_number）
            coord_row_index = set_number
        except ValueError:
            raise ValueError(f"组名格式错误，应为SET+数字（如SET001）：{group_name}")

        # 从坐标CSV读取对应行的xy值（CSV包含表头，列名自动识别）
        try:
            # 读取CSV（保留表头，自动识别列名）
            coord_df = pd.read_csv(label_path)

            # 检查是否包含x和y列
            if 'x_mm' not in coord_df.columns or 'y_mm' not in coord_df.columns:
                raise ValueError(f"坐标CSV必须包含'x'和'y'列，当前列名：{coord_df.columns.tolist()}")

            # 检查索引是否有效（考虑表头占用一行）
            if coord_row_index < 1 or coord_row_index > len(coord_df):
                raise IndexError(
                    f"组名{group_name}对应索引{coord_row_index - 1}（数据行），超出坐标CSV范围（共{len(coord_df)}行数据）"
                )

            # 提取xy坐标（索引从0开始，所以减1）
            x = float(coord_df.iloc[coord_row_index-1]['x_mm'])
            y = float(coord_df.iloc[coord_row_index-1]['y_mm'])

            # 新增：将坐标转换为微米后按20um步长四舍五入（假设原始单位是毫米）
            # 1. 毫米转微米（1mm = 1000um）
            x_um = x * 1000
            y_um = y * 1000

            # 2. 按20um步长四舍五入（先除以20取整，再乘以20）
            step = self.step_um
            x_um_rounded = round(x_um / step)
            y_um_rounded = round(y_um / step)

            # 将坐标变换成行列数
            x = x_um_rounded
            y = y_um_rounded

        except FileNotFoundError:
            raise FileNotFoundError(f"坐标CSV文件不存在：{label_path}")
        except Exception as e:
            raise ValueError(f"读取坐标CSV失败：{str(e)}")

        # 解析行列信息
        row = y
        col = x

        # 从CSV文件读取通道名（假设列名为 Time, Pos_01, Pos_05, ...）
        try:
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
            channel_names = [col for col in header if col.startswith('Pos_')]
        except Exception as e:
            raise ValueError(f"无法从CSV文件解析通道名: {csv_path}") from e

        return file_name, group_name, signal_name, row, col, channel_names

    def _parse_metadata_xy(self, csv_path, labels_root_dir):
        """从CSV路径解析元数据，并提取有效通道名"""
        # 格式示例 E:\Project_YCJ\Super_Res_Figure\Super_Res_Data\segmented_output\ xue\SET001\segmented_signals\ xx.csv
        parts = csv_path.split(os.sep)
        file_name = parts[-4]  # 原始文件名（如 zhong）
        group_name = parts[-3]  # 组名（如 SET001）
        signal_name = os.path.splitext(parts[-1])[0]  # 信号名

        # 注意：标签文件的命名要和tdms文件的命名保持一致！
        # 2. 自动查找对应的标签CSV文件
        label_path = os.path.join(labels_root_dir, f"{file_name}.csv")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"未找到与文件名匹配的标签CSV：{label_path}")
        # print(f"自动匹配标签文件：{label_path}")  # 可选：用于调试

        # 从组名提取行索引（SET001 -> 1，对应坐标CSV表头后第1行）
        try:
            # 提取SET后的数字（如"SET001" -> "001" -> 1）
            set_number = int(group_name.replace("SET", ""))
            # 坐标CSV行索引（CSV表头占1行，数据从索引1开始，所以等于set_number）
            coord_row_index = set_number
        except ValueError:
            raise ValueError(f"组名格式错误，应为SET+数字（如SET001）：{group_name}")

        # 从坐标CSV读取对应行的xy值（CSV包含表头，列名自动识别）
        try:
            # 读取CSV（保留表头，自动识别列名）
            coord_df = pd.read_csv(label_path)

            # 检查是否包含x和y列
            if 'x_mm' not in coord_df.columns or 'y_mm' not in coord_df.columns:
                raise ValueError(f"坐标CSV必须包含'x'和'y'列，当前列名：{coord_df.columns.tolist()}")

            # 检查索引是否有效（考虑表头占用一行）
            if coord_row_index < 1 or coord_row_index > len(coord_df):
                raise IndexError(
                    f"组名{group_name}对应索引{coord_row_index - 1}（数据行），超出坐标CSV范围（共{len(coord_df)}行数据）"
                )

            # 提取xy坐标（索引从0开始，所以减1）
            x = float(coord_df.iloc[coord_row_index-1]['x_mm'])
            y = float(coord_df.iloc[coord_row_index-1]['y_mm'])


        except FileNotFoundError:
            raise FileNotFoundError(f"坐标CSV文件不存在：{label_path}")
        except Exception as e:
            raise ValueError(f"读取坐标CSV失败：{str(e)}")

        # 解析行列信息
        row = y
        col = x

        # 从CSV文件读取通道名（假设列名为 Time, Pos_01, Pos_05, ...）
        try:
            with open(csv_path, 'r') as f:
                header = f.readline().strip().split(',')
            channel_names = [col for col in header if col.startswith('Pos_')]
        except Exception as e:
            raise ValueError(f"无法从CSV文件解析通道名: {csv_path}") from e

        return file_name, group_name, signal_name, row, col, channel_names

    def _downsample_signal(self, data, original_fs=1000, target_fs=200):
        """
        高质量降采样方法
        :param data: 输入信号 (C, T)
        :param original_fs: 原始采样率 (Hz)
        :param target_fs: 目标采样率 (Hz)
        :return: 降采样后的信号 (C, T')
        """
        down_factor = original_fs // target_fs

        # 方法1: 抗混叠滤波+降采样 (推荐)
        if down_factor > 1:
            # 设计抗混叠滤波器
            nyquist = target_fs / 2
            b, a = signal.butter(4, nyquist / original_fs * 0.9, btype='low')

            # 应用滤波器并降采样
            downsampled = np.zeros((data.shape[0], data.shape[1] // down_factor))
            for c in range(data.shape[0]):
                filtered = signal.filtfilt(b, a, data[c])
                downsampled[c] = filtered[::down_factor]
            return downsampled

        return data

    def _load_signal(self, csv_path):
        """高效加载信号数据并进行降采样"""
        try:
            # 使用pandas的低内存模式读取
            df = pd.read_csv(csv_path, usecols=lambda x: x != 'Time')
            raw_signal = df.values.T  # (C, T)
            return self._downsample_signal(raw_signal, self.original_fs, self.target_fs)
        except Exception as e:
            print(f"加载文件失败: {csv_path}, 错误: {str(e)}")
            return None

    def _process_batch(self, h5_file, batch_paths, start_idx):
        """处理批次数据并保存通道映射信息"""
        batch_data = []
        batch_metadata = []

        # 第一阶段：并行加载信号和元数据
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._load_signal_and_metadata, p): p for p in batch_paths}
            for future in as_completed(futures):
                csv_path = futures[future]
                try:
                    signal, metadata = future.result()
                    if signal is not None:
                        batch_data.append(signal)
                        batch_metadata.append(metadata)
                except Exception as e:
                    print(f"处理文件失败: {csv_path}, 错误: {str(e)}")
                    continue

        # 第二阶段：写入HDF5
        for i, (signal, metadata) in enumerate(zip(batch_data, batch_metadata)):
            sample_idx = start_idx + i
            file_name, group_name, signal_name, row, col, channel_names = metadata

            # 创建HDF5组并保存数据
            sample_group = h5_file.create_group(f"sample_{sample_idx:07d}")
            sample_group.create_dataset(
                "signal",
                data=signal,  # 原始信号 (C_i, 1000)
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            sample_group.attrs.update({
                "file_name": file_name,
                "group_name": group_name,
                "signal_name": signal_name,
                "row": row,
                "col": col,
                "channel_names": channel_names  # 关键：保存有效通道名列表
            })

        return len(batch_data)

    def _load_signal_and_metadata(self, csv_path):
        """加载信号数据并解析元数据"""
        try:
            # 加载CSV数据（跳过Time列）
            signal = self._load_signal(csv_path)  # 调用统一加载方法
            if signal is None:
                return None, None

            # 解析元数据（包括通道名）
            if self.mode == "origin":
                metadata = self._parse_metadata(csv_path)
            if self.mode == "classification":
                metadata = self._parse_metadata_row_col(csv_path, self.labels_root_dir)
            if self.mode == "regression":
                metadata = self._parse_metadata_xy(csv_path, self.labels_root_dir)
            return signal, metadata
        except Exception as e:
            print(f"加载文件失败: {csv_path}, 错误: {str(e)}")
            return None, None

    def _load_checkpoint(self):
        """加载检查点信息"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return int(f.read().strip())
        return 0

    def _save_checkpoint(self, processed_samples):
        """保存检查点"""
        with open(self.checkpoint_file, 'w') as f:
            f.write(str(processed_samples))

    def _cleanup_checkpoint(self):
        """清理检查点文件"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

    def _validate_channel_names(self, channel_names):
        """检查通道名是否合法（Pos_01到Pos_32且无重复）"""
        seen = set()
        for name in channel_names:
            if not name.startswith('Pos_'):
                raise ValueError(f"非法通道名前缀: {name}")
            try:
                idx = int(name.split('_')[-1])
                if idx < 1 or idx > 32:
                    raise ValueError(f"通道编号越界: {name}")
                if name in seen:
                    raise ValueError(f"重复通道名: {name}")
                seen.add(name)
            except ValueError as e:
                raise ValueError(f"通道名格式错误: {name}") from e

    def _all_validate_channel_consistency(self, h5_file):
        """验证所有样本的通道名是否合法（可选步骤）"""
        print("\n开始验证通道名一致性...")
        for sample_key in tqdm(h5_file.keys(), desc="验证通道名"):
            if sample_key.startswith('sample_'):
                channel_names = h5_file[sample_key].attrs["channel_names"]
                self._validate_channel_names(channel_names)  # 调用校验方法
        print("通道名验证通过！")

    def _sample_validate_channel_consistency(self, h5_file):
        """验证通道名顺序和全局位置的对应关系"""
        sample_keys = list(h5_file.keys())
        if len(sample_keys) > 10000:
            sample_keys = np.random.choice(sample_keys, size=1000, replace=False)

        for key in tqdm(sample_keys, desc="验证通道映射"):
            if key.startswith('sample_'):
                group = h5_file[key]
                channel_names = group.attrs["channel_names"]
                signal = group["signal"][:]  # (C_i, T)

                # 检查通道数量一致性
                assert len(channel_names) == signal.shape[0], \
                    f"{key}: 通道名数量与信号维度不匹配"

                # 检查通道名是否对应全局位置
                for local_idx, name in enumerate(channel_names):
                    global_idx = int(name.split('_')[-1]) - 1
                    assert 0 <= global_idx < 32, f"{key}: 非法全局索引 {global_idx}"

    def merge_to_hdf5(self):
        """主合并流程"""
        csv_paths = self._collect_csv_paths()
        if not csv_paths:
            raise ValueError("未找到任何CSV文件，请检查输入路径")

        total_samples = len(csv_paths)
        processed_samples = self._load_checkpoint()

        print(f"\n{'=' * 50}")
        print(f"开始合并 {total_samples} 个信号样本")
        print(f"输入目录: {self.input_root}")
        print(f"输出文件: {self.output_h5_path}")
        print(f"配置参数: 批次大小={self.batch_size}, 工作线程={self.max_workers}")
        print(f"从检查点恢复: {processed_samples}" if processed_samples > 0 else "全新运行")
        print(f"{'=' * 50}\n")

        mode = 'a' if processed_samples > 0 else 'w'
        with h5py.File(self.output_h5_path, mode) as hf:
            if processed_samples == 0:
                hf.attrs.update({
                    "total_samples": total_samples,
                    "input_root": self.input_root,
                    "batch_size": self.batch_size,
                    "max_channels": 32,  # 新增：明确记录最大通道数
                    "channel_prefix": "Pos_",  # 新增：记录通道名前缀
                })

            progress_bar = tqdm(total=total_samples, initial=processed_samples,
                              desc="合并进度", unit="样本")

            while processed_samples < total_samples:
                batch_paths = csv_paths[processed_samples:processed_samples + self.batch_size]
                batch_size = self._process_batch(hf, batch_paths, processed_samples)

                if batch_size == 0:
                    print(f"警告: 批次 {processed_samples // self.batch_size} 无有效样本")
                    processed_samples += len(batch_paths)
                else:
                    processed_samples += batch_size

                progress_bar.update(batch_size)

                if (processed_samples // self.batch_size) % self.checkpoint_interval == 0:
                    self._save_checkpoint(processed_samples)
                    hf.flush()
                    progress_bar.set_postfix_str("检查点已保存")

            if processed_samples >= total_samples:
                self._sample_validate_channel_consistency(hf)
                self._cleanup_checkpoint()
                progress_bar.set_postfix_str("完成!")

        print(f"\n合并完成! 共处理 {processed_samples}/{total_samples} 个样本")
        print(f"输出文件大小: {os.path.getsize(self.output_h5_path) / (1024 ** 3):.2f} GB")


