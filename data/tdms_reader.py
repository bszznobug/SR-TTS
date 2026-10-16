"""
功能：
1. 读取某文件夹下所有 .tdms 文件
2. 对每个 .tdms 文件执行同样的分割/保存/绘图流程
3. 输出结果按原始文件名创建子文件夹
"""
import os
from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import find_peaks

# ========= 0. 全局配置 =========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


# ------------ 1. 工具函数 ------------
def build_rename_map(prefix: str = "Pos", n: int = 32) -> dict:
    return {str(i): f"{prefix}_{i:02d}" for i in range(1, n + 1)}


def build_ordered_names(prefix: str = "Pos", n: int = 32) -> list:
    return [f"{prefix}_{i:02d}" for i in range(1, n + 1)]


def apply_filter(data: np.ndarray, fs: float, ftype: str, cutoff, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    if ftype == 'band':
        wn = np.asarray(cutoff) / nyq
    else:
        wn = float(cutoff) / nyq
    sos = signal.butter(order, wn, btype=ftype, output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    data[:] = filtered
    return data


def demean_inplace(data: np.ndarray) -> None:
    data -= np.mean(data)


def one_hz_energy_ratio(sig: np.ndarray, fs: float) -> float:
    n = sig.size
    P = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mask = (freqs >= 0.5) & (freqs <= 2)
    if not np.any(mask):
        return 0.0
    e_1hz = np.sum(P[mask])
    e_all = np.sum(P)
    return e_1hz / (e_all + 1e-12)


def find_signal_peaks(signal_data: np.ndarray, fs: float, min_peak_distance: float ) -> tuple:
    neg_peaks, _ = find_peaks(-signal_data, distance=int(fs * min_peak_distance))
    pos_peaks, _ = find_peaks(signal_data, distance=int(fs * min_peak_distance))
    valid_peaks = []
    for neg in neg_peaks:
        subsequent_pos = pos_peaks[pos_peaks > neg]
        if len(subsequent_pos) > 0:
            valid_peaks.append((neg, subsequent_pos[0]))
    return valid_peaks


def get_reference_channel(df: pd.DataFrame, ordered_names: list) -> str:
    max_amp = -np.inf
    ref_ch = ordered_names[0]
    for ch in ordered_names:
        data = df[ch].values
        amp = np.max(np.abs(data))
        if amp > max_amp:
            max_amp = amp
            ref_ch = ch
    return ref_ch


def segment_signal(df: pd.DataFrame, peaks: list, fs: float, window_size: float = 1.0, peak_start_idx=2, peak_end_idx=20) -> list:
    """
    根据参考通道的峰位置，对所有通道进行信号分割。
    舍弃第一个和最后一个信号段，不补零。
    返回: 分割后的信号列表（从第二个信号开始，到倒数第二个结束）。
    """
    segments = []
    half_window = int(fs * window_size / 2)

    # 只遍历中间段：跳过第一个和最后一个
    for neg_peak, pos_peak in peaks[peak_start_idx: peak_end_idx]:
        center = (neg_peak + pos_peak) // 2
        start = max(0, center - half_window)
        end = min(len(df), center + half_window)

        segment = df.iloc[start:end].copy()
        segments.append(segment)

    return segments


def save_segments_to_csv(segments: list, g_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, seg in enumerate(segments):
        csv_path = os.path.join(output_dir, f"{g_name}_signal_{i + 1:03d}.csv")
        seg.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"   已保存信号段：{csv_path}")


def plot_full_signal_with_cuts(df: pd.DataFrame, peaks: list, g_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        if col != 'Time':
            plt.plot(df['Time'], df[col], alpha=0.5, label=col)

    y_max = np.max(df.iloc[:, 1:].values)
    y_min = np.min(df.iloc[:, 1:].values)
    y_range = y_max - y_min
    label_offset = y_range * 0.05

    for i, (neg_peak, pos_peak) in enumerate(peaks):
        plt.axvline(x=df['Time'].iloc[neg_peak], color='r', linestyle='--', alpha=0.7)
        plt.axvline(x=df['Time'].iloc[pos_peak], color='g', linestyle='--', alpha=0.7)
        label_y = y_max - (i % 5) * label_offset
        plt.text(
            df['Time'].iloc[neg_peak],
            label_y,
            f"Signal {i + 1}",
            ha='center',
            va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    plt.title(f"完整信号与切割位置 - {g_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{g_name}_full_signal_with_cuts.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   已保存完整信号图：{fig_path}")


def plot_segments(segments: list, g_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for i, seg in enumerate(segments):
        plt.figure(figsize=(10, 4))
        for col in seg.columns:
            if col != 'Time':
                plt.plot(seg['Time'], seg[col], label=col)
        plt.title(f"信号段 {i + 1} - {g_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{g_name}_signal_{i + 1:03d}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   已保存信号段图：{fig_path}")


def read_group_to_df(group, rename_map: dict, ordered_names: list) -> pd.DataFrame:
    time = None
    df = pd.DataFrame()
    for ch in group.channels():
        if time is None:
            try:
                time = ch.time_track()
            except Exception:
                time = np.arange(len(ch))
            df['Time'] = time
        new_name = rename_map.get(ch.name, ch.name)
        df[new_name] = ch[:]
    df = df.reindex(columns=['Time'] + ordered_names)
    return df


# ------------ 2. 主流程 ------------
def process_single_tdms(tdms_path: str, output_root: str,
                       channel_count=32, channel_prefix="Pos",
                       DO_FILTER=True, FILTER_TYPE='low', FS=1000, CUTOFF=30,
                       TH_RATIO=0.20, min_peak_distance=0.8, window_size=1.0, peak_start_idx=3, peak_end_idx=53):
    """
    处理单个 TDMS 文件（与原先 main() 内容几乎一致，只是把路径参数化）
    """
    # 固定 32 通道
    rename_map = build_rename_map(channel_prefix, channel_count)
    all_ordered_names = build_ordered_names(channel_prefix, channel_count)


    # 读 TDMS
    tdms = TdmsFile.read(tdms_path)

    # ### 新增 ### 用文件名（不含扩展名）创建输出子文件夹
    base_name = os.path.splitext(os.path.basename(tdms_path))[0]
    file_output_dir = os.path.join(output_root, base_name)

    # 逐组处理
    for group in tdms.groups():
        g_name = group.name
        print(f'>>> 处理组：{g_name}  文件：{base_name}')

        # 组内输出路径
        group_dir = os.path.join(file_output_dir, g_name)
        csv_dir = os.path.join(group_dir, 'segmented_signals')
        plot_dir = os.path.join(group_dir, 'segmented_plots')
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        ordered_names = all_ordered_names.copy()
        df = read_group_to_df(group, rename_map, ordered_names)

        # 1. 可选滤波
        if DO_FILTER:
            for col in ordered_names:
                apply_filter(df[col].values, FS, FILTER_TYPE, CUTOFF)

        # 2. 去 DC
        for col in ordered_names:
            demean_inplace(df[col].values)

        # 3. 计算 1 Hz 能量占比并筛选有效通道
        ratio_dict = {}
        for col in ordered_names:
            ratio_dict[col] = one_hz_energy_ratio(df[col].values, FS)

        valid_ch = [c for c in ordered_names if ratio_dict[c] > TH_RATIO]
        df = df[['Time'] + valid_ch]

        # 4. 选择参考通道
        ref_ch = get_reference_channel(df, valid_ch)
        print(f"    参考通道：{ref_ch}")

        # 5. 寻找峰位置
        peaks = find_signal_peaks(df[ref_ch].values, FS, min_peak_distance)
        print(f"    找到 {len(peaks)} 个信号段")

        # 6. 分割信号
        segments = segment_signal(df, peaks, FS, window_size, peak_start_idx, peak_end_idx)

        # 7. 保存
        save_segments_to_csv(segments, g_name, csv_dir)

        # 8. 绘图
        # plot_full_signal_with_cuts(df, peaks, g_name, plot_dir)
        # plot_segments(segments, g_name, plot_dir)



