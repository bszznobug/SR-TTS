import os
import yaml
import sys
import glob
# 添加路径以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
# 导入自定义模块
from tdms_reader import process_single_tdms as process_tdms_file
from data_merger import SmartDataMerger


def get_config():
    """
    加载配置文件（优先级：主目录main/config.yaml > 当前目录config_data.yaml）
    :return: 配置字典, 项目根目录路径
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)  # 项目根目录（当前目录的上一级）

    # 优先加载主配置文件
    main_config_path = os.path.join(root_dir, 'main', 'config.yaml')
    data_config_path = os.path.join(current_dir, 'config_data.yaml')

    # 确定最终使用的配置文件路径
    if os.path.exists(main_config_path):
        config_path = main_config_path
    elif os.path.exists(data_config_path):
        config_path = data_config_path
    else:
        raise FileNotFoundError(
            f"未找到配置文件！\n"
            f"1. 主配置路径: {main_config_path}\n"
            f"2. 数据配置路径: {data_config_path}"
        )

    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 兼容逻辑：如果主配置里没有paths，尝试补充数据配置
    if 'paths' not in cfg and os.path.exists(data_config_path) and config_path != data_config_path:
        with open(data_config_path, 'r', encoding='utf-8') as f2:
            data_cfg = yaml.safe_load(f2)
            cfg.update(data_cfg)

    print(f"成功加载配置文件: {config_path}")
    return cfg, root_dir


def process_tdms_batch(config):
    """批量处理 TDMS 文件"""
    tdms_folder = config["paths"]["tdms_input_folder"]
    output_root = config["paths"]["tdms_output_root"]

    # 找到所有 .tdms 文件
    tdms_list = glob.glob(os.path.join(tdms_folder, "*.tdms"))
    if not tdms_list:
        raise ValueError("未找到任何 TDMS 文件，请确认路径是否正确。")

    print(f"共发现 {len(tdms_list)} 个 TDMS 文件，开始批量处理...\n")
    print(f"选取的峰段为 {config['tdms_process']['peak_start_idx']} -{config['tdms_process']['peak_end_idx']}\n")
    for tdms_path in tdms_list:
        process_tdms_file(tdms_path, output_root,
    channel_count=config["tdms_process"]["channel_count"],
    channel_prefix=config["tdms_process"]["channel_prefix"],
    DO_FILTER=config["tdms_process"]["do_filter"],
    FILTER_TYPE=config["tdms_process"]["filter_type"],
    FS=config["tdms_process"]["fs"],
    CUTOFF=config["tdms_process"]["cutoff"],
    TH_RATIO=config["tdms_process"]["th_ratio"],
    min_peak_distance=config["tdms_process"]["min_peak_distance"],
    window_size=config["tdms_process"]["window_size"],
    peak_start_idx = config["tdms_process"]["peak_start_idx"],
    peak_end_idx = config["tdms_process"]["peak_end_idx"])

    print("\nTDMS 批量处理完成！分割后的文件已保存至:", output_root)


def merge_to_h5(config):
    """将分割后的 CSV 文件合并为 H5 数据集"""
    input_root = config["paths"]["tdms_output_root"]
    output_h5 = config["paths"]["h5_output_path"]
    labels_root_dir = config["paths"]["labels_root_dir"]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    # 创建合并器实例
    merger = SmartDataMerger(
        input_root=input_root,
        output_h5_path=output_h5,
        labels_root_dir=labels_root_dir,
        mode=config["data_merger"]["mode"],
        batch_size=config["data_merger"]["batch_size"],
        max_workers=config["data_merger"]["max_workers"],
        checkpoint_interval=config["data_merger"]["checkpoint_interval"],
        compression=config["data_merger"]["compression"],
        compression_opts=config["data_merger"]["compression_opts"],
        original_fs=config["data_merger"]["original_fs"],
        target_fs=config["data_merger"]["target_fs"],
        step_um=config["data_merger"]["step_um"]
    )

    # 执行合并
    print(f"合并模式为 {config['data_merger']['mode']}，合并后数据采样率为{config['data_merger']['target_fs']}，数据集将保存在{output_h5}\n")
    merger.merge_to_hdf5()
    print(f"\nH5 数据集合并完成！文件保存至: {output_h5}")


def main():
    """数据预处理主流程"""
    try:
        # 1. 加载配置
        print("加载配置文件...")
        config, root_dir = get_config()

        # 2. 批量处理 TDMS 文件（分割、保存、绘图）
        print("\n===== 开始处理 TDMS 文件 =====")
        process_tdms_batch(config)

        # 3. 合并分割后的 CSV 为 H5 数据集
        print("\n===== 开始合并为 H5 数据集 =====")
        merge_to_h5(config)

        print("\n===== 数据预处理全部完成！=====")

    except Exception as e:
        print(f"\n预处理过程出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()