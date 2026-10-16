# 使用学习率预热调度器
# 能够完整训练200轮逐渐收敛
import yaml
import torch
from tqdm import tqdm
import pandas as pd
import os
import time
import gc
import numpy as np
import torch.nn.functional as F

# 导入自定义模块
from src.Loss import MultiTaskLoss
from utils.visualization import plot_training_curves
from data.loader import create_high_performance_loaders
from src.model import SuperResolutionSensorModel


# 读取yaml配置（新增分类任务专属配置）
def load_config(config_path="config.yaml"):
    """加载yaml配置文件并固定随机种子"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 自动适配num_workers和device
    config['num_workers'] = min(8, os.cpu_count() // 2) if os.cpu_count() else 0
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 固定随机种子（消除训练随机性，保证可复现）
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = config['enable_benchmark']

    # TF32加速配置（NVIDIA Ampere及以上架构有效）
    if config.get('enable_tf32', True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 分类任务默认配置（防止配置缺失）
    config.setdefault('label_smoothing', 0.1)  # 标签平滑缓解过拟合
    config.setdefault('lr_warmup_epochs', 5)  # 学习率预热轮数
    config.setdefault('max_grad_norm', 1.0)  # 最大梯度范数
    config.setdefault('class_weight', True)  # 是否使用类别权重
    return config


# 加载配置
config = load_config()


# 兼容PyTorch版本的学习率打印函数
def print_lr(optimizer, epoch):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"📌 Epoch {epoch + 1} - 学习率: {current_lr:.2e}")


# 滑动平均计算（平滑指标波动）
class RunningAverage:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return self.get_avg()

    def get_avg(self):
        return np.mean(self.values) if self.values else 0.0


# 学习率预热调度器（解决初始学习率不适配问题）
class WarmupReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, warmup_epochs=5, init_lr=1e-5, **kwargs):
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.current_epoch = 0
        # 保存原始LR参数
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        super().__init__(optimizer, **kwargs)

    def step(self, metrics, epoch=None):
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：线性提升学习率
            warmup_lr = self.init_lr + (self.base_lrs[0] - self.init_lr) * (self.current_epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = warmup_lr
            self.current_epoch += 1
            print_lr(self.optimizer, self.current_epoch - 1)
            return
        # 预热结束后，使用原始的ReduceLROnPlateau逻辑
        if epoch is None:
            epoch = self.current_epoch
            self.current_epoch += 1
        super().step(metrics, epoch)


def train_model_high_performance(model, train_loader, val_loader):
    """适配高维行列分类任务的训练函数（提升准确率核心版）"""
    device = config['device']
    model = model.to(device)
    num_rows, num_cols = config['num_rows'], config['num_cols']
    print(f"📌 分类任务维度：行{num_rows}类 | 列{num_cols}类")

    # ========== 1. 优化器+学习率策略（核心：预热+平缓衰减） ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5),
        betas=(config.get('betas_1', 0.9), config.get('betas_2', 0.999))
    )

    # 学习率调度：先5轮预热（线性从1e-5升到基础LR），再平缓衰减
    scheduler = WarmupReduceLROnPlateau(
        optimizer,
        warmup_epochs=config['lr_warmup_epochs'],
        init_lr=1e-5,
        mode='min',
        factor=0.7,  # 更平缓的衰减（从0.5→0.7）
        patience=10,  # 更长的耐心（从8→10）
        threshold=1e-4,
        min_lr=1e-6,
        # verbose=True
    )

    # ========== 2. 损失函数优化（核心：标签平滑+类别权重） ==========
    # 计算类别权重（解决样本不均衡，高维分类必加）
    row_class_weights, col_class_weights = None, None
    if config['class_weight']:
        print("⚙️ 计算类别权重（解决样本不均衡）...")
        row_counts = torch.zeros(num_rows, device=device)
        col_counts = torch.zeros(num_cols, device=device)
        with torch.no_grad():
            for _, _, labels, _ in train_loader:
                row_labels = labels[:, 0].to(device)
                col_labels = labels[:, 1].to(device)
                row_counts += torch.bincount(row_labels, minlength=num_rows)
                col_counts += torch.bincount(col_labels, minlength=num_cols)
        # 计算逆频率权重
        row_class_weights = 1.0 / (row_counts / row_counts.sum() + 1e-6)
        col_class_weights = 1.0 / (col_counts / col_counts.sum() + 1e-6)
        # 权重归一化
        row_class_weights = row_class_weights / row_class_weights.mean()
        col_class_weights = col_class_weights / col_class_weights.mean()
        print(f"✅ 行类别权重计算完成 | 列类别权重计算完成")

    # 自定义多任务损失（加入标签平滑+类别权重，兼容原MultiTaskLoss）
    def custom_multitask_loss(outputs, row_labels, col_labels):
        if model.joint_classification:
            row_probs = torch.max(outputs, dim=2)[0]
            col_probs = torch.max(outputs, dim=1)[0]
            row_logits = F.log_softmax(row_probs, dim=1)
            col_logits = F.log_softmax(col_probs, dim=1)
        else:
            row_logits, col_logits = outputs
            row_logits = F.log_softmax(row_logits, dim=1)
            col_logits = F.log_softmax(col_logits, dim=1)

        # 标签平滑（缓解过拟合，提升泛化）
        def label_smoothing(labels, num_classes, eps=config['label_smoothing']):
            one_hot = F.one_hot(labels, num_classes).float()
            return (1 - eps) * one_hot + eps / num_classes

        row_smooth = label_smoothing(row_labels, num_rows).to(device)
        col_smooth = label_smoothing(col_labels, num_cols).to(device)

        # 交叉熵损失（带类别权重+标签平滑）
        row_loss = -torch.sum(row_smooth * row_logits, dim=1)
        col_loss = -torch.sum(col_smooth * col_logits, dim=1)
        if row_class_weights is not None:
            row_loss = row_loss * row_class_weights[row_labels]
        if col_class_weights is not None:
            col_loss = col_loss * col_class_weights[col_labels]

        # 多任务损失加权（行/列损失各0.5，可配置）
        loss = 0.5 * row_loss.mean() + 0.5 * col_loss.mean()
        return loss

    # ========== 3. 训练初始化（优化指标判断+保存逻辑） ==========
    best_val_acc = 0.0  # 高维分类优先看准确率，而非损失
    patience_counter = 0
    loss_improve_threshold = 1e-4
    acc_improve_threshold = 2e-3  # 提高准确率提升阈值（从1e-3→2e-3）
    os.makedirs(config['result_save_dir'], exist_ok=True)

    # 滑动平均器（加大窗口，更稳定）
    train_loss_smoother = RunningAverage(window_size=8)
    val_loss_smoother = RunningAverage(window_size=8)
    val_acc_smoother = RunningAverage(window_size=8)

    # 训练历史（新增行/列单独准确率，方便排查）
    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [], 'train_row_acc': [], 'train_col_acc': [],
        'val_loss': [], 'val_acc': [], 'val_row_acc': [], 'val_col_acc': [],
        'learning_rate': [], 'throughput': []
    }

    # GPU预热（匹配实际输入，避免预热无效）
    if torch.cuda.is_available() and config['warmup_batches'] > 0:
        print("GPU预热中...")
        with torch.no_grad():
            dummy_shape = config['dummy_input_shape']
            for _ in range(config['warmup_batches']):
                dummy_input = torch.randn(*dummy_shape).to(device, non_blocking=True)
                dummy_mask = torch.ones(dummy_shape[0], dummy_shape[1], dtype=torch.bool).to(device, non_blocking=True)
                _ = model(dummy_input, dummy_mask)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("GPU预热完成！")

    # ========== 4. 核心训练循环（优化准确率计算+梯度更新） ==========
    print("🚀 启动高维分类训练（聚焦准确率提升）...")
    for epoch in range(config['num_epochs']):
        # 训练阶段：开启模型BN/Dropout，累计行/列单独准确率
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_row_correct = 0
        train_col_correct = 0
        train_total = 0
        total_batch_time = 0.0
        total_samples = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        for batch_idx, (signals, masks, labels, _) in enumerate(pbar):
            batch_start = time.time()
            # 异步数据加载
            signals = signals.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            row_labels = labels[:, 0].to(device, non_blocking=True)
            col_labels = labels[:, 1].to(device, non_blocking=True)
            batch_size = len(labels)

            # 梯度清零
            optimizer.zero_grad(set_to_none=True)   # 清空模型所有参数的梯度值

            # 前向传播
            outputs = model(signals, masks)         # 得到预测值
            # 使用自定义损失（标签平滑+类别权重）
            loss = custom_multitask_loss(outputs, row_labels, col_labels)

            # 反向传播+梯度裁剪（防止梯度消失/爆炸，高维分类关键）
            loss.backward()      # 反向传播计算出梯度，才能知道权重更新的方向
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])  # 对梯度进行裁剪
            optimizer.step()     # 根据计算的梯度进行参数的更新

            # ========== 关键：拆分行/列准确率计算（方便排查哪部分学不会） ==========
            if model.joint_classification:
                row_probs = torch.max(outputs, dim=2)[0]
                col_probs = torch.max(outputs, dim=1)[0]
                _, row_preds = torch.max(row_probs, dim=1)
                _, col_preds = torch.max(col_probs, dim=1)
            else:
                row_logits, col_logits = outputs
                _, row_preds = torch.max(row_logits, 1)
                _, col_preds = torch.max(col_logits, 1)

            # 单独统计行/列准确率
            batch_row_correct = (row_preds == row_labels).sum().item()
            batch_col_correct = (col_preds == col_labels).sum().item()
            batch_correct = ((row_preds == row_labels) & (col_preds == col_labels)).sum().item()

            # 累计指标
            train_loss += loss.item() * batch_size  # 按批次大小加权，避免批次不均影响
            train_correct += batch_correct
            train_row_correct += batch_row_correct
            train_col_correct += batch_col_correct
            train_total += batch_size
            total_batch_time += (time.time() - batch_start)
            total_samples += batch_size

            # 进度条实时显示行/列准确率（核心排查项）
            if batch_idx % 5 == 0:  # 更频繁更新，及时观察
                current_lr = optimizer.param_groups[0]['lr']
                gpu_util = torch.cuda.utilization() if torch.cuda.is_available() else 0
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_correct / batch_size:.4f}',
                    'row_acc': f'{batch_row_correct / batch_size:.4f}',
                    'col_acc': f'{batch_col_correct / batch_size:.4f}',
                    'GPU': f'{gpu_util}%',
                    'LR': f'{current_lr:.2e}'
                })

        # 训练阶段统计（按样本数平均，更准确）
        avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        train_row_acc = train_row_correct / train_total if train_total > 0 else 0.0
        train_col_acc = train_col_correct / train_total if train_total > 0 else 0.0
        avg_throughput = total_samples / total_batch_time if total_batch_time > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        smoothed_train_loss = train_loss_smoother.update(avg_train_loss)

        # 验证阶段：无梯度+行/列准确率拆分
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_row_correct = 0
        val_col_correct = 0
        val_total = 0
        with torch.no_grad():
            for signals, masks, labels, _ in val_loader:
                signals = signals.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                row_labels = labels[:, 0].to(device, non_blocking=True)
                col_labels = labels[:, 1].to(device, non_blocking=True)
                batch_size = len(labels)

                outputs = model(signals, masks)
                loss = custom_multitask_loss(outputs, row_labels, col_labels)

                # 行/列预测与准确率统计
                if model.joint_classification:
                    row_probs = torch.max(outputs, dim=2)[0]
                    col_probs = torch.max(outputs, dim=1)[0]
                    _, row_preds = torch.max(row_probs, dim=1)
                    _, col_preds = torch.max(col_probs, dim=1)
                else:
                    row_logits, col_logits = outputs
                    _, row_preds = torch.max(row_logits, 1)
                    _, col_preds = torch.max(col_logits, 1)

                batch_row_correct = (row_preds == row_labels).sum().item()
                batch_col_correct = (col_preds == col_labels).sum().item()
                batch_correct = ((row_preds == row_labels) & (col_preds == col_labels)).sum().item()

                val_loss += loss.item() * batch_size
                val_correct += batch_correct
                val_row_correct += batch_row_correct
                val_col_correct += batch_col_correct
                val_total += batch_size

        # 验证阶段统计
        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        val_row_acc = val_row_correct / val_total if val_total > 0 else 0.0
        val_col_acc = val_col_correct / val_total if val_total > 0 else 0.0
        # 验证指标滑动平均（更稳定的模型评估）
        smoothed_val_loss = val_loss_smoother.update(avg_val_loss)
        smoothed_val_acc = val_acc_smoother.update(val_accuracy)

        # GPU缓存清理（轻量，不影响训练速度）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 记录训练历史（含行/列单独准确率）
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['train_row_acc'].append(train_row_acc)
        history['train_col_acc'].append(train_col_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_row_acc'].append(val_row_acc)
        history['val_col_acc'].append(val_col_acc)
        history['learning_rate'].append(current_lr)
        history['throughput'].append(avg_throughput)

        # 打印详细日志（重点看行/列准确率差距）
        print(f"\n📊 Epoch {epoch + 1}/{config['num_epochs']} 详细统计:")
        print(
            f"🔹 训练：Loss {avg_train_loss:.4f} | Acc {train_accuracy:.4f} | 行Acc {train_row_acc:.4f} | 列Acc {train_col_acc:.4f}")
        print(
            f"🔹 验证：Loss {avg_val_loss:.4f} | Acc {val_accuracy:.4f} | 行Acc {val_row_acc:.4f} | 列Acc {val_col_acc:.4f}")
        print(f"🔹 学习率：{current_lr:.2e} | 平均吞吐量：{avg_throughput:.1f} samples/s")

        # ========== 5. 学习率更新+模型保存（优先准确率，适配高维分类） ==========
        # 学习率基于平滑后的验证损失更新
        scheduler.step(smoothed_val_loss)

        # 模型保存逻辑：优先看**平滑后的验证准确率**（高维分类核心指标）
        is_best = False
        acc_improvement = smoothed_val_acc - best_val_acc
        loss_improvement = best_val_acc - smoothed_val_loss  # 辅助判断

        if acc_improvement > acc_improve_threshold:
            best_val_acc = smoothed_val_acc
            is_best = True
            print(f"📈 验证准确率刷新新高（提升 {acc_improvement:.4f}）: {best_val_acc:.4f}")
        # 损失大幅下降但准确率未升时，也保存（防止模型陷入局部最优）
        elif loss_improvement > loss_improve_threshold * 5 and val_accuracy > 0.15:
            is_best = True
            print(f"📉 验证损失大幅下降，保存模型：{smoothed_val_loss:.4f}")

        if is_best:
            torch.save(model.state_dict(), config['model_path'])
            patience_counter = 0
            print(f"✅ 保存最佳模型到 {config['model_path']}!\n")
        else:
            patience_counter += 1
            print(f"⚠️  验证指标无显著改善，早停计数: {patience_counter}/{config['patience']}\n")

            # 触发早停（保留足够训练空间）
            if patience_counter >= config['patience']:
                print(f"\n🛑 早停触发！在第 {epoch + 1} 轮停止训练")
                print(f"🏆 训练结束 - 最佳验证准确率: {best_val_acc:.4f}")
                break

    # 保存训练历史（含行/列准确率，方便后续分析）
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(config['result_save_dir'], 'training_history_with_row_col.csv')
    history_df.to_csv(history_csv_path, index=False)
    print(f"\n📁 训练历史（含行/列准确率）已保存到 {history_csv_path}")

    # 绘制训练曲线（兼容无可视化环境）
    try:
        plot_training_curves(history_df, csv_path=history_csv_path)  # 曲线与CSV同名保存
        print("📈 训练曲线绘制完成！")
    except Exception as e:
        print(f"⚠️  训练曲线绘制失败: {str(e)}")

    return model


if __name__ == "__main__":
    # 环境初始化（GPU/CPU配置）
    def setup_environment():
        if torch.cuda.is_available():
            gpu_id = config.get('gpu_id', 0)
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            print(f"⚡ 使用GPU: {torch.cuda.get_device_name(gpu_id)}")
            print(f"💾 GPU总内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3:.1f} GB")
            device = torch.device(f"cuda:{gpu_id}")
        else:
            print("⚙️ 使用CPU训练（建议使用GPU，高维分类训练慢）")
            device = torch.device('cpu')
        return device


    # 初始化训练环境
    device = setup_environment()
    H5_PATH = config.get("raw_h5_path", "")
    assert os.path.exists(H5_PATH), f"❌ 数据文件不存在: {H5_PATH}"

    # 数据加载（优先配置批次，保证数据加载稳定性）
    try:
        train_loader, val_loader = create_high_performance_loaders(
            H5_PATH,
            batch_size=config.get("batch_size", 128),
            # num_workers=config.get("num_workers", 0),
            full_load=config.get("full_load", False)
        )
        print(
            f"✅ 数据加载成功 - 批次大小: {config.get('batch_size', 128)} | 训练集批次: {len(train_loader)} | 验证集批次: {len(val_loader)}")
    except Exception as e:
        print(f"❌ 标准加载失败: {str(e)[:100]}...")
        print(f"🔄 回退到低内存配置 - 批次大小: {config['fallback_batch_size']}")
        train_loader, val_loader = create_high_performance_loaders(
            H5_PATH,
            batch_size=config["fallback_batch_size"],
            # num_workers=config["fallback_num_workers"],
            full_load=False
        )

    # 模型初始化（确认联合分类模式）
    model = SuperResolutionSensorModel(
        num_rows=config["num_rows"],
        num_cols=config["num_cols"],
        num_channels=config["num_channels"],
        seq_len=config["seq_len"],
        joint_classification=config["joint_mode"]
    )
    print(f"\n🤖 模型模式：{'联合分类' if config['joint_mode'] else '独立分类'}")

    # 打印模型参数量（确认模型能力匹配任务）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数量统计:")
    print(f"   总参数量: {total_params:,} (建议≥100万，适配51×51分类)")
    print(f"   可训练参数量: {trainable_params:,}")

    # 启动训练
    print(f"\n🚀 启动超分辨率传感器分类训练 - 配置文件: config.yaml")
    trained_model = train_model_high_performance(model, train_loader, val_loader)
    print("\n🎉 训练流程全部完成！")