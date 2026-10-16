# 使用学习率预热调度器
# 能够完整训练200轮逐渐收敛
import sys
import os

# 将项目根目录添加到sys.path，确保从任意目录执行都能找到src/utils/data模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import yaml
import torch
from tqdm import tqdm
import pandas as pd
import time
import gc
import numpy as np
import torch.nn.functional as F

# 导入自定义模块
from utils.visualization import plot_training_curves
from data.loader import  create_curve_finetune_loaders, create_plane_full_loaders
from src.model import SuperResolutionSensorModel

# ==================== 增量学习核心组件 ====================

class EWC:
    """
    Elastic Weight Consolidation (弹性权重巩固)
    通过Fisher信息矩阵度量每个参数对旧任务(平面)的重要性，
    在新任务(曲面)训练时惩罚重要参数的剧烈变化，从而防止灾难性遗忘。

    原理: L_ewc = (λ/2) * Σ F_i * (θ_i - θ_i_old)²
    其中 F_i 是Fisher信息矩阵的对角线近似，表示参数θ_i对旧任务损失函数的敏感度
    """
    def __init__(self, model, dataloader, config, device):
        """
        初始化EWC: 保存旧参数快照并计算Fisher信息矩阵

        :param model: 已加载平面预训练权重的模型
        :param dataloader: 平面数据加载器（用于计算Fisher）
        :param config: 配置字典
        :param device: 计算设备
        """
        self.device = device
        self.ewc_lambda = config.get("ewc_lambda", 500.0)
        self.ewc_enabled = config.get("use_ewc", True)

        # 保存旧模型参数的快照 θ_i_old（训练前状态）
        self.old_params = {}
        for name, param in model.named_parameters():
            self.old_params[name] = param.clone().detach()

        # 在平面数据上计算Fisher信息矩阵对角线 F_i
        if self.ewc_enabled and dataloader is not None:
            self.fisher = self._compute_fisher(model, dataloader, config)
            print(f"✅ EWC初始化完成 | λ={self.ewc_lambda} | "
                  f"保护参数数={len(self.fisher)}")
        else:
            self.fisher = {}
            print("⚠️ EWC未启用（无平面数据或已禁用）")

    def _compute_fisher(self, model, dataloader, config):
        """
        计算Fisher信息矩阵的对角近似: F_i ≈ E[(∂L/∂θ_i)²]

        对每个平面数据样本计算损失梯度，累积梯度的平方作为
        Fisher对角线的无偏估计。值越大说明该参数对平面任务越重要。
        """
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param, device=self.device)

        model.train()
        num_samples = 0
        max_fisher_samples = config.get("ewc_fisher_samples", 500)

        for batch_idx, (signals, masks, labels, _) in enumerate(dataloader):
            if num_samples >= max_fisher_samples:
                break

            signals = signals.to(self.device)
            masks = masks.to(self.device)
            row_labels = labels[:, 0].to(self.device)
            col_labels = labels[:, 1].to(self.device)
            batch_size = signals.shape[0]

            model.zero_grad()
            outputs = model(signals, masks)

            loss = custom_multitask_loss(
                outputs=outputs,
                row_labels=row_labels,
                col_labels=col_labels,
                model=model,
                num_rows=config["num_rows"],
                num_cols=config["num_cols"],
                eps=config.get("label_smoothing", 0.1),
                device=self.device
            )
            loss.backward()

            # 累积梯度平方: F_i += (∂L/∂θ_i)²
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2) * batch_size

            num_samples += batch_size

        # 归一化: 除以总样本数得到期望 E[(∂L/∂θ_i)²]
        for name in fisher:
            fisher[name] = torch.clamp(fisher[name] / max(1, num_samples), min=1e-8)

        model.zero_grad()
        print(f"📊 Fisher矩阵计算完成 (使用 {num_samples} 个平面样本)")
        return fisher

    def penalty(self, model):
        """
        计算EWC正则化惩罚项: L_ewc = (λ/2) * Σ F_i * (θ_i - θ_i_old)²

        :param model: 当前训练中的模型
        :return: EWC惩罚损失标量
        """
        if not self.ewc_enabled or not self.fisher:
            return torch.tensor(0.0, device=self.device)

        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher and name in self.old_params:
                # F_i * (θ_i - θ_i_old)²
                diff = param - self.old_params[name]
                loss += (self.fisher[name] * (diff ** 2)).sum()

        return 0.5 * self.ewc_lambda * loss


def knowledge_distillation_loss(student_outputs, teacher_outputs, model,
                                 T=4.0, joint_classification=False):
    """
    知识蒸馏损失 (Learning without Forgetting)
    让学生模型(新模型)的输出分布接近教师模型(冻结的旧模型)，
    从而在学习新知识时保留旧任务的决策边界。

    使用温度软化 + KL散度:
    L_kd = KL( softmax(teacher/T) || softmax(student/T) ) * T²

    :param student_outputs: 学生模型(当前训练模型)的logits
    :param teacher_outputs: 教师模型(冻结的平面预训练模型)的logits
    :param model: 模型实例(用于判断分类模式)
    :param T: 温度参数，越大输出分布越平滑(默认4.0)
    :param joint_classification: 是否联合分类模式
    :return: 蒸馏损失标量
    """
    if joint_classification:
        # 联合模式: outputs形状为 [B, R, C]
        # 边缘化得到行/列logits
        student_row_logits = torch.max(student_outputs, dim=2)[0]   # [B, R]
        student_col_logits = torch.max(student_outputs, dim=1)[0]   # [B, C]
        teacher_row_logits = torch.max(teacher_outputs, dim=2)[0]
        teacher_col_logits = torch.max(teacher_outputs, dim=1)[0]

        kd_row = F.kl_div(
            F.log_softmax(student_row_logits / T, dim=1),
            F.softmax(teacher_row_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        kd_col = F.kl_div(
            F.log_softmax(student_col_logits / T, dim=1),
            F.softmax(teacher_col_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        return 0.5 * kd_row + 0.5 * kd_col
    else:
        # 独立模式: outputs为 (row_logits [B, R], col_logits [B, C])
        student_row, student_col = student_outputs
        teacher_row, teacher_col = teacher_outputs

        kd_row = F.kl_div(
            F.log_softmax(student_row / T, dim=1),
            F.softmax(teacher_row / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        kd_col = F.kl_div(
            F.log_softmax(student_col / T, dim=1),
            F.softmax(teacher_col / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        return 0.5 * kd_row + 0.5 * kd_col


class PlaneReplayBuffer:
    """
    平面数据重放缓冲区 (Memory Replay)
    存储一部分平面数据样本，在曲面增量训练时随机混合，
    通过"复习"旧数据来强化对旧知识的记忆。

    策略: 环形缓冲区，固定容量，新样本覆盖最旧的样本
    """
    def __init__(self, max_samples=1000):
        self.max_samples = max_samples
        self.buffer = []       # 存储 (signals, masks, labels) 元组
        self.current_idx = 0   # 环形缓冲区写入指针
        self.is_full = False

    def add_batch(self, signals, masks, labels):
        """添加一批平面样本到缓冲区"""
        signals_cpu = signals.detach().cpu()
        masks_cpu = masks.detach().cpu()
        labels_cpu = labels.detach().cpu()

        for i in range(signals_cpu.shape[0]):
            sample = (
                signals_cpu[i].clone(),
                masks_cpu[i].clone(),
                labels_cpu[i].clone()
            )
            if self.is_full:
                self.buffer[self.current_idx] = sample
            else:
                self.buffer.append(sample)

            self.current_idx += 1
            if self.current_idx >= self.max_samples:
                self.current_idx = 0
                self.is_full = True

    def sample(self, n, device):
        """从缓冲区随机采样n个样本"""
        if len(self.buffer) == 0:
            return None, None, None

        indices = np.random.choice(len(self.buffer), min(n, len(self.buffer)), replace=False)

        signals_list, masks_list, labels_list = [], [], []
        for idx in indices:
            s, m, l = self.buffer[idx]
            signals_list.append(s)
            masks_list.append(m)
            labels_list.append(l)

        signals = torch.stack(signals_list).to(device)
        masks = torch.stack(masks_list).to(device)
        labels = torch.stack(labels_list).to(device)
        return signals, masks, labels

    def fill(self, dataloader, device):
        """用平面数据预填充缓冲区"""
        print(f"🔄 填充平面重放缓冲区 (容量={self.max_samples})...")
        filled = 0
        for signals, masks, labels, _ in dataloader:
            if filled >= self.max_samples:
                break
            self.add_batch(signals.to(device), masks.to(device), labels.to(device))
            filled += signals.shape[0]
        print(f"✅ 重放缓冲区就绪: {len(self.buffer)} 个平面样本")

    def __len__(self):
        return len(self.buffer)


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
    config.setdefault('acc_improve_threshold', 0.001)  # 准确率提升阈值，超过才保存模型

    # ========== 增量学习默认配置 ==========
    # EWC (弹性权重巩固)
    config.setdefault('use_ewc', True)              # 是否启用EWC
    config.setdefault('ewc_lambda', 500.0)           # EWC正则化强度（越大越保护旧权重）
    config.setdefault('ewc_fisher_samples', 500)     # 计算Fisher使用的平面样本数

    # 知识蒸馏 (KD)
    config.setdefault('use_kd', True)                # 是否启用知识蒸馏
    config.setdefault('kd_temperature', 4.0)         # 蒸馏温度（越大输出越平滑）
    config.setdefault('kd_weight', 0.5)              # 蒸馏损失权重

    # 记忆重放 (Memory Replay)
    config.setdefault('use_replay', True)            # 是否启用平面数据重放
    config.setdefault('replay_buffer_size', 1000)    # 重放缓冲区容量
    config.setdefault('replay_weight', 0.3)          # 重放蒸馏损失权重

    # 按层选择性解冻 (warmup阶段结束后生效)
    # "head_only":           仅分类头 (最保守，纯迁移学习)
    # "head+cnn":            分类头 + 空间CNN
    # "head+transformer":    分类头 + Transformer时序层
    # "head+cnn+transformer":分类头 + CNN + Transformer (默认)
    # "all":                 全部解冻 (含嵌入/注意力/投影)
    config.setdefault('unfreeze_layers', 'head+cnn+transformer')
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

def get_layer_group(name):
    """
    根据参数名称判断属于哪个层组。
    返回: 'head' | 'cnn' | 'transformer' | 'attention' | 'embed' | 'projection'
    """
    if "row_head" in name or "col_head" in name or "joint_head" in name:
        return 'head'
    if "spatial_cnn" in name:
        return 'cnn'
    if "temporal_encoder" in name:
        return 'transformer'
    if "channel_attention" in name:
        return 'attention'
    if "channel_embed" in name or "pos_encoder" in name:
        return 'embed'
    if "projection" in name:
        return 'projection'
    return 'other'


def apply_layer_freeze(model, unfreeze_layers="head_only"):
    """
    按配置选择性冻结/解冻模型的各个层组。

    unfreeze_layers 选项:
      - "head_only":           仅训练分类头（最保守，纯迁移学习）
      - "head+cnn":            训练分类头 + 空间CNN层
      - "head+transformer":    训练分类头 + Transformer时序层
      - "head+cnn+transformer":训练分类头 + CNN + Transformer
      - "all":                 全部解冻（含嵌入层、注意力、投影层）

    分类头永远可训练。
    """
    # 定义哪些层组可以训练
    trainable_groups = {'head'}  # head 始终可训练

    if unfreeze_layers == "head_only":
        pass  # 只训练 head
    elif unfreeze_layers == "head+cnn":
        trainable_groups.add('cnn')
    elif unfreeze_layers == "head+transformer":
        trainable_groups.add('transformer')
    elif unfreeze_layers == "head+cnn+transformer":
        trainable_groups.add('cnn')
        trainable_groups.add('transformer')
    elif unfreeze_layers == "all":
        trainable_groups.update(['cnn', 'transformer', 'attention', 'embed', 'projection'])
    else:
        print(f"⚠️ 未知的 unfreeze_layers='{unfreeze_layers}'，回退为 'head_only'")

    # 执行冻结/解冻
    frozen_count = 0
    trainable_count = 0
    layer_stats = {}  # 统计每层的可训练参数数

    for name, param in model.named_parameters():
        group = get_layer_group(name)

        if group in trainable_groups:
            param.requires_grad = True
            trainable_count += param.numel()
            layer_stats[group] = layer_stats.get(group, 0) + param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    # 打印层冻结状态
    total = frozen_count + trainable_count
    print(f"\n{'─'*50}")
    print(f"📋 层冻结策略: {unfreeze_layers}")
    print(f"{'─'*50}")
    print(f"  {'层组':<20} {'状态':<8} {'参数量':>12}")
    print(f"  {'─'*40}")

    all_groups = ['embed', 'attention', 'transformer', 'cnn', 'projection', 'head']
    for g in all_groups:
        count = layer_stats.get(g, 0)
        if g in trainable_groups:
            status = "🔓 训练"
        else:
            status = "🔒 冻结"
        if count > 0 or g == 'head':
            print(f"  {g:<20} {status:<8} {count:>12,}")

    print(f"  {'─'*40}")
    print(f"  {'总计':<20} {'':<8} {total:>12,}")
    print(f"  可训练: {trainable_count/total*100:.1f}% | 冻结: {frozen_count/total*100:.1f}%")
    print(f"{'─'*50}\n")

    return trainable_groups


def build_incremental_optimizer(model, config, trainable_groups=None):
    """
    增量学习分层优化器。
    根据可训练的层组分配不同学习率：
      - 分类头(head):    基础学习率 (大)
      - CNN/Transformer:  基础学习率 × backbone_lr_scale (中)
      - embed/attention/projection: 基础学习率 × backbone_lr_scale (小，仅all模式下可训练)
    """
    base_lr = config['learning_rate']
    backbone_scale = config.get('backbone_lr_scale', 0.1)

    head_params = []
    cnn_transformer_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = get_layer_group(name)
        if group == 'head':
            head_params.append(param)
        elif group in ('cnn', 'transformer'):
            cnn_transformer_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})
    if cnn_transformer_params:
        param_groups.append({"params": cnn_transformer_params,
                             "lr": base_lr * backbone_scale})
    if other_params:
        param_groups.append({"params": other_params,
                             "lr": base_lr * backbone_scale * 0.5})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.get('weight_decay', 1e-5),
        betas=(config.get('betas_1', 0.9), config.get('betas_2', 0.999))
    )
    return optimizer


def custom_multitask_loss(outputs, row_labels, col_labels, model, num_rows, num_cols, row_class_weights=None, col_class_weights=None, eps=0.1, device="cuda"):
    """
    多任务分类损失：标签平滑 + 类别权重
    :param outputs: 模型输出，联合模式 [B, R, C]；独立模式 (row_logits, col_logits)
    :param row_labels: 行标签 [B]
    :param col_labels: 列标签 [B]
    :param model: 模型实例，判断joint_classification
    :param num_rows: 行类别总数
    :param num_cols: 列类别总数
    :param row_class_weights: 行类别均衡权重
    :param col_class_weights: 列类别均衡权重
    :param eps: 标签平滑系数
    :param device: 计算设备
    :return: 加权多任务总损失标量
    """
    if model.joint_classification:
        row_probs = torch.max(outputs, dim=2)[0]
        col_probs = torch.max(outputs, dim=1)[0]
        row_logits = F.log_softmax(row_probs, dim=1)
        col_logits = F.log_softmax(col_probs, dim=1)
    else:
        row_logits, col_logits = outputs
        row_logits = F.log_softmax(row_logits, dim=1)
        col_logits = F.log_softmax(col_logits, dim=1)

    # 标签平滑生成
    def label_smoothing(labels, num_classes):
        one_hot = F.one_hot(labels, num_classes).float()
        return (1 - eps) * one_hot + eps / num_classes

    row_smooth = label_smoothing(row_labels, num_rows).to(device)
    col_smooth = label_smoothing(col_labels, num_cols).to(device)

    # 逐样本交叉熵
    row_loss = -torch.sum(row_smooth * row_logits, dim=1)
    col_loss = -torch.sum(col_smooth * col_logits, dim=1)

    # 类别权重加权
    if row_class_weights is not None:
        row_loss = row_loss * row_class_weights[row_labels]
    if col_class_weights is not None:
        col_loss = col_loss * col_class_weights[col_labels]

    # 行、列损失各0.5加权求和
    loss = 0.5 * row_loss.mean() + 0.5 * col_loss.mean()
    return loss


# ==================== 辅助评估函数 ====================

def evaluate_on_dataset(model, dataloader, device, config, desc="评估"):
    """
    在给定数据集上评估模型准确率（用于监控旧知识遗忘程度）

    :return: dict 包含 loss, acc, row_acc, col_acc
    """
    model.eval()
    total_loss = 0.0
    total_correct = total_row_correct = total_col_correct = total_samples = 0

    with torch.no_grad():
        for signals, masks, labels, _ in tqdm(dataloader, desc=desc, leave=False):
            signals = signals.to(device)
            masks = masks.to(device)
            row_labels = labels[:, 0].to(device)
            col_labels = labels[:, 1].to(device)
            batch_size = signals.shape[0]

            outputs = model(signals, masks)
            loss = custom_multitask_loss(
                outputs=outputs,
                row_labels=row_labels,
                col_labels=col_labels,
                model=model,
                num_rows=config["num_rows"],
                num_cols=config["num_cols"],
                eps=config.get("label_smoothing", 0.1),
                device=device
            )

            if model.joint_classification:
                row_probs = torch.max(outputs, dim=2)[0]
                col_probs = torch.max(outputs, dim=1)[0]
                _, row_preds = torch.max(row_probs, dim=1)
                _, col_preds = torch.max(col_probs, dim=1)
            else:
                row_logits, col_logits = outputs
                _, row_preds = torch.max(row_logits, 1)
                _, col_preds = torch.max(col_logits, 1)

            total_loss += loss.item() * batch_size
            total_row_correct += (row_preds == row_labels).sum().item()
            total_col_correct += (col_preds == col_labels).sum().item()
            total_correct += ((row_preds == row_labels) & (col_preds == col_labels)).sum().item()
            total_samples += batch_size

    return {
        'loss': total_loss / total_samples if total_samples else 0,
        'acc': total_correct / total_samples if total_samples else 0,
        'row_acc': total_row_correct / total_samples if total_samples else 0,
        'col_acc': total_col_correct / total_samples if total_samples else 0,
    }


# ==================== 增量学习主函数 ====================

def incremental_learning_curve(plane_pretrain_path, curve_train_loader,
                                curve_val_loader, plane_train_loader=None):
    """
    增量学习主函数：基于平面预训练模型，通过曲面数据增量学习，
    同时通过 EWC + 知识蒸馏 + 记忆重放 防止灾难性遗忘。

    训练流程:
    ┌─────────────────────────────────────────────────────┐
    │ 1. 加载平面预训练模型 → 学生模型 (待训练)            │
    │ 2. 复制一份冻结 → 教师模型 (蒸馏参考)                │
    │ 3. 在平面数据上计算 Fisher 矩阵 → EWC 惩罚依据       │
    │ 4. 填充平面样本到重放缓冲区 → 复习用                 │
    │                                                       │
    │ 每个训练 step:                                       │
    │   ├─ 曲面分类损失 (学习新知识)                       │
    │   ├─ EWC 惩罚 (保护旧任务重要权重)                   │
    │   ├─ KD 蒸馏损失 (保留旧任务决策边界)                │
    │   └─ 重放蒸馏损失 (复习平面数据)                     │
    │                                                       │
    │ 每个 epoch 后:                                       │
    │   ├─ 曲面验证集评估 (监控新知识学习)                 │
    │   └─ 平面验证集评估 (监控旧知识遗忘)                 │
    └─────────────────────────────────────────────────────┘

    :param plane_pretrain_path: 平面预训练权重路径
    :param curve_train_loader: 曲面微调训练集
    :param curve_val_loader: 曲面验证集
    :param plane_train_loader: 平面数据加载器（用于EWC计算和重放）
    :return: 增量学习后的模型
    """
    device = config['device']
    joint_mode = config["joint_mode"]
    num_rows = config["num_rows"]
    num_cols = config["num_cols"]

    # ========== 阶段1: 初始化学生模型并加载平面预训练权重 ==========
    print("\n" + "="*60)
    print("🚀 增量学习初始化")
    print("="*60)

    model = SuperResolutionSensorModel(
        num_rows=num_rows, num_cols=num_cols,
        num_channels=config["num_channels"], seq_len=config["seq_len"],
        joint_classification=joint_mode
    )
    pretrain_dict = torch.load(plane_pretrain_path, map_location=device)
    model.load_state_dict(pretrain_dict, strict=True)
    model = model.to(device)
    print(f"✅ 学生模型: 已加载平面预训练权重 {plane_pretrain_path}")

    # ========== 阶段2: 创建冻结的教师模型（用于知识蒸馏） ==========
    use_kd = config.get('use_kd', True)
    teacher_model = None
    if use_kd:
        teacher_model = SuperResolutionSensorModel(
            num_rows=num_rows, num_cols=num_cols,
            num_channels=config["num_channels"], seq_len=config["seq_len"],
            joint_classification=joint_mode
        )
        teacher_model.load_state_dict(pretrain_dict, strict=True)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        # 冻结教师模型所有参数
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("✅ 教师模型: 已冻结 (用于知识蒸馏)")

    # ========== 阶段3: 初始化增量学习组件 ==========
    # 3a. EWC: 计算Fisher信息矩阵
    ewc = None
    use_ewc = config.get('use_ewc', True)
    if use_ewc and plane_train_loader is not None:
        ewc = EWC(model, plane_train_loader, config, device)
    elif use_ewc and plane_train_loader is None:
        print("⚠️ EWC需要平面数据但未提供，已禁用EWC")

    # 3b. 重放缓冲区: 用平面样本预填充
    replay_buffer = None
    use_replay = config.get('use_replay', True)
    if use_replay and plane_train_loader is not None:
        replay_buffer = PlaneReplayBuffer(
            max_samples=config.get('replay_buffer_size', 1000)
        )
        replay_buffer.fill(plane_train_loader, device)
    elif use_replay and plane_train_loader is None:
        print("⚠️ 重放缓冲区需要平面数据但未提供，已禁用Replay")

    # 3c. 解冻策略：先按配置全部放开，建优化器，再回退到head_only做warmup
    unfreeze_layers = config.get("unfreeze_layers", "head+cnn+transformer")

    # 先按目标配置解冻所有需要的层，创建优化器（保证optimizer覆盖所有潜在可训练参数）
    apply_layer_freeze(model, unfreeze_layers=unfreeze_layers)
    optimizer = build_incremental_optimizer(model, config)

    # 3d. 学习率调度器
    scheduler = WarmupReduceLROnPlateau(
        optimizer,
        warmup_epochs=config['lr_warmup_epochs'],
        init_lr=1e-5,
        mode='min', factor=0.7, patience=10,
        threshold=1e-4, min_lr=1e-6
    )

    # ========== 阶段4: 训练状态初始化 ==========
    best_val_acc = 0.0
    best_plane_acc = 0.0  # 跟踪平面准确率（检测遗忘）
    patience_counter = 0
    train_loss_smoother = RunningAverage(window_size=8)
    val_loss_smoother = RunningAverage(window_size=8)
    val_acc_smoother = RunningAverage(window_size=8)

    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [], 'train_row_acc': [],
        'train_col_acc': [],
        'val_loss': [], 'val_acc': [], 'val_row_acc': [], 'val_col_acc': [],
        'plane_loss': [], 'plane_acc': [], 'plane_row_acc': [], 'plane_col_acc': [],
        'cls_loss': [], 'ewc_loss': [], 'kd_loss': [], 'replay_loss': [],
        'learning_rate': []
    }

    freeze_epoch = config["freeze_backbone_epochs"]
    num_epochs = config["num_epochs"]
    kd_temp = config.get('kd_temperature', 4.0)
    kd_weight = config.get('kd_weight', 0.5)
    replay_weight = config.get('replay_weight', 0.3)

    # 是否有平面数据用于验证遗忘程度
    has_plane_val = (plane_train_loader is not None)

    print(f"\n{'='*60}")
    print(f"📋 增量学习配置总览")
    print(f"{'='*60}")
    print(f"  模式: incremental (EWC + KD + Replay)")
    print(f"  EWC: {'✓' if ewc and ewc.ewc_enabled else '✗'} | λ={config.get('ewc_lambda', 500)}")
    print(f"  KD:  {'✓' if use_kd else '✗'} | T={kd_temp} | weight={kd_weight}")
    print(f"  Replay: {'✓' if replay_buffer else '✗'} | "
          f"buffer={len(replay_buffer) if replay_buffer else 0} | weight={replay_weight}")
    print(f"  解冻策略: head_only(前{freeze_epoch}轮) → {unfreeze_layers}(后续)")
    print(f"  总轮数={num_epochs}")
    print(f"{'='*60}\n")

    # ========== 阶段5: 增量训练主循环 ==========
    for epoch in range(num_epochs):
        # --- 分阶段解冻策略 ---
        if epoch == 0:
            # 第一轮：强制只训练分类头
            apply_layer_freeze(model, unfreeze_layers="head_only")
            print(f"🔒 Epoch 1-{freeze_epoch}: 仅训练分类头(head_only)")

        if epoch == freeze_epoch:
            # 解冻目标层：按配置选择性解冻
            apply_layer_freeze(model, unfreeze_layers=unfreeze_layers)
            print(f"🔓 Epoch {epoch+1}+: 解冻目标层 → {unfreeze_layers}")

        model.train()

        # 训练统计量
        total_cls_loss = 0.0    # 曲面分类损失
        total_ewc_loss = 0.0    # EWC正则化损失
        total_kd_loss = 0.0     # 知识蒸馏损失
        total_replay_loss = 0.0 # 重放蒸馏损失
        train_correct = train_row_correct = train_col_correct = train_total = 0

        pbar = tqdm(curve_train_loader, desc=f"增量学习 Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (signals, masks, labels, _) in enumerate(pbar):
            signals = signals.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            row_labels = labels[:, 0].to(device)
            col_labels = labels[:, 1].to(device)
            batch_size = signals.shape[0]

            optimizer.zero_grad(set_to_none=True)

            # ---- 损失1: 曲面分类损失（学习新知识） ----
            outputs = model(signals, masks)
            cls_loss = custom_multitask_loss(
                outputs=outputs, row_labels=row_labels, col_labels=col_labels,
                model=model, num_rows=num_rows, num_cols=num_cols,
                eps=config["label_smoothing"], device=device
            )

            total_loss = cls_loss
            cls_loss_val = cls_loss.item()

            # ---- 损失2: EWC惩罚（保护旧任务重要权重） ----
            ewc_loss_val = 0.0
            if ewc is not None:
                ewc_loss = ewc.penalty(model)
                total_loss = total_loss + ewc_loss
                ewc_loss_val = ewc_loss.item()

            # ---- 损失3: 知识蒸馏（保留旧任务决策边界） ----
            kd_loss_val = 0.0
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(signals, masks)
                kd_loss = knowledge_distillation_loss(
                    student_outputs=outputs,
                    teacher_outputs=teacher_outputs,
                    model=model, T=kd_temp,
                    joint_classification=joint_mode
                )
                total_loss = total_loss + kd_weight * kd_loss
                kd_loss_val = kd_loss.item()

            # ---- 损失4: 记忆重放（复习平面数据防遗忘） ----
            replay_loss_val = 0.0
            if replay_buffer is not None and teacher_model is not None:
                # 从缓冲区采样平面数据（采样数量=当前曲面batch的一半）
                replay_n = max(1, batch_size // 2)
                plane_sig, plane_mask, _ = replay_buffer.sample(replay_n, device)
                if plane_sig is not None:
                    # 学生模型对平面数据的预测
                    student_plane_out = model(plane_sig, plane_mask)
                    # 教师模型对平面数据的预测（作为软目标）
                    with torch.no_grad():
                        teacher_plane_out = teacher_model(plane_sig, plane_mask)
                    replay_loss = knowledge_distillation_loss(
                        student_outputs=student_plane_out,
                        teacher_outputs=teacher_plane_out,
                        model=model, T=kd_temp,
                        joint_classification=joint_mode
                    )
                    total_loss = total_loss + replay_weight * replay_loss
                    replay_loss_val = replay_loss.item()

            # ---- 反向传播与优化 ----
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()

            # ---- 统计曲面分类准确率 ----
            if model.joint_classification:
                row_probs = torch.max(outputs, dim=2)[0]
                col_probs = torch.max(outputs, dim=1)[0]
                _, row_preds = torch.max(row_probs, dim=1)
                _, col_preds = torch.max(col_probs, dim=1)
            else:
                row_logits, col_logits = outputs
                _, row_preds = torch.max(row_logits, 1)
                _, col_preds = torch.max(col_logits, 1)

            batch_correct = ((row_preds == row_labels) & (col_preds == col_labels)).sum().item()
            batch_row_correct = (row_preds == row_labels).sum().item()
            batch_col_correct = (col_preds == col_labels).sum().item()

            total_cls_loss += cls_loss_val * batch_size
            total_ewc_loss += ewc_loss_val
            total_kd_loss += kd_loss_val
            total_replay_loss += replay_loss_val
            train_correct += batch_correct
            train_row_correct += batch_row_correct
            train_col_correct += batch_col_correct
            train_total += batch_size

            # 进度条显示
            if batch_idx % 5 == 0:
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'cls': f'{cls_loss_val:.4f}',
                    'ewc': f'{ewc_loss_val:.4f}',
                    'kd': f'{kd_loss_val:.4f}',
                    'replay': f'{replay_loss_val:.4f}',
                    'acc': f'{batch_correct/batch_size:.4f}',
                    'LR': f'{lr:.2e}'
                })

        # ==================== Epoch 结束：指标汇总 ====================

        # 曲面训练集指标
        avg_cls_loss = total_cls_loss / train_total if train_total else 0
        avg_ewc_loss = total_ewc_loss / len(curve_train_loader)
        avg_kd_loss = total_kd_loss / len(curve_train_loader)
        avg_replay_loss = total_replay_loss / len(curve_train_loader)
        train_acc = train_correct / train_total if train_total else 0
        train_row_acc = train_row_correct / train_total if train_total else 0
        train_col_acc = train_col_correct / train_total if train_total else 0
        smoothed_train_loss = train_loss_smoother.update(avg_cls_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 曲面验证集评估
        curve_val_metrics = evaluate_on_dataset(
            model, curve_val_loader, device, config, desc="曲面验证"
        )

        smoothed_val_loss = val_loss_smoother.update(curve_val_metrics['loss'])
        smoothed_val_acc = val_acc_smoother.update(curve_val_metrics['acc'])

        # 平面验证集评估（监控遗忘 — 用训练集的另一部分做验证）
        plane_val_metrics = {'loss': 0, 'acc': 0, 'row_acc': 0, 'col_acc': 0}
        if has_plane_val and epoch % 5 == 0:  # 每5轮评估一次平面性能（节省时间）
            plane_val_metrics = evaluate_on_dataset(
                model, plane_train_loader, device, config, desc="平面遗忘检测"
            )

        # ==================== 日志记录 ====================
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_cls_loss)
        history['train_acc'].append(train_acc)
        history['train_row_acc'].append(train_row_acc)
        history['train_col_acc'].append(train_col_acc)
        history['val_loss'].append(curve_val_metrics['loss'])
        history['val_acc'].append(curve_val_metrics['acc'])
        history['val_row_acc'].append(curve_val_metrics['row_acc'])
        history['val_col_acc'].append(curve_val_metrics['col_acc'])
        history['plane_loss'].append(plane_val_metrics['loss'])
        history['plane_acc'].append(plane_val_metrics['acc'])
        history['plane_row_acc'].append(plane_val_metrics['row_acc'])
        history['plane_col_acc'].append(plane_val_metrics['col_acc'])
        history['cls_loss'].append(avg_cls_loss)
        history['ewc_loss'].append(avg_ewc_loss)
        history['kd_loss'].append(avg_kd_loss)
        history['replay_loss'].append(avg_replay_loss)
        history['learning_rate'].append(current_lr)

        # ==================== 控制台输出 ====================
        print(f"\n{'─'*50}")
        print(f"📌 增量学习 Epoch {epoch+1}/{num_epochs}")
        print(f"  曲面训练 ── Loss:{avg_cls_loss:.4f} Acc:{train_acc:.4f} "
              f"行{train_row_acc:.4f} 列{train_col_acc:.4f}")
        print(f"  曲面验证 ── Loss:{curve_val_metrics['loss']:.4f} "
              f"Acc:{curve_val_metrics['acc']:.4f} "
              f"行{curve_val_metrics['row_acc']:.4f} 列{curve_val_metrics['col_acc']:.4f}")
        if has_plane_val and epoch % 5 == 0:
            print(f"  平面遗忘监控 ── Acc:{plane_val_metrics['acc']:.4f} "
                  f"行{plane_val_metrics['row_acc']:.4f} 列{plane_val_metrics['col_acc']:.4f}")
        print(f"  损失组件 ── cls:{avg_cls_loss:.4f} ewc:{avg_ewc_loss:.4f} "
              f"kd:{avg_kd_loss:.4f} replay:{avg_replay_loss:.4f}")
        print(f"  学习率:{current_lr:.2e}")

        # ==================== 学习率更新 ====================
        scheduler.step(smoothed_val_loss)

        # ==================== 最优模型保存 ====================
        is_best = False
        acc_improvement = smoothed_val_acc - best_val_acc
        if acc_improvement > config['acc_improve_threshold']:
            best_val_acc = smoothed_val_acc
            best_plane_acc = plane_val_metrics['acc']
            is_best = True
            patience_counter = 0
            print(f"📈 曲面验证准确率新高 → {best_val_acc:.4f} "
                  f"(平面准确率: {best_plane_acc:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ 无提升，早停计数 {patience_counter}/{config['patience']}")

        if is_best:
            save_path = os.path.join(config['result_save_dir'],
                                     "best_incremental_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存增量学习最优模型至 {save_path}")

        if patience_counter >= config['patience']:
            print(f"\n🛑 早停触发！最优曲面验证准确率 {best_val_acc:.4f} "
                  f"(对应平面准确率 {best_plane_acc:.4f})")
            break

    # ==================== 训练结束：保存日志与最终评估 ====================
    df = pd.DataFrame(history)
    csv_path = os.path.join(config['result_save_dir'],
                            "incremental_learning_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📁 训练历史已保存至 {csv_path}")

    # 最终平面性能评估（完整评估遗忘程度）
    if has_plane_val:
        final_plane = evaluate_on_dataset(
            model, plane_train_loader, device, config, desc="最终平面评估"
        )
        print(f"\n{'='*50}")
        print(f"🏁 增量学习完成 - 最终评估")
        print(f"  曲面验证准确率: {best_val_acc:.4f}")
        print(f"  平面数据准确率: {final_plane['acc']:.4f} "
              f"(行{final_plane['row_acc']:.4f} 列{final_plane['col_acc']:.4f})")
        print(f"  旧知识保留率: {final_plane['acc']/best_plane_acc*100:.1f}% "
              f"(相对训练过程中的最佳平面准确率)" if best_plane_acc > 0 else "")
        print(f"{'='*50}")

    try:
        plot_training_curves(df, csv_path=csv_path)
        print("📊 增量学习训练曲线已生成")
    except Exception as e:
        print(f"曲线绘制失败: {e}")

    return model


if __name__ == "__main__":
    # 环境初始化
    def setup_environment():
        if torch.cuda.is_available():
            gpu_id = config.get('gpu_id', 0)
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            print(f"⚡ 使用GPU: {torch.cuda.get_device_name(gpu_id)}")
            device = torch.device(f"cuda:{gpu_id}")
        else:
            print("使用CPU训练")
            device = torch.device('cpu')
        return device
    device = setup_environment()

    # 1. 加载曲面微调数据集
    curve_train_loader, curve_val_loader = create_curve_finetune_loaders(
        curve_h5_path=config["curve_h5_path"],
        batch_size=config.get("batch_size", 128),
        full_load=config.get("full_load", False)
    )

    # 2. 加载平面训练集（用于 EWC / 知识蒸馏 / 记忆重放）
    H5_PATH = config["train_h5_path"]
    assert os.path.exists(H5_PATH), f"平面数据集不存在: {H5_PATH}"
    plane_train_loader, _ = create_plane_full_loaders(
        H5_PATH, batch_size=config["batch_size"], full_load=config["full_load"]
    )
    print(f"✅ 平面全量数据集已加载: {H5_PATH}")

    # 3. 启动增量学习
    print("\n" + "="*60)
    print("🎯 启动增量学习模式 (EWC + 知识蒸馏 + 记忆重放)")
    print("="*60)
    finetuned_model = incremental_learning_curve(
        plane_pretrain_path=config["pretrain_weights_path"],
        curve_train_loader=curve_train_loader,
        curve_val_loader=curve_val_loader,
        plane_train_loader=plane_train_loader
    )
    print("🎉 曲面数据增量学习全部完成！")