"""
损失函数模块
依据 PROJECT.md 第3章实现
"""
import torch
import torch.nn as nn
class MultiTaskLoss(nn.Module):
    """多任务损失函数，支持独立和联合分类模式"""

    def __init__(self, row_weights=None, col_weights=None, joint_classification=True):
        super().__init__()
        self.joint_classification = joint_classification
        if joint_classification:
            self.joint_loss = nn.CrossEntropyLoss()
        else:
            self.row_loss = nn.CrossEntropyLoss(weight=row_weights)
            self.col_loss = nn.CrossEntropyLoss(weight=col_weights)

    def forward(self, outputs, row_labels, col_labels):
        if self.joint_classification:
            # 联合分类模式
            joint_labels = row_labels * outputs.size(2) + col_labels  # 将(row,col)转换为单一标签
            loss = self.joint_loss(outputs.view(outputs.size(0), -1), joint_labels)
            return loss
        else:
            # 独立分类模式
            row_loss = self.row_loss(outputs[0], row_labels)
            col_loss = self.col_loss(outputs[1], col_labels)
            return row_loss + col_loss

# class MultiTaskLoss(nn.Module):
#     def __init__(self, joint_classification=True, num_rows=8, num_cols=8):
#         super().__init__()
#         self.joint_classification = joint_classification
#         self.num_rows = num_rows
#         self.num_cols = num_cols
#         self.cross_entropy = nn.CrossEntropyLoss()
#
#     def forward(self, outputs, row_labels, col_labels):
#         if self.joint_classification:
#             # 联合模式：outputs [B, R, C] → 展平为[B, R*C]
#             outputs_flat = outputs.reshape(-1, self.num_rows * self.num_cols)
#             # 标签：row_labels[B] + col_labels[B] → 展平为一维标签[B]（如row=2,col=3 → 2*8+3=19）
#             joint_labels = row_labels * self.num_cols + col_labels
#             return self.cross_entropy(outputs_flat, joint_labels)
#         else:
#             # 独立模式：分别计算行/列损失并求和
#             row_logits, col_logits = outputs
#             loss_row = self.cross_entropy(row_logits, row_labels)
#             loss_col = self.cross_entropy(col_logits, col_labels)
#             return (loss_row + loss_col) / 2  # 平均损失，平衡行/列学习

class RegressionLoss(nn.Module):  # 回归损失函数
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()  # 连续值回归用MSE

    def forward(self, outputs, true_coords):
        # outputs: [B, 2]（预测x/y）；true_coords: [B, 2]（真实x/y，小数）
        return self.mse_loss(outputs, true_coords)