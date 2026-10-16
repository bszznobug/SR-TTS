import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):  # d_model=32（embed_dim//2），max_len=100（时序长度）
        super().__init__()
        # 1. 生成位置索引：[100] → [100,1]（每个时间点对应一个位置编号0-99）
        position = torch.arange(max_len).unsqueeze(1)
        # 2. 生成频率分母：遵循Transformer原论文的公式，让不同位置有不同频率的正弦/余弦编码
        # 公式：div_term = exp(2i * -ln(10000)/d_model)，i是维度索引
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 3. 初始化位置编码矩阵：[100, 32]
        pe = torch.zeros(max_len, d_model)
        # 4. 偶数维度用正弦编码，奇数维度用余弦编码（互补编码，覆盖更多位置特征）
        pe[:, 0::2] = torch.sin(position * div_term)  # 0,2,4...维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 1,3,5...维度
        # 5. 注册为缓冲区（不参与训练，仅作为固定参数）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x形状：[B*C, T, 32] → 取前T个时间点的位置编码，加到输入上
        return x + self.pe[:x.size(1)]


class ChannelAttention(nn.Module):
    def __init__(self, num_channels=32, embed_dim=64, reduction_ratio=4):
        super().__init__()
        self.embed_dim = 64  # 注意力输出维度，与后续transformer的输入维度对齐
        # 压缩-激活-恢复：降维后计算注意力，减少计算量
        self.reduction = nn.Sequential(
            # 32 → 8（32/4）：降维，捕捉通道间的非线性关系
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(),  # 引入非线性，增强特征表达
            # 8 → 64：映射到注意力维度，为后续加权做准备
            nn.Linear(num_channels // reduction_ratio, embed_dim))
        self.softmax = nn.Softmax(dim=-1)  # 归一化注意力权重

    def forward(self, x, mask=None):
        B, C, T = x.size()  # [B,32,100]
        # 1. 全局平均池化：压缩时序维度，得到每个通道的全局特征 [B,32,100] → [B,32]
        x_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        # 2. 掩码处理：无效通道的特征置0（比如故障传感器）
        if mask is not None:
            x_pool = x_pool * mask.float()
        # 3. 计算通道注意力权重：[B,32] → [B,64]
        attn = self.reduction(x_pool)
        # 4. 归一化：权重之和=1，确保注意力的可解释性
        attn = self.softmax(attn)
        # 5. 加权求和：用注意力权重对原始通道信号加权 [B,64] × [B,32,100] → [B,64,100]
        # einsum解释：be(batch,embed) × bct(batch,channel,time) → bet(batch,embed,time)
        output = torch.einsum('be,bct->bet', attn, x)
        return output
class SuperResolutionSensorModel(nn.Module):
    """分类任务模型，改进的超分辨传感器定位模型，支持独立和联合分类模式"""

    def __init__(self, num_rows=8, num_cols=8, num_channels=32, seq_len=100, joint_classification=True):
        super().__init__()
        self.embed_dim = 64
        self.num_heads = 4
        self.num_layers = 3
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.joint_classification = joint_classification  # 是否使用联合分类模式

        # 1. 输入处理层
        self.channel_embed = nn.Linear(1, self.embed_dim // 2)
        self.pos_encoder = PositionalEncoding(self.embed_dim // 2, seq_len)

        # 2. 通道注意力层
        self.channel_attention = ChannelAttention(num_channels, self.embed_dim)

        # 3. 时间处理层 (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        # 4. 空间特征提取 (CNN)
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(self.embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 5. 分类头
        if self.joint_classification:
            # 联合分类模式: 输出所有可能的(row,col)组合
            self.joint_head = nn.Linear(256, num_rows * num_cols)
        else:
            # 独立分类模式
            self.row_head = nn.Linear(256, num_rows)
            self.col_head = nn.Linear(256, num_cols)

        # 投影层
        self.projection = nn.Linear(self.embed_dim // 2 * num_channels + self.embed_dim, self.embed_dim)

    def forward(self, x, mask):
        B, C, T = x.size()  # 获取信号输入的形状[B, 32, 100]

        x_embedded = self.channel_embed(x.permute(0, 2, 1).unsqueeze(-1))  # [B, T, C, E//2]
        x_embedded = x_embedded.permute(0, 2, 1, 3)  # [B, C, T, E//2]
        x_embedded = x_embedded.reshape(B * C, T, -1)  # [B*C, T, E//2]
        x_embedded = self.pos_encoder(x_embedded)
        x_embedded = x_embedded.reshape(B, C, T, -1)  # [B, C, T, E//2]
        x_embedded = x_embedded.permute(0, 3, 1, 2)  # [B, E//2, C, T]
        x_embedded = x_embedded.reshape(B, -1, T)  # [B, E//2*C, T]

        # 2. 应用通道注意力（给32个通道进行打分加权，输出64维，对接transformer）
        channel_features = self.channel_attention(x, mask)  # [B, E, T]

        # 3. 结合特征
        combined = torch.cat([x_embedded, channel_features], dim=1)  # [B, E//2*C + E, T]
        combined = combined.permute(0, 2, 1)  # [B, T, E//2*C + E]
        projected = self.projection(combined)  # [B, T, E]

        # 4. 时间处理（使用transformer，可以建模任意时间点的关系）
        temporal_features = self.temporal_encoder(projected)  # [B, T, E]
        temporal_features = temporal_features.permute(0, 2, 1)  # [B, E, T]

        # 5. 空间特征提取和分类预测
        spatial_features = self.spatial_cnn(temporal_features)
        spatial_features = spatial_features.squeeze(-1)

        if self.joint_classification:
            # 联合分类模式
            joint_logits = self.joint_head(spatial_features)  # [B, num_rows*num_cols]
            joint_logits = joint_logits.view(B, self.num_rows, self.num_cols)  # [B, num_rows, num_cols]
            return joint_logits
        else:
            # 独立分类模式
            row_logits = self.row_head(spatial_features)  # [B, num_rows]
            col_logits = self.col_head(spatial_features)  # [B, num_cols]
            return row_logits, col_logits

# class SuperResolutionSensorModel(nn.Module):
#     """分类任务模型，改进的超分辨传感器定位模型，支持独立和联合分类模式"""
#
#     def __init__(self, num_rows=8, num_cols=8, num_channels=32, seq_len=100, joint_classification=True):
#         super().__init__()
#         self.embed_dim = 64
#         self.num_heads = 4
#         self.num_layers = 3
#         self.num_rows = num_rows
#         self.num_cols = num_cols
#         self.num_channels = num_channels
#         self.seq_len = seq_len
#         self.joint_classification = joint_classification  # 是否使用联合分类模式
#
#         # 1. 输入处理层
#         self.channel_embed = nn.Linear(1, self.embed_dim // 2)  # 从单通道信号，映射到高位空间（嵌入），能够增强特征表达能力
#         self.pos_encoder = PositionalEncoding(self.embed_dim // 2, seq_len)
#
#         # 2. 通道注意力层
#         self.channel_attention = ChannelAttention(num_channels, self.embed_dim)
#
#         # 3. 时间处理层 (Transformer)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim,
#             nhead=self.num_heads,  # 注意力头，4个头可以分别关注不同的时序模式，每个头可以处理16维特征
#             dim_feedforward=256,   # 前馈网络维度，通常设置为d_model的4倍
#             dropout=0.1,
#             batch_first=True
#         )
#         self.temporal_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)  # num_layers=3：3 层编码器堆叠，逐层提取更抽象的时序特征（1 层太浅，5 层以上易过拟合）
#
#         # 4. 空间特征提取 (CNN)
#         self.spatial_cnn = nn.Sequential(
#             # 64→128维，卷积核3（捕捉相邻3个时间点的局部特征），padding=1保持时序长度不变
#             nn.Conv1d(self.embed_dim, 128, kernel_size=3, padding=1),  # kernel_size=3：1D 卷积核大小，适合捕捉时序信号的局部连续特征（比如 3 个时间点的趋势）
#                                                                                   # padding=1：补零，保证卷积后时序长度不变（100→100）
#             nn.BatchNorm1d(128),   # 批量归一化，加速训练，防止梯度消失
#             nn.ReLU(),             # 引入非线性
#             # 128→256维，进一步提取深层特征
#             nn.Conv1d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             # 全局平均池化：[B,256,100] → [B,256,1]（压缩时序维度，得到全局特征）
#             nn.AdaptiveAvgPool1d(1)     # 自适应池化，无论输入时序长度是多少，都输出 1 维，增强鲁棒性
#         )
#
#         # 5. 分类头
#         # 无激活函数：坐标是连续值（比如 x∈[0,8], y∈[0,8]），不需要 sigmoid/softmax 等激活函数限制输出范围
#         # 256→2：将高维特征映射到最终的坐标输出，是回归任务的标准设计
#         if self.joint_classification:
#             # 联合分类模式: 输出所有可能的(row,col)组合
#             self.joint_head = nn.Linear(256, num_rows * num_cols)
#         else:
#             # 独立分类模式
#             self.row_head = nn.Linear(256, num_rows)
#             self.col_head = nn.Linear(256, num_cols)
#
#         # 投影层
#         self.projection = nn.Linear(self.embed_dim // 2 * num_channels + self.embed_dim, self.embed_dim)
#
#     def forward(self, x, mask):
#         B, C, T = x.size()  # 获取信号输入的形状[B, 32, 100]
#
#         # 1. 通道嵌入和时间位置编码（将单个时间点的数据变成特征+位置编码）
#         # 维度变换 + 嵌入：[B,32,100] → [B,100,32] → [B,100,32,1] → [B,100,32,32]
#         x_embedded = self.channel_embed(x.permute(0, 2, 1).unsqueeze(-1))  # [B, T, C, E//2]
#         # 调整维度：[B, 100, 32, 32] → [B, 32, 100, 32]（通道维度提前）
#         x_embedded = x_embedded.permute(0, 2, 1, 3)  # [B, C, T, E//2]
#         # 展平适配位置编码：[B, 32, 100, 32] → [B * 32, 100, 32]（把每个通道当成独立样本）
#         x_embedded = x_embedded.reshape(B * C, T, -1)  # [B*C, T, E//2]

#         # 注入位置信息：[B*32,100,32] → 每个通道的时序信号都带上时间位置
#         x_embedded = self.pos_encoder(x_embedded)
#         # 恢复维度：[B*32,100,32] → [B,32,100,32]
#         x_embedded = x_embedded.reshape(B, C, T, -1)  # [B, C, T, E//2]
#         # 展平通道+嵌入维度：[B,32,100,32] → [B,32*32,100] = [B,1024,100]
#         x_embedded = x_embedded.permute(0, 3, 1, 2)  # [B, E//2, C, T]
#         x_embedded = x_embedded.reshape(B, -1, T)  # [B, E//2*C, T]
#
#         # 2. 应用通道注意力（给32个通道进行打分加权，输出64维，对接transformer）
#         channel_features = self.channel_attention(x, mask)  # [B, E, T]
#
#         # 3. 结合特征
#         # 拼接通道嵌入特征（时间特征）和注意力特征（空间特征）
#         combined = torch.cat([x_embedded, channel_features], dim=1)  # [B, E//2*C + E, T]
#         combined = combined.permute(0, 2, 1)  # [B, T, E//2*C + E]
#         # 投影到64维
#         projected = self.projection(combined)  # [B, T, E]
#
#         # 4. 时间处理（使用transformer，可以建模任意时间点的关系）
#         # 时序编码：[B,100,64] → [B,100,64]（捕捉时间点之间的依赖关系）
#         temporal_features = self.temporal_encoder(projected)  # [B, T, E]
#         # 维度调整：[B,100,64] → [B,64,100]（适配后续CNN的输入格式）
#         temporal_features = temporal_features.permute(0, 2, 1)  # [B, E, T]
#
#         # 5. 空间特征提取和分类预测
#         # CNN提取特征：[B,64,100] → [B,256,1]，将transformer输出的每个时间点的特征通过CNN来捕捉，形成全局特征
#         spatial_features = self.spatial_cnn(temporal_features)
#         # 移除冗余维度：[B,256,1] → [B,256]
#         spatial_features = spatial_features.squeeze(-1)
#
#         if self.joint_classification:
#             # 联合分类模式
#             joint_logits = self.joint_head(spatial_features)  # [B, num_rows*num_cols]
#             joint_logits = joint_logits.view(B, self.num_rows, self.num_cols)  # [B, num_rows, num_cols]
#             return joint_logits
#         else:
#             # 独立分类模式
#             row_logits = self.row_head(spatial_features)  # [B, num_rows]
#             col_logits = self.col_head(spatial_features)  # [B, num_cols]
#             return row_logits, col_logits


class SuperResolutionSensorModelRegression(nn.Module):
    """回归任务模型：输入信号，输出连续的(x, y)坐标"""

    def __init__(self, num_channels=32, seq_len=100):
        super().__init__()
        # 模型基础参数（移除分类相关的num_rows/num_cols/joint_classification）
        self.embed_dim = 64  # 特征嵌入维度
        self.num_heads = 4  # Transformer注意力头数
        self.num_layers = 3  # Transformer编码器层数
        self.num_channels = num_channels  # 输入信号通道数
        self.seq_len = seq_len  # 时间序列长度

        # 1. 输入处理：通道嵌入 + 位置编码
        self.channel_embed = nn.Linear(1, self.embed_dim // 2)  # 单通道信号→低维嵌入
        self.pos_encoder = PositionalEncoding(self.embed_dim // 2, seq_len)  # 注入时序位置信息

        # 2. 通道注意力：强化重要通道特征
        self.channel_attention = ChannelAttention(
            num_channels=num_channels,
            embed_dim=self.embed_dim,
            reduction_ratio=4
        )

        # 3. 时序特征提取：Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=256,  # 前馈网络隐藏层维度
            dropout=0.1,  # 防止过拟合
            batch_first=True  # 输入格式：[B, T, d_model]
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 4. 空间特征压缩：1D-CNN + 全局池化
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(self.embed_dim, 128, kernel_size=3, padding=1),  # 时序维度卷积
            nn.BatchNorm1d(128),  # 批量归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # 进一步提取特征
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 压缩为[B, 256, 1]，保留关键特征
        )

        # 5. 回归头：输出2个连续值（x坐标和y坐标），无激活函数（支持任意实数）
        self.regression_head = nn.Linear(256, 2)

        # 特征融合投影层：统一通道嵌入与注意力特征的维度
        self.projection = nn.Linear(
            in_features=(self.embed_dim // 2) * num_channels + self.embed_dim,
            out_features=self.embed_dim
        )

    def forward(self, x, mask):
        """
        前向传播：输入信号→提取特征→输出坐标预测
        x: 输入信号，形状[B, C, T]（B:批次，C:通道数，T:时序长度）
        mask: 掩码，形状[B, C]（用于标记有效通道）
        return: 预测坐标，形状[B, 2]（[x, y]）
        """
        B, C, T = x.size()  # 解析输入形状

        # --------------------------
        # 步骤1：通道嵌入 + 位置编码
        # --------------------------
        # 通道嵌入：将每个通道的时序信号映射到低维空间
        x_embedded = self.channel_embed(x.permute(0, 2, 1).unsqueeze(-1))  # [B, T, C, embed_dim//2]
        x_embedded = x_embedded.permute(0, 2, 1, 3)  # [B, C, T, embed_dim//2]

        # 适配位置编码：展平为[B*C, T, embed_dim//2]
        x_embedded = x_embedded.reshape(B * C, T, -1)
        x_embedded = self.pos_encoder(x_embedded)  # 注入位置信息
        x_embedded = x_embedded.reshape(B, C, T, -1)  # [B, C, T, embed_dim//2]

        # 通道维度展平：[B, embed_dim//2 * C, T]
        x_embedded = x_embedded.permute(0, 3, 1, 2)  # [B, embed_dim//2, C, T]
        x_embedded = x_embedded.reshape(B, -1, T)  # [B, (embed_dim//2)*C, T]

        # --------------------------
        # 步骤2：通道注意力加权
        # --------------------------
        channel_features = self.channel_attention(x, mask)  # [B, embed_dim, T]

        # --------------------------
        # 步骤3：特征融合与维度统一
        # --------------------------
        # 拼接通道嵌入特征和注意力特征：[B, (embed_dim//2*C)+embed_dim, T]
        combined = torch.cat([x_embedded, channel_features], dim=1)
        # 转换为Transformer输入格式：[B, T, (embed_dim//2*C)+embed_dim]
        combined = combined.permute(0, 2, 1)
        # 投影到统一维度：[B, T, embed_dim]
        projected = self.projection(combined)

        # --------------------------
        # 步骤4：时序特征编码（Transformer）
        # --------------------------
        temporal_features = self.temporal_encoder(projected)  # [B, T, embed_dim]
        # 转换为CNN输入格式：[B, embed_dim, T]
        temporal_features = temporal_features.permute(0, 2, 1)

        # --------------------------
        # 步骤5：空间特征压缩 + 回归预测
        # --------------------------
        # CNN提取空间特征并压缩：[B, 256, 1] → [B, 256]
        spatial_features = self.spatial_cnn(temporal_features)
        spatial_features = spatial_features.squeeze(-1)  # 移除冗余维度

        # 回归头输出坐标：[B, 2]（x和y）
        coord_preds = self.regression_head(spatial_features)

        return coord_preds