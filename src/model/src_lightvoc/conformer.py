import torch.nn as nn


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d_model]
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d_model]
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, d_model, T]
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, d_model]
        return x + residual


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d_model] -> требуется [T, B, d_model] для self_attn
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(0, 1)  # [T, B, d_model]
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        x = attn_output.transpose(0, 1)  # [B, T, d_model]
        return x + residual


class ConformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, d_model]
        x = self.ffn1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.final_layer_norm(x)
        return x
