import math
import sys, os
from functools import partial
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

import equinox as eqx
import igl

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
plt.ion()

import utils
import config
import layers
import fem_model
import system_utils
import main_greedy_cubature

# 移除运行期 NNLS 精修依赖

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, '..')

system_name = "fem"
problem_name = "bunny"
data_dir = "bunny_models/trajectory_data"
pca_basis = "bunny_models/PCA_basis"
output_dir = "bunny_models/transformer_cubature_gamble_softmax"
max_n = 100  # was 300; memory constraint
sub_n = 20
tol_error = 0.01
pca_dim = 131
gradient_weighting = True

# Training params (stability-tuned)
n_epochs = 500
lr = 1e-4
lr_decay_frac = 0.95
lr_decay_every = 50
lr_warm_up_iters = 100

# Transformer size (adapted for RTX 3090 24GB)
BIG_EMBED_DIM = 512
BIG_HIDDEN = [512, 256]
BIG_LAYER_NUM = 8
NUM_HEADS = 16

temperature = 0.2
refine_eval_nnls = False  # 评估阶段禁用 NNLS 精修，统一用 PGD

Train_detailed_log_ENABLE = False  # 新增：默认关闭详细日志

class ModernTransformer(eqx.Module):
    embedding_proj: eqx.Module
    position_embedding: jnp.ndarray
    transformer_blocks: list
    layer_norm_final: eqx.Module
    score_head: eqx.Module
    weight_head: eqx.Module
    dropout_rate: float

    def __init__(self, feature_dim, embed_dim, layer_num, num_heads, key, dropout_rate=0.1, max_seq_len=1000):
        keys = jax.random.split(key, layer_num + 4)
        self.dropout_rate = dropout_rate
        
        # Feature projection (direct)
        self.embedding_proj = layers.create_model({
            'model_type': 'MLP-Layers',
            'activation': 'ELU',
            'in_dim': feature_dim,
            'hidden_layer_sizes': [],  # remove hidden layers, direct projection
            'out_dim': embed_dim
        }, keys[0])
        
        # Learnable positional embeddings
        self.position_embedding = jax.random.normal(keys[1], (max_seq_len, embed_dim)) * 0.02
        
        # Transformer blocks
        self.transformer_blocks = []
        for i in range(layer_num):
            block = TransformerBlock(embed_dim, num_heads, keys[i + 2])
            self.transformer_blocks.append(block)
        
        # Final layer normalization
        self.layer_norm_final = LayerNorm(embed_dim, keys[-2])
        
        # Output heads
        self.score_head = layers.create_model({
            'model_type': 'MLP-Layers',
            'activation': 'ELU',
            'in_dim': embed_dim,
            'hidden_layer_sizes': [embed_dim//2, embed_dim//4, embed_dim//8, embed_dim//16],
            'out_dim': 1
        }, keys[-1])
        
        self.weight_head = layers.create_model({
            'model_type': 'MLP-Layers',
            'activation': 'ELU',
            'in_dim': embed_dim,
            'hidden_layer_sizes': [embed_dim//2, embed_dim//4, embed_dim//8, embed_dim//16],
            'out_dim': 1
        }, keys[-1])

    def __call__(self, features, *, key=None, training=False):
        # features: [n_elements, feature_dim]
        seq_len = features.shape[0]
        
        # 1. Feature embedding
        x = jax.vmap(self.embedding_proj)(features)  # [n_elements, embed_dim]
        
        # 2. Positional encoding
        pos_embed = self.position_embedding[:seq_len]  # [n_elements, embed_dim]
        x = x + pos_embed
        
        # 3. Transformer block processing
        for block in self.transformer_blocks:
            x = block(x, key=key, training=training)
        
        # 4. Final layer normalization
        x = self.layer_norm_final(x)
        
        # 5. Output heads
        scores = jax.vmap(self.score_head)(x).squeeze(-1)
        weights = jax.vmap(self.weight_head)(x).squeeze(-1)
        
        return scores, weights

class TransformerBlock(eqx.Module):
    multi_head_attention: eqx.Module
    feed_forward: eqx.Module
    layer_norm1: eqx.Module
    layer_norm2: eqx.Module
    dropout_rate: float

    def __init__(self, embed_dim, num_heads, key, dropout_rate=0.1):
        keys = jax.random.split(key, 4)
        self.dropout_rate = dropout_rate
        
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads, keys[0])
        
        # Simplified feed-forward
        self.feed_forward = layers.create_model({
            'model_type': 'MLP-Layers',
            'activation': 'ELU',
            'in_dim': embed_dim,
            'hidden_layer_sizes': [embed_dim * 2],  # single hidden layer
            'out_dim': embed_dim
        }, keys[1])
        
        self.layer_norm1 = LayerNorm(embed_dim, keys[2])
        self.layer_norm2 = LayerNorm(embed_dim, keys[3])

    def __call__(self, x, *, key=None, training=False):
        # Self-attention with residual connection
        attn_out = self.multi_head_attention(x, x, x, rng_key=key, training=training)
        x = self.layer_norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = jax.vmap(self.feed_forward)(x)
        x = self.layer_norm2(x + ff_out)
        
        return x

class MultiHeadAttention(eqx.Module):
    num_heads: int
    head_dim: int
    embed_dim: int
    q_proj: eqx.Module
    k_proj: eqx.Module
    v_proj: eqx.Module
    out_proj: eqx.Module
    scale: float

    def __init__(self, embed_dim, num_heads, key):
        keys = jax.random.split(key, 4)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / jnp.sqrt(self.head_dim)
        
        # Q, K, V projections
        self.q_proj = layers.create_model({
            'model_type': 'MLP-Layers',
            'activation': 'ELU',
            'in_dim': embed_dim,
            'hidden_layer_sizes': [],
            'out_dim': embed_dim
        }, keys[0])
        
        self.k_proj = layers.create_model({
            'model_type': 'MLP-Layers',
            'activation': 'ELU',
            'in_dim': embed_dim,
            'hidden_layer_sizes': [],
            'out_dim': embed_dim
        }, keys[1])
        
        self.v_proj = layers.create_model({
            'model_type': 'MLP-Layers',
            'activation': 'ELU',
            'in_dim': embed_dim,
            'hidden_layer_sizes': [],
            'out_dim': embed_dim
        }, keys[2])
        
        self.out_proj = layers.create_model({
            'model_type': 'MLP-Layers',
            'activation': 'ELU',
            'in_dim': embed_dim,
            'hidden_layer_sizes': [embed_dim],
            'out_dim': embed_dim
        }, keys[3])

    def __call__(self, query, key_input, value, *, rng_key=None, training=False, mask=None):
        batch_size, seq_len, _ = query.shape if query.ndim == 3 else (1, query.shape[0], query.shape[1])
        
        # If 2D, add batch dimension
        if query.ndim == 2:
            query = query[None, :, :]
            key_input = key_input[None, :, :]
            value = value[None, :, :]
            added_batch_dim = True
        else:
            added_batch_dim = False
        
        # Q, K, V projection
        Q = jax.vmap(jax.vmap(self.q_proj))(query)  # [batch, seq_len, embed_dim]
        K = jax.vmap(jax.vmap(self.k_proj))(key_input)
        V = jax.vmap(jax.vmap(self.v_proj))(value)
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask (if provided)
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)
        
        # Softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Attention output
        attn_output = jnp.matmul(attn_weights, V)
        
        # Merge heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = jax.vmap(jax.vmap(self.out_proj))(attn_output)
        
        # Remove batch dimension (if added)
        if added_batch_dim:
            output = output[0]
        
        return output

class LayerNorm(eqx.Module):
    gamma: jnp.ndarray
    beta: jnp.ndarray
    eps: float

    def __init__(self, embed_dim, key, eps=1e-6):
        self.gamma = jnp.ones(embed_dim)
        self.beta = jnp.zeros(embed_dim)
        self.eps = eps

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta

def extract_element_features(A, b, history_len=5):
    """
    Numpy feature extraction (used only outside JAX autodiff).
    Returns [n_elements, 33].
    """
    n_samples, n_elements = A.shape
    feature_list = []
    
    history_len = min(history_len, n_samples)
    
    for i in range(n_elements):
        a_hist = A[-history_len:, i]
        b_hist = b[-history_len:]
        a_full = A[:, i]
        
        # === 基础统计特征 (12个) ===
        norm_hist = np.linalg.norm(a_hist) + 1e-10
        dot_hist = np.dot(a_hist, b_hist)
        b_norm = np.linalg.norm(b_hist) + 1e-10
        cosine_hist = dot_hist / (norm_hist * b_norm)
        
        mean_hist = np.mean(a_hist)
        std_hist = np.std(a_hist) + 1e-10
        var_hist = np.var(a_hist)
        max_hist = np.max(a_hist)
        min_hist = np.min(a_hist)
        median_hist = np.median(a_hist)
        range_hist = max_hist - min_hist
        
        # 相关性
        if history_len > 1 and np.std(a_hist) > 1e-10 and np.std(b_hist) > 1e-10:
            corr_hist = np.corrcoef(a_hist, b_hist)[0, 1]
            if np.isnan(corr_hist):
                corr_hist = 0.0
        else:
            corr_hist = 0.0
        
        # === 位置和周期编码 (6个) ===
        idx_norm = i / n_elements
        idx_sin1 = np.sin(2 * np.pi * i / n_elements)
        idx_cos1 = np.cos(2 * np.pi * i / n_elements)
        idx_sin2 = np.sin(4 * np.pi * i / n_elements)
        idx_cos2 = np.cos(4 * np.pi * i / n_elements)
        idx_log = np.log(i + 1) / np.log(n_elements + 1)
        
        # === 时序和梯度特征 (10个) ===
        if history_len > 1:
            # 一阶和二阶差分
            diff1 = np.diff(a_hist)
            mean_diff1 = np.mean(diff1) if diff1.size > 0 else 0.0
            std_diff1 = np.std(diff1) if diff1.size > 0 else 0.0
            max_diff1 = np.max(diff1) if diff1.size > 0 else 0.0
            min_diff1 = np.min(diff1) if diff1.size > 0 else 0.0
            
            diff2 = np.diff(diff1) if diff1.size > 1 else np.array([0.0])
            mean_diff2 = np.mean(diff2) if diff2.size > 0 else 0.0
            std_diff2 = np.std(diff2) if diff2.size > 0 else 0.0
            
            # 趋势分析
            if len(a_hist) > 2:
                x_trend = np.arange(len(a_hist))
                trend_coef = np.polyfit(x_trend, a_hist, 1)
                trend_slope, trend_intercept = trend_coef[0], trend_coef[1]
                # 二次趋势
                if len(a_hist) > 3:
                    trend_quad = np.polyfit(x_trend, a_hist, 2)[0]
                else:
                    trend_quad = 0.0
            else:
                trend_slope = trend_intercept = trend_quad = 0.0
                
            # 变异系数
            cv = std_hist / (np.abs(mean_hist) + 1e-10)
        else:
            mean_diff1 = std_diff1 = max_diff1 = min_diff1 = 0.0
            mean_diff2 = std_diff2 = trend_slope = trend_intercept = trend_quad = cv = 0.0
        
        # === 分布和统计特征 (10个) ===
        percentile_5 = np.percentile(a_full, 5)
        percentile_25 = np.percentile(a_full, 25)
        percentile_75 = np.percentile(a_full, 75)
        percentile_95 = np.percentile(a_full, 95)
        iqr = percentile_75 - percentile_25
        
        # 偏度和峰度
        if len(a_full) > 3 and np.std(a_full) > 1e-10:
            standardized = (a_full - np.mean(a_full)) / np.std(a_full)
            skewness = np.mean(standardized ** 3)
            kurtosis = np.mean(standardized ** 4) - 3
            # 更高阶矩
            moment_5 = np.mean(standardized ** 5)
            moment_6 = np.mean(standardized ** 6)
        else:
            skewness = kurtosis = moment_5 = moment_6 = 0.0
        
        # 分位数比率
        q_ratio = (percentile_75 - percentile_25) / (percentile_95 - percentile_5 + 1e-10)
        
        # === 频域和能量特征 (8个) ===
        energy = np.sum(a_hist ** 2)
        power = energy / len(a_hist)
        rms = np.sqrt(np.mean(a_hist ** 2))
        peak_to_rms = np.max(np.abs(a_hist)) / (rms + 1e-10)
        
        # 零交叉率和峰值计数
        if len(a_hist) > 1:
            zero_crossings = np.sum(np.diff(np.sign(a_hist - np.mean(a_hist))) != 0)
            zero_crossing_rate = zero_crossings / len(a_hist)
            
            # 峰值检测（简化）
            peaks = np.sum((a_hist[1:-1] > a_hist[:-2]) & (a_hist[1:-1] > a_hist[2:]))
            peak_rate = peaks / len(a_hist)
        else:
            zero_crossing_rate = peak_rate = 0.0
        
        # 谱质心的简化计算
        if len(a_hist) > 1:
            fft_vals = np.abs(np.fft.fft(a_hist))
            freq_bins = np.arange(len(fft_vals))
            spectral_centroid = np.sum(freq_bins * fft_vals) / (np.sum(fft_vals) + 1e-10)
            spectral_rolloff = np.sum(fft_vals > 0.85 * np.max(fft_vals)) / len(fft_vals)
        else:
            spectral_centroid = spectral_rolloff = 0.0
        
        # === 相对和比较特征 (6个) ===
        global_mean = np.mean(a_full)
        global_std = np.std(a_full) + 1e-10
        global_max = np.max(a_full)
        global_min = np.min(a_full)
        
        local_vs_global_mean = (mean_hist - global_mean) / global_std
        local_vs_global_std = std_hist / global_std
        local_vs_global_range = range_hist / (global_max - global_min + 1e-10)
        
        # 与相邻元素的关系
        if i > 0:
            neighbor_prev = A[-history_len:, i-1]
            corr_prev = np.corrcoef(a_hist, neighbor_prev)[0, 1] if history_len > 1 else 0.0
            if np.isnan(corr_prev):
                corr_prev = 0.0
        else:
            corr_prev = 0.0
            
        if i < n_elements - 1:
            neighbor_next = A[-history_len:, i+1]
            corr_next = np.corrcoef(a_hist, neighbor_next)[0, 1] if history_len > 1 else 0.0
            if np.isnan(corr_next):
                corr_next = 0.0
        else:
            corr_next = 0.0
        
        spatial_smooth = (corr_prev + corr_next) / 2
        
        # === 信息论特征 (2个) ===
        # 信息熵
        if len(a_hist) > 1:
            hist_counts, _ = np.histogram(a_hist, bins=min(8, len(a_hist)))
            hist_probs = hist_counts / (np.sum(hist_counts) + 1e-10)
            entropy = -np.sum(hist_probs * np.log(hist_probs + 1e-10))
            
            # 近似复杂度（连续相等值的比例）
            complexity = 1 - np.sum(np.diff(a_hist) == 0) / (len(a_hist) - 1) if len(a_hist) > 1 else 0
        else:
            entropy = complexity = 0.0
        
        # --- 新增动态投影特征 (基于当前 residual/向量 b 的全长) ---
        norm_full = np.linalg.norm(a_full) + 1e-10
        norm_b_full = np.linalg.norm(b) + 1e-10
        g_i = np.dot(a_full, b)
        cos_full = g_i / (norm_full * norm_b_full)
        kappa_i = g_i / (norm_full ** 2 + 1e-10)  # 类似最小二乘单变量解

        # 选择最重要的30个特征 + 3个动态特征 = 33
        feature = np.array([
            # 基础统计 (8个)
            norm_hist, dot_hist, cosine_hist, mean_hist, std_hist, 
            max_hist, min_hist, corr_hist,
            # 位置编码 (3个)
            idx_norm, idx_sin1, idx_cos1,
            # 时序特征 (6个)
            mean_diff1, std_diff1, mean_diff2, 
            trend_slope, trend_intercept, cv,
            # 分布特征 (5个)
            percentile_25, percentile_75, iqr,
            skewness, kurtosis,
            # 频域特征 (4个)
            energy, rms, zero_crossing_rate, spectral_centroid,
            # 相对特征 (4个)
            local_vs_global_mean, local_vs_global_std,
            corr_prev, corr_next,
            # 新增 3 个动态特征
            g_i, cos_full, kappa_i
        ])
        
        feature = np.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)
        feature_list.append(feature)
    
    return np.stack(feature_list, axis=0)  # [n_elements, 33]

def extract_element_features_jax(A, b, history_len=5):
    """
    JAX vectorized feature extraction. Output shape [n_elements, 33].
    Last 3 dims: (g_i, cosine_full, kappa_i).
    """
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    n_samples, n_elements = A.shape
    L = jnp.minimum(history_len, n_samples)
    A_hist = A[-L:, :]              # [L, M]
    b_hist = b[-L:]                 # [L]

    # 基础统计（历史窗口）
    norm_hist = jnp.linalg.norm(A_hist, axis=0) + 1e-10
    b_norm = jnp.linalg.norm(b_hist) + 1e-10
    dot_hist = jnp.sum(A_hist * b_hist[:, None], axis=0)
    cosine_hist = dot_hist / (norm_hist * b_norm)

    mean_hist = jnp.mean(A_hist, axis=0)
    std_hist = jnp.std(A_hist, axis=0) + 1e-10
    max_hist = jnp.max(A_hist, axis=0)
    min_hist = jnp.min(A_hist, axis=0)

    # corr_hist 与 b 的相关
    A_center = A_hist - mean_hist
    b_center = b_hist - jnp.mean(b_hist)
    corr_hist = jnp.sum(A_center * b_center[:, None], axis=0) / (
        (jnp.linalg.norm(A_center, axis=0) * jnp.linalg.norm(b_center) + 1e-10)
    )

    # 位置 / 周期编码
    idx = jnp.arange(n_elements)
    idx_norm = idx / jnp.maximum(1, n_elements - 1)
    idx_sin1 = jnp.sin(2 * jnp.pi * idx / n_elements)
    idx_cos1 = jnp.cos(2 * jnp.pi * idx / n_elements)

    # 差分与趋势
    def safe_stats_diff(arr):
        return (jnp.mean(arr, axis=0), jnp.std(arr, axis=0),
                jnp.max(arr, axis=0), jnp.min(arr, axis=0))
    if L > 1:
        diff1 = A_hist[1:, :] - A_hist[:-1, :]
        mean_diff1, std_diff1, _, _ = safe_stats_diff(diff1)
    else:
        mean_diff1 = std_diff1 = jnp.zeros(n_elements)
        diff1 = jnp.zeros((0, n_elements))
    if diff1.shape[0] > 1:
        diff2 = diff1[1:, :] - diff1[:-1, :]
        mean_diff2 = jnp.mean(diff2, axis=0)
    else:
        mean_diff2 = jnp.zeros(n_elements)

    # 线性趋势（最小二乘）
    if L > 1:
        x = jnp.arange(L).astype(A.dtype)
        x_mean = jnp.mean(x)
        x_center = x - x_mean
        var_x = jnp.sum(x_center ** 2) + 1e-10
        cov_xa = jnp.sum(x_center[:, None] * (A_hist - mean_hist), axis=0)
        trend_slope = cov_xa / var_x
        trend_intercept = mean_hist - trend_slope * x_mean
    else:
        trend_slope = trend_intercept = jnp.zeros(n_elements)

    cv = std_hist / (jnp.abs(mean_hist) + 1e-10)

    # 分布特征（全历史所有样本 A）
    percentile_25 = jnp.percentile(A, 25.0, axis=0)
    percentile_75 = jnp.percentile(A, 75.0, axis=0)
    iqr = percentile_75 - percentile_25

    mean_full = jnp.mean(A, axis=0)
    std_full = jnp.std(A, axis=0) + 1e-10
    standardized_full = (A - mean_full) / std_full
    skewness = jnp.mean(standardized_full ** 3, axis=0)
    kurtosis = jnp.mean(standardized_full ** 4, axis=0) - 3.0

    # 频域 / 能量（基于历史窗口）
    energy = jnp.sum(A_hist ** 2, axis=0)
    rms = jnp.sqrt(jnp.mean(A_hist ** 2, axis=0))
    if L > 1:
        mean_center = A_hist - mean_hist
        sign_seq = jnp.sign(mean_center)
        zero_crossing_rate = jnp.sum(
            jnp.not_equal(sign_seq[1:, :], sign_seq[:-1, :]), axis=0
        ) / (L - 1)
        fft_vals = jnp.abs(jnp.fft.fft(A_hist, axis=0))
        freq_bins = jnp.arange(L).reshape(-1, 1)
        spectral_centroid = jnp.sum(freq_bins * fft_vals, axis=0) / (jnp.sum(fft_vals, axis=0) + 1e-10)
    else:
        zero_crossing_rate = spectral_centroid = jnp.zeros(n_elements)

    # 相对特征（与全局分布相比）
    local_vs_global_mean = (mean_hist - mean_full) / std_full
    local_vs_global_std = std_hist / std_full

    # 邻接相关
    if n_elements > 1 and L > 1:
        A_hist_mean = mean_hist
        A_hist_center = A_hist - A_hist_mean
        A_prev_center = A_hist_center[:, :-1]
        A_next_center = A_hist_center[:, 1:]
        num_prev = jnp.sum(A_next_center * A_prev_center, axis=0)
        den_prev = (jnp.linalg.norm(A_next_center, axis=0) *
                    jnp.linalg.norm(A_prev_center, axis=0) + 1e-10)
        corr_pair_prev = num_prev / den_prev
        corr_prev = jnp.concatenate([jnp.zeros(1), corr_pair_prev])
        corr_next = jnp.concatenate([corr_pair_prev, jnp.zeros(1)])
    else:
        corr_prev = corr_next = jnp.zeros(n_elements)

    # 已有特征计算结束后新增动态特征 (基于全长度 A 与当前 b/residual)
    g_full = jnp.sum(A * b[:, None], axis=0)                        # g_i
    norm_A_full = jnp.linalg.norm(A, axis=0) + 1e-10
    norm_b_full = jnp.linalg.norm(b) + 1e-10
    cos_full = g_full / (norm_A_full * norm_b_full)                 # 余弦
    kappa_full = g_full / (norm_A_full ** 2 + 1e-10)                # 类似单列最小二乘系数

    features = jnp.stack([
        # ...existing code (原 30 个特征)...
        norm_hist,
        dot_hist,
        cosine_hist,
        mean_hist,
        std_hist,
        max_hist,
        min_hist,
        corr_hist,
        idx_norm,
        idx_sin1,
        idx_cos1,
        mean_diff1,
        std_diff1,
        mean_diff2,
        trend_slope,
        trend_intercept,
        cv,
        percentile_25,
        percentile_75,
        iqr,
        skewness,
        kurtosis,
        energy,
        rms,
        zero_crossing_rate,
        spectral_centroid,
        local_vs_global_mean,
        local_vs_global_std,
        corr_prev,
        corr_next,
        # 新增 3 个动态特征
        g_full,
        cos_full,
        kappa_full
    ], axis=1)  # [M, 33]

    features = jnp.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features

# 选择机制：训练用 Gamble Softmax 返回 probs*weights 可微；推理用 top-k
def transformer_cubature_selection(
    A, b, max_elements, transformer_static, transformer_params,
    temperature=0.2, topk=None, candidate_mask=None, history_len=5, key=None, mode="train"):
    """
    Differentiable selection (train: gamble softmax weighted sum; eval: top-k).
    A: [n_samples, n_elements], b: [n_samples]
    mode: "train" 使用概率加权, "eval" 使用top-k
    """
    full_model: ModernTransformer = eqx.combine(transformer_params, transformer_static)
    feats = extract_element_features_jax(A, b, history_len=history_len)
    scores, raw_weights = full_model(feats, key=key, training=(key is not None))
    if candidate_mask is not None:
        scores = jnp.where(candidate_mask, scores, -1e9)
    probs = jax.nn.softmax(scores / temperature)
    weights = jax.nn.softplus(raw_weights)

    if mode == "train":
        cubature_weights = probs * weights
        return cubature_weights, probs
    else:
        k = topk if topk is not None else max_elements
        topk_indices = jnp.argsort(probs)[::-1][:k]
        selected_weights = weights[topk_indices]
        return topk_indices, selected_weights

# === 新增：标量幅值对齐 + Lipschitz-PGD 非负精修 ===
def scalar_magnitude_align(A_sub, b, w):
    """
    求 alpha* >= 0, 使 ||A_sub (alpha w) - b||^2 最小。alpha = <Aw,b>/||Aw||^2。
    返回对齐后的 w_aligned = alpha * w。
    """
    Aw = A_sub @ w
    num = jnp.dot(Aw, b)
    denom = jnp.dot(Aw, Aw) + 1e-12
    alpha = jnp.maximum(0.0, num / denom)
    return alpha * w, alpha

def spectral_norm_sq(A_sub, iters=5):
    """
    近似 ||A_sub||_2^2，power-iteration（少步数，JAX 兼容）。
    """
    n = A_sub.shape[1]
    v = jnp.ones((n,), dtype=A_sub.dtype) / jnp.sqrt(n)
    for _ in range(iters):
        u = A_sub @ v
        u = u / (jnp.linalg.norm(u) + 1e-12)
        v = A_sub.T @ u
        v = v / (jnp.linalg.norm(v) + 1e-12)
    sigma = jnp.linalg.norm(A_sub @ v)
    return sigma * sigma + 1e-12

def pgd_nonneg_refine(A_sub, b, w0, steps=5, step_mul=1.0, align_each_step=True):
    """
    非负 PGD：步长取 1/L, L≈||A_sub||_2^2；每步 grad = A_sub^T (A_sub w - b)，投影 ReLU。
    可选每步/仅首末做标量幅值对齐。
    """
    L = spectral_norm_sq(A_sub, iters=5)
    eta = step_mul / L
    w, _ = scalar_magnitude_align(A_sub, b, w0)  # 先做一次幅值对齐
    for _ in range(steps):
        grad = A_sub.T @ (A_sub @ w - b)
        w = jax.nn.relu(w - eta * grad)
        if align_each_step:
            w, _ = scalar_magnitude_align(A_sub, b, w)
    if not align_each_step:
        w, _ = scalar_magnitude_align(A_sub, b, w)
    return w

def transformer_predict_scores_weights(A, b, transformer_static, transformer_params,
                                       temperature=0.2, candidate_mask=None, history_len=5, key=None):
    """
    仅前向：返回 scores、probs、weights。供 train/eval 统一调用。
    """
    full_model: ModernTransformer = eqx.combine(transformer_params, transformer_static)
    feats = extract_element_features_jax(A, b, history_len=history_len)
    scores, raw_weights = full_model(feats, key=key, training=(key is not None))
    if candidate_mask is not None:
        scores = jnp.where(candidate_mask, scores, -1e9)
    probs = jax.nn.softmax(scores / temperature)
    weights = jax.nn.softplus(raw_weights)
    return scores, probs, weights

def eval_topk_pgd(A, b, transformer_static, transformer_params,
                  topk, temperature=0.2, candidate_mask=None, history_len=5,
                  pgd_steps=10):
    """
    评估路径：TopK 选择 + 标量对齐 + PGD 非负精修（JAX），无 SciPy。
    返回 (w_full, residual)（均在 A,b 的归一化尺度上）。
    """
    _, probs, weights = transformer_predict_scores_weights(
        A, b, transformer_static, transformer_params, temperature, candidate_mask, history_len, key=None
    )
    k = int(topk)
    sel = jnp.argsort(probs)[::-1][:k]
    A_sub = A[:, sel]
    w0_sub = weights[sel]
    w_ref_sub = pgd_nonneg_refine(A_sub, b, w0_sub, steps=pgd_steps, step_mul=1.0, align_each_step=True)
    M = A.shape[1]
    w_full = jnp.zeros((M,), dtype=A.dtype).at[sel].set(w_ref_sub)
    residual = b - A @ w_full
    return w_full, residual

# === 新增：ST-TopK（前向硬 TopK + 反向用 probs 的直通近似） ===
def st_topk_mask(probs: jnp.ndarray, k: int):
    """
    返回:
      hard_idx: TopK 的硬索引
      hard_mask: 二值掩码 (0/1)
      st_mask: 直通掩码，前向=hard_mask，反向梯度=probs
    """
    k = int(k)
    hard_idx = jnp.argsort(probs)[::-1][:k]
    hard_mask = jnp.zeros_like(probs).at[hard_idx].set(1.0)
    # ST: hard + probs - stop_gradient(probs)
    st_mask = hard_mask + probs - jax.lax.stop_gradient(probs)
    return hard_idx, hard_mask, st_mask

# ...existing code...

def train_transformer_cubature(
    A, b, max_elements, n_epochs=1000, lr=1e-5,
    lr_decay_frac=0.995, lr_decay_every=10,
    lr_warm_up_iters=100, embed_dim=1024,
    temperature=1.0, batch_size=8192,
    topk=None, candidate_mask=None,
    history_len=7, teacher_w=None,
    entropy_weight=1e-3, distill_weight=1e-1,
    train_detailed_log=False
):
    """
    Train gamble-softmax differentiable selector for NNLS: min ||A w - b||, w>=0.
    """
    key = jax.random.PRNGKey(2024)
    feature_dim = 33  
    layer_num = BIG_LAYER_NUM
    num_heads = NUM_HEADS

    jax.clear_caches()
    transformer = ModernTransformer(feature_dim, embed_dim, layer_num, num_heads, key)
    transformer_params, transformer_static = eqx.partition(transformer, eqx.is_array)

    def step_func(i):
        warmup = jnp.clip((i + 1) / (lr_warm_up_iters + 1), 0, 1)
        decay = (lr_decay_frac ** (i // lr_decay_every))
        return lr * warmup * decay

    opt = optimizers.adam(step_func, b1=0.9, b2=0.999, eps=1e-8)
    opt_state = opt.init_fn(transformer_params)

    N = A.shape[0]
    M = A.shape[1]
    actual_batch_size = min(int(batch_size), int(N))

    def loss_fn(params, A_batch, b_batch, key):
        # === 单步“硬 TopK 前向 + ST 反传” ===
        # 1) 前向获得 scores/probs/weights
        _, probs, weights = transformer_predict_scores_weights(
            A_batch, b_batch, transformer_static, params,
            temperature=temperature, candidate_mask=candidate_mask,
            history_len=history_len, key=key
        )

        k_refine = int(topk or max_elements)

        # 2) ST-TopK 掩码：前向硬，反向走 probs
        sel, hard_mask, st_mask = st_topk_mask(probs, k_refine)

        # 3) 用 ST 掩码门控初始权重（前向=硬 TopK 权重；反向对 probs 可导）
        w0_full = weights * st_mask

        # 4) 取硬 TopK 子系统并做幅值对齐 + PGD 非负精修
        A_sub = A_batch[:, sel]
        w0_sub = w0_full[sel]
        w0_sub_aligned, alpha_star = scalar_magnitude_align(A_sub, b_batch, w0_sub)
        w_ref_sub = pgd_nonneg_refine(A_sub, b_batch, w0_sub_aligned, steps=5, step_mul=1.0, align_each_step=True)

        # 5) 回填 full 向量
        M = A_batch.shape[1]
        w_ref_full = jnp.zeros((M,), dtype=A_batch.dtype).at[sel].set(w_ref_sub)

        # 6) 残差与多项损失
        final_residual = b_batch - A_batch @ w_ref_full

        mse = jnp.mean(final_residual ** 2)
        mae = jnp.mean(jnp.abs(final_residual))
        relerr = jnp.linalg.norm(final_residual) / (jnp.linalg.norm(b_batch) + 1e-12)

        huber_delta = 0.5
        huber_loss = jnp.mean(jnp.where(
            jnp.abs(final_residual) <= huber_delta,
            0.5 * final_residual ** 2,
            huber_delta * (jnp.abs(final_residual) - 0.5 * huber_delta)
        ))

        # 稀疏/平滑正则
        l1_sparse = 1e-4 * jnp.sum(jnp.abs(w_ref_full))
        l2_reg = 5e-5 * jnp.sum(w_ref_full ** 2)
        w_diff = w_ref_full[1:] - w_ref_full[:-1]
        w_smooth = 5e-4 * jnp.sum(w_diff ** 2)

        # 概率熵正则（对 probs）
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))

        # 幅值对齐损失（alpha* ≈ 1）
        amp_align_loss = (alpha_star - 1.0) ** 2

        # 蒸馏（仅在硬 TopK 支持上）
        distill = 0.0
        if teacher_w is not None:
            t_sub = teacher_w[sel]
            sn = jnp.linalg.norm(w_ref_sub) + 1e-8
            tn = jnp.linalg.norm(t_sub) + 1e-8
            distill = jnp.mean((w_ref_sub / sn - t_sub / tn) ** 2)

        loss = (10.0 * relerr +
                2.0 * mse +
                0.5 * mae +
                huber_loss +
                l1_sparse + l2_reg + w_smooth +
                entropy_weight * entropy +
                5.0 * distill_weight * distill +
                0.5 * amp_align_loss)

        return loss, (relerr, mse, mae, huber_loss, l1_sparse, l2_reg, w_smooth, float(amp_align_loss))

    print(f"Start Transformer training for {n_epochs} epochs, batch_size={actual_batch_size}, N={N}, M={M}")
    best_rel_err = float('inf')
    patience = 200
    patience_counter = 0
    rng_np = np.random.default_rng(0)

    # === 新增：提前缓存全量归一化系统用于一致评估 ===
    A_full_eval = A
    b_full_eval = b
    n_refinement_steps_eval = 3  # 训练中 & 测试中统一

    metrics_history = [] if train_detailed_log else None
    train_t0 = time.time()  # 新增：总体训练计时

    for epoch in range(n_epochs):
        temp = max(0.01, float(temperature) * (0.999 ** epoch))
        key = jax.random.PRNGKey(epoch)

        # --- 采样 batch (仅用于反向传播) ---
        if N >= actual_batch_size:
            idxs = rng_np.choice(N, actual_batch_size, replace=False)
        else:
            idxs = rng_np.choice(N, actual_batch_size, replace=True)
        A_batch = A[idxs, :]
        b_batch = b[idxs]

        # 修复：使用正确的参数访问方式
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            opt.params_fn(opt_state), A_batch, b_batch, key
        )
        relerr, mse, mae, huber_loss, l1_sparse, l2_reg, w_smooth, amp_loss = aux

        # 简单梯度裁剪
        grad_norm = jax.tree_util.tree_reduce(lambda x, y: x + jnp.linalg.norm(y), grads, 0.0)
        max_grad_norm = 5.0
        if grad_norm > max_grad_norm:
            factor = max_grad_norm / grad_norm
            grads = jax.tree_util.tree_map(lambda x: x * factor, grads)

        opt_state = opt.update_fn(epoch, grads, opt_state)
        params_cur = opt.params_fn(opt_state)

        # === 评估：TopK + PGD （无 NNLS） ===
        transformer_params_cur = opt.params_fn(opt_state)
        w_eval, residual_eval = eval_topk_pgd(
            A_full_eval, b_full_eval, transformer_static, transformer_params_cur,
            topk=(topk or max_elements), temperature=temp, candidate_mask=candidate_mask,
            history_len=history_len, pgd_steps=10
        )
        rel_err_eval = float(jnp.linalg.norm(residual_eval) / (jnp.linalg.norm(b_full_eval) + 1e-12))

        print(f"[{epoch:04d}] loss={float(loss_val):.6f} "
              f"train_rel(batch)={float(relerr):.6f} mse={float(mse):.6f} mae={float(mae):.6f} "
              f"huber={float(huber_loss):.6f} amp={float(amp_loss):.6f} | "
              f"eval_rel(TopK+PGD)={rel_err_eval:.6f} temp={temp:.3f}")

        if train_detailed_log:
            metrics_history.append({
                "epoch": int(epoch),
                "loss": float(loss_val),
                "train_rel_batch": float(relerr),
                "iter_full_rel": float(rel_err_eval),
                "refine_full_rel": float(rel_err_eval)
            })

        # 以精修后误差作为早停判據（更稳定）
        eval_metric = rel_err_eval
        if eval_metric < best_rel_err:
            best_rel_err = eval_metric
            patience_counter = 0
        else:
            patience_counter += 1

        if eval_metric < 0.15:
            print(f"Target rel_err achieved at epoch {epoch}: {eval_metric:.6f}")
            break
        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch}, best eval rel={best_rel_err:.6f}")
            break

    total_train_time = time.time() - train_t0
    print(f"[训练] 总耗时 {total_train_time:.2f} s")

    if train_detailed_log and metrics_history:
        import matplotlib.pyplot as _plt
        ep = [m["epoch"] for m in metrics_history]
        loss_arr = [m["loss"] for m in metrics_history]
        iter_arr = [m["iter_full_rel"] for m in metrics_history]
        ref_arr = [m["refine_full_rel"] for m in metrics_history]
        fig, axs = _plt.subplots(1, 3, figsize=(12, 4))
        axs[0].plot(ep, loss_arr); axs[0].set_title("Loss")
        axs[1].plot(ep, iter_arr); axs[1].set_title("Eval RelErr (Iter)")
        axs[2].plot(ep, ref_arr); axs[2].set_title("Eval RelErr (Refine)")
        fig.tight_layout()
        plot_path = os.path.join(output_dir, "training_metrics.png")
        fig.savefig(plot_path, dpi=150)
        np.savez(os.path.join(output_dir, "metrics.npz"),
                 epoch=ep, loss=loss_arr, iter_rel=iter_arr, refine_rel=ref_arr)
        print(f"[详细日志] 训练指标与图像已保存: {plot_path}")

    return eqx.combine(opt.params_fn(opt_state), transformer_static)

def main():
    print("==== Transformer Cubature with Gamble Softmax (NNLS) ====")
    _ = jnp.zeros(())
    system, system_def = config.construct_system_from_name(system_name, problem_name)

    # Read PCA basis
    print(f'Reading PCA basis from {pca_basis}')
    U_full = igl.read_dmat(os.path.join(pca_basis, "basis.dmat"))
    if gradient_weighting:
        Sigma_full = igl.read_dmat(os.path.join(pca_basis, "eigenvalues.dmat"))
    else:
        Sigma_full = np.ones(U_full.shape[1], dtype=float)
    # 降 PCA 维度以控算量
    actual_pca_dim = min(64, U_full.shape[1])
    U = jnp.array(U_full[:, :actual_pca_dim])
    Sigma = jnp.array(Sigma_full[:actual_pca_dim])

    # 用 FEM 构造 NNLS 的 A,b（相对误差规范化）
    N_frames = 120
    cand_size = max(sub_n * 10, 100)
    A_mat, b_vec, cand_inds = build_nnls_data(
        system, system_def, U, Sigma, data_dir, num_frames=N_frames, candidate_size=cand_size, seed=42
    )
    print(f"NNLS data: A={A_mat.shape}, b={b_vec.shape}, cand={len(cand_inds)}")

    # ===== 缓存归一化前的原始尺度矩阵/向量 =====
    A_raw_full = A_mat
    b_raw_full = b_vec

    # 列归一化
    col_norms = jnp.linalg.norm(A_mat, axis=0) + 1e-8
    A_mat = A_mat / col_norms

    # 训练/测试划分（按行划分）
    split_ratio = 0.9
    split_idx = int(A_mat.shape[0] * split_ratio)
    A_train, A_test = A_mat[:split_idx], A_mat[split_idx:]
    b_train, b_test = b_vec[:split_idx], b_vec[split_idx:]

    # 原始尺度对应切分（用于最终评估）
    A_raw_train, A_raw_test = A_raw_full[:split_idx], A_raw_full[split_idx:]
    b_raw_train, b_raw_test = b_raw_full[:split_idx], b_raw_full[split_idx:]

    # 新增：候选掩码（基于 |A^T b| 的相关性，保留前 6*max_n 列，增大候选范围）
    M = A_mat.shape[1]
    keep_K = int(min(M, max(6 * max_n, 150)))  # 增大候选倍数
    corr = jnp.abs(A_train.T @ b_train)  # [M]
    keep_idx = jnp.argsort(corr)[-keep_K:]
    candidate_mask = jnp.zeros((M,), dtype=bool).at[keep_idx].set(True)

    # 新增：教师 NNLS（仅一次，做蒸馏目标，不进入训练/评估主路径）
    import numpy as _np
    import scipy.optimize as _scopt
    A_nnls = _np.asarray(A_train[:, _np.asarray(keep_idx)])
    b_nnls = _np.asarray(b_train)
    w_small, residual_nnls = _scopt.nnls(A_nnls, b_nnls)
    teacher_w = _np.zeros((M,), dtype=_np.float64)
    teacher_w[_np.asarray(keep_idx)] = w_small
    teacher_w = jnp.array(teacher_w)
    
    # 打印教师NNLS的基准误差
    teacher_rel_err = jnp.linalg.norm(b_train - A_train @ teacher_w) / (jnp.linalg.norm(b_train) + 1e-12)
    print(f"Teacher NNLS baseline rel_err: {float(teacher_rel_err):.6f}")

    # Timing start (training through refined NNLS evaluation)
    train_start_time = time.time()

    print("Start training transformer selector (Gamble Softmax, differentiable NNLS)...")
    model_tree = train_transformer_cubature(
        A_train, b_train, max_n,
        n_epochs=n_epochs, lr=lr,
        lr_decay_frac=lr_decay_frac, lr_decay_every=lr_decay_every,
        lr_warm_up_iters=lr_warm_up_iters, embed_dim=BIG_EMBED_DIM,
        temperature=temperature, batch_size=min(4096, A_train.shape[0]),
        topk=max_n, candidate_mask=candidate_mask, history_len=5,
        teacher_w=teacher_w, entropy_weight=5e-3, distill_weight=5e-1,
        train_detailed_log=Train_detailed_log_ENABLE
    )

    # Save model + metadata
    print(f"Saving model to {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "transformer_model.eqx")
    with open(model_path, "wb") as f:
        eqx.tree_serialise_leaves(f, model_tree)
    # 可选：保存辅助元数据（列归一化、候选掩码等）
    meta_path = os.path.join(output_dir, "aux_meta.npz")
    np.savez(meta_path,
             col_norms=np.asarray(col_norms),
             keep_idx=np.asarray(keep_idx),
             candidate_mask=np.asarray(candidate_mask))
    print(f"Model saved: {model_path}; metadata: {meta_path}")

    # === 测试阶段（归一化尺度）：TopK + PGD ===
    transformer_params, transformer_static = eqx.partition(model_tree, eqx.is_array)
    w_norm_test, residual_norm_test = eval_topk_pgd(
        A_test, b_test, transformer_static, transformer_params,
        topk=max_n, temperature=temperature, candidate_mask=candidate_mask, history_len=5, pgd_steps=10
    )
    test_rel_norm = float(jnp.linalg.norm(residual_norm_test) / (jnp.linalg.norm(b_test) + 1e-12))
    support = jnp.where(w_norm_test > 0)[0]
    print(f"Test (normalized) rel_err(TopK+PGD)={test_rel_norm:.6f} | k={support.size}")

    # === 映射到原始尺度 ===
    w_raw = w_norm_test / col_norms
    residual_raw = b_raw_test - A_raw_test @ w_raw
    rel_err_raw = float(jnp.linalg.norm(residual_raw) / (jnp.linalg.norm(b_raw_test) + 1e-12))
    print(f"Raw scale rel_err (mapped TopK+PGD normalized weights) = {rel_err_raw:.6f}")

    # 不再进行 raw-scale NNLS refine
    if refine_eval_nnls:
        print("refine_eval_nnls=True 但已统一 PGD 路径，跳过 NNLS。")

if __name__ == "__main__":
    main()