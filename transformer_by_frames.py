import math
import sys, os
from functools import partial
import json
import time
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import re

# 注入占位的 layers 模块，防止 equinox 内部 import layers 命中项目业务模块而循环导入
import types as _types
if "layers" not in sys.modules:
    sys.modules["layers"] = _types.ModuleType("layers")

import numpy as np
import jax
import jax.numpy as jnp
import optax

import equinox as eqx
import igl

import utils
import config
import custom_layers
import fem_model
import system_utils


SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, '..')


def extract_element_features_jax(A, b, history_len=5):
    """
    JAX vectorized feature extraction. Output shape [n_elements, 35].
    Last 5 dims: (g_i, cosine_full, kappa_i, appearance_count, appearance_ratio).
    """
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    n_samples, n_elements = A.shape
    L = jnp.minimum(history_len, n_samples)
    A_hist = A[-L:, :]              # [L, M]
    b_hist = b[-L:]                 # [L]
    
    norm_hist = jnp.linalg.norm(A_hist, axis=0) + 1e-10
    b_norm = jnp.linalg.norm(b_hist) + 1e-10
    dot_hist = jnp.sum(A_hist * b_hist[:, None], axis=0)
    cosine_hist = dot_hist / (norm_hist * b_norm)

    mean_hist = jnp.mean(A_hist, axis=0)
    std_hist = jnp.std(A_hist, axis=0) + 1e-10
    max_hist = jnp.max(A_hist, axis=0)
    min_hist = jnp.min(A_hist, axis=0)

    A_center = A_hist - mean_hist
    b_center = b_hist - jnp.mean(b_hist)
    corr_hist = jnp.sum(A_center * b_center[:, None], axis=0) / (
        (jnp.linalg.norm(A_center, axis=0) * jnp.linalg.norm(b_center) + 1e-10)
    )

    idx = jnp.arange(n_elements)
    idx_norm = idx / jnp.maximum(1, n_elements - 1)
    idx_sin1 = jnp.sin(2 * jnp.pi * idx / n_elements)
    idx_cos1 = jnp.cos(2 * jnp.pi * idx / n_elements)

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

    percentile_25 = jnp.percentile(A, 25.0, axis=0)
    percentile_75 = jnp.percentile(A, 75.0, axis=0)
    iqr = percentile_75 - percentile_25

    mean_full = jnp.mean(A, axis=0)
    std_full = jnp.std(A, axis=0) + 1e-10
    standardized_full = (A - mean_full) / std_full
    skewness = jnp.mean(standardized_full ** 3, axis=0)
    kurtosis = jnp.mean(standardized_full ** 4, axis=0) - 3.0

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

    local_vs_global_mean = (mean_hist - mean_full) / std_full
    local_vs_global_std = std_hist / std_full

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

    g_full = jnp.sum(A * b[:, None], axis=0)                        # g_i
    norm_A_full = jnp.linalg.norm(A, axis=0) + 1e-10
    norm_b_full = jnp.linalg.norm(b) + 1e-10
    cos_full = g_full / (norm_A_full * norm_b_full)                 # 余弦
    kappa_full = g_full / (norm_A_full ** 2 + 1e-10)                # 类似单列最小二乘系数

    appearance_count = jnp.sum(jnp.abs(A_hist) > 1e-10, axis=0)     # 元素在历史窗口中出现的次数
    appearance_ratio = appearance_count / L                         # 元素出现的帧占比

    features = jnp.stack([
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
        g_full,
        cos_full,
        kappa_full,
        appearance_count,
        appearance_ratio
    ], axis=1)  # [M, 35]

    features = jnp.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features

class SequentialDataLoader:
    
    def __init__(self, initial_options_dir, history_len=6):
        self.initial_options_dir = initial_options_dir
        self.history_len = history_len
        self.available_frames = self._scan_available_frames()
        self.max_frame = max(self.available_frames) if self.available_frames else 0
        
    def _scan_available_frames(self):
        """扫描可用的帧文件"""
        files = os.listdir(self.initial_options_dir)
        indices_files = [f for f in files if f.endswith("_indices.dmat") and f.startswith("displacements_")]  # 修改：匹配 displacements_ 前缀
        
        frame_numbers = []
        for f in indices_files:
            try:
                # 使用正则提取 frame_num，例如 从 "displacements_0_indices.dmat" 提取 "0"
                match = re.search(r'displacements_(\d+)_indices\.dmat', f)
                if match:
                    frame_num = int(match.group(1))
                    weights_file = f"displacements_{frame_num}_weights.dmat"  # 修改：匹配实际文件名
                    if os.path.exists(os.path.join(self.initial_options_dir, weights_file)):
                        frame_numbers.append(frame_num)
            except ValueError:
                continue
                
        frame_numbers.sort()  # 确保按帧号排序
        print(f"发现 {len(frame_numbers)} 个可用帧: {frame_numbers[:5]}...{frame_numbers[-5:] if len(frame_numbers) > 10 else frame_numbers[5:]}")
        return frame_numbers
    
    def load_frame_data(self, frame_num):
        """加载单帧的indices和weights数据"""
        base = f"displacements_{frame_num}"
        indices_path = os.path.join(self.initial_options_dir, f"{base}_indices.dmat")
        weights_path = os.path.join(self.initial_options_dir, f"{base}_weights.dmat")
        
        if not (os.path.exists(indices_path) and os.path.exists(weights_path)):
            return None, None
            
        indices = igl.read_dmat(indices_path).flatten().astype(int)
        weights = igl.read_dmat(weights_path).flatten()
        return indices, weights
    
    def get_continuous_sequences(self, min_seq_length=None):
        """
        获取所有连续的帧序列
        min_seq_length: 最小序列长度，默认为history_len + 1
        """
        if min_seq_length is None:
            min_seq_length = self.history_len + 1
            
        continuous_seqs = []
        current_seq = []
        
        for i, frame in enumerate(self.available_frames):
            if i == 0 or frame == self.available_frames[i-1] + 1:
                current_seq.append(frame)
            else:
                if len(current_seq) >= min_seq_length:
                    continuous_seqs.append(current_seq)
                current_seq = [frame]
        
        if len(current_seq) >= min_seq_length:
            continuous_seqs.append(current_seq)
            
        print(f"找到 {len(continuous_seqs)} 个连续序列，总共包含 {sum(len(seq) for seq in continuous_seqs)} 帧")
        return continuous_seqs
    
    def generate_sliding_windows(self, batch_size=4, valid_ratio=0.2, shuffle=True):
        """
        生成滑动窗口序列，供训练使用
        返回: (train_windows, valid_windows)
        每个window是(input_frames, target_frame)的元组
        """
        # 获取所有可用的连续序列
        sequences = self.get_continuous_sequences()
        
        all_windows = []
        # 为每个连续序列生成滑动窗口
        for seq in sequences:
            for i in range(len(seq) - self.history_len):
                # 输入窗口: [t, t+1, ..., t+history_len-1]
                # 目标: t+history_len
                input_window = seq[i:i+self.history_len]
                target_frame = seq[i+self.history_len]
                all_windows.append((input_window, target_frame))
        
        # 随机打乱
        if shuffle:
            np.random.shuffle(all_windows)
        
        # 划分训练集和验证集
        split_idx = int(len(all_windows) * (1 - valid_ratio))
        train_windows = all_windows[:split_idx]
        valid_windows = all_windows[split_idx:]
        
        # 按batch分组
        train_batches = [train_windows[i:i + batch_size] for i in range(0, len(train_windows), batch_size)]
        valid_batches = [valid_windows[i:i + batch_size] for i in range(0, len(valid_windows), batch_size)]
        
        print(f"生成 {len(train_batches)} 个训练批次，{len(valid_batches)} 个验证批次")
        return train_batches, valid_batches

class SlidingWindowFeatureExtractor:
    """
    基于滑动窗口的特征提取器 - 从前一个窗口滑动到下一个窗口
    """
    def __init__(self, dataloader, fem_total_elements, use_time_encoder=True):
        self.dataloader = dataloader
        self.fem_total_elements = fem_total_elements
        self.use_time_encoder = use_time_encoder
        
    def build_history_window(self, frame_window):
        """
        构建一个窗口的特征
        frame_window: 连续帧的列表 [t, t+1, ..., t+H-1]
        """
        window_len = len(frame_window)
        A = np.zeros((window_len, self.fem_total_elements), dtype=np.float32)
        b = np.zeros(window_len, dtype=np.float32)
        
        # 顺序填充A和b
        for i, frame in enumerate(frame_window):
            indices, weights = self.dataloader.load_frame_data(frame)
            if indices is not None and weights is not None:
                A[i, indices] = weights
                b[i] = np.linalg.norm(weights)  # 使用权重的L2范数作为b
        
        return jnp.array(A), jnp.array(b)
    
    def extract_batch_features(self, batch_windows):
        """
        为一个批次的滑动窗口提取特征
        batch_windows: [(input_window1, target1), (input_window2, target2), ...]
        返回：
          batch_inputs: list of dict, 每个 dict 包含:
            如果 use_time_encoder=True: 'features': jnp.array, shape (K, H, 1) -- per-element 时间序列
            如果 use_time_encoder=False: 'features': jnp.array, shape (K, 35) -- 35维特征
            'indices' : ndarray, 元素索引 length K
            'window_frames': input_window
          batch_targets: list of dict, 每个 dict 包含:
            'weights': target_weights (length K)
            'frame': target_frame
        """
        batch_inputs = []
        batch_targets = []
        
        for input_window, target_frame in batch_windows:
            # 为输入窗口构建特征矩阵 A (H, M) 和 b (H)
            A, b = self.build_history_window(input_window)  # A: [H, M]
            
            # 获取目标帧的权重和对应的元素索引
            target_indices, target_weights = self.dataloader.load_frame_data(target_frame)
            if target_indices is None or target_weights is None:
                continue
            
            if self.use_time_encoder:
                # 只选择目标帧中存在的元素的时间序列
                # A[:, target_indices] -> [H, K], 转置 -> [K, H]
                # 扩展最后一维为 per_time_dim (这里为 1)，得到 [K, H, 1]
                seq_vals = jnp.array(A[:, target_indices]).T  # [K, H]
                seq_vals = jnp.expand_dims(seq_vals, -1)     # [K, H, 1]
                features = seq_vals
            else:
                # 使用35维特征
                all_features = extract_element_features_jax(A, b, history_len=len(input_window))
                features = all_features[target_indices]  # [K, 35]
            
            batch_inputs.append({
                'features': features,
                'indices': target_indices,
                'window_frames': input_window
            })
            
            batch_targets.append({
                'weights': target_weights,
                'frame': target_frame
            })
        
        return batch_inputs, batch_targets


class TransformerBlock(eqx.Module):
    mha: "MultiHeadSelfAttention"
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, model_dims, num_heads, key):
        if model_dims % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        k_attn, k_l1, k_l2 = jax.random.split(key, 3)
        self.mha = MultiHeadSelfAttention(model_dims, num_heads, k_attn)
        self.linear1 = eqx.nn.Linear(model_dims, model_dims * 4, key=k_l1)
        self.linear2 = eqx.nn.Linear(model_dims * 4, model_dims, key=k_l2)
        self.norm1 = eqx.nn.LayerNorm(model_dims)
        self.norm2 = eqx.nn.LayerNorm(model_dims)

    def __call__(self, x, key=None):
        def _apply_layer_norm(norm, tensor):
            orig_shape = tensor.shape
            flat = tensor.reshape(-1, orig_shape[-1])
            normed = eqx.filter_vmap(norm)(flat)
            return normed.reshape(orig_shape)

        def apply_linear_with_vmap(linear, tensor):
            orig_shape = tensor.shape
            flat = tensor.reshape(-1, orig_shape[-1])
            transformed = eqx.filter_vmap(linear)(flat)
            out_dim = transformed.shape[-1]
            return transformed.reshape(*orig_shape[:-1], out_dim)

        attn_out = self.mha(x)
        x_res = x + attn_out
        x_norm = _apply_layer_norm(self.norm1, x_res)

        y = apply_linear_with_vmap(self.linear1, x_norm)
        y = jax.nn.gelu(y)
        y = apply_linear_with_vmap(self.linear2, y)

        out = x_norm + y
        return _apply_layer_norm(self.norm2, out)
    
class MultiHeadSelfAttention(eqx.Module):
    proj_q: eqx.nn.Linear
    proj_k: eqx.nn.Linear
    proj_v: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    num_heads: int
    head_dim: int

    def __init__(self, model_dims, num_heads, key):
        k_q, k_k, k_v, k_o = jax.random.split(key, 4)
        self.proj_q = eqx.nn.Linear(model_dims, model_dims, key=k_q)
        self.proj_k = eqx.nn.Linear(model_dims, model_dims, key=k_k)
        self.proj_v = eqx.nn.Linear(model_dims, model_dims, key=k_v)
        self.proj_out = eqx.nn.Linear(model_dims, model_dims, key=k_o)
        self.num_heads = num_heads
        self.head_dim = model_dims // num_heads

    def __call__(self, x):
        seq_len, batch, embed_dim = x.shape

        def apply_linear(layer, tensor):
            return jax.vmap(jax.vmap(layer))(tensor)

        q = apply_linear(self.proj_q, x)
        k = apply_linear(self.proj_k, x)
        v = apply_linear(self.proj_v, x)

        q = jnp.transpose(q.reshape(seq_len, batch, self.num_heads, self.head_dim), (1, 2, 0, 3))
        k = jnp.transpose(k.reshape(seq_len, batch, self.num_heads, self.head_dim), (1, 2, 0, 3))
        v = jnp.transpose(v.reshape(seq_len, batch, self.num_heads, self.head_dim), (1, 2, 0, 3))

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        attn_output = jnp.transpose(attn_output, (2, 0, 1, 3)).reshape(seq_len, batch, embed_dim)

        return apply_linear(self.proj_out, attn_output)

class TransformerStack(eqx.Module):
    blocks: tuple[TransformerBlock, ...]

    def __init__(self, num_layers, model_dims, num_heads, key):
        if num_layers < 1:
            self.blocks = ()
        else:
            keys = jax.random.split(key, num_layers)
            self.blocks = tuple(
                TransformerBlock(model_dims, num_heads, k) for k in keys
            )

    def __call__(self, x, key=None):
        curr_key = key
        for block in self.blocks:
            subkey = None
            if curr_key is not None:
                curr_key, subkey = jax.random.split(curr_key)
            x = block(x, key=subkey)
        return x

class Seq2SeqTransformer(eqx.Module):
    """序列到序列Transformer，支持 per-element temporal encoding"""
    temporal_input: eqx.nn.Linear
    temporal_encoder: TransformerStack
    encoder: TransformerStack
    history_embedding: eqx.Module
    output_projection: eqx.nn.Linear
    
    def __init__(self, per_time_dim, hidden_dim, num_layers, num_heads, key, temp_layers=1):
        """
        per_time_dim: 每帧的时间特征维度（例如 1: 仅权重）
        hidden_dim: per-element embedding 以及跨元素 Transformer 的 model_dim
        num_layers/num_heads: 用于跨元素 encoder/decoder
        temp_layers: temporal encoder 的层数 (默认 1)
        """
        key_temp_proj, key_temp_stack, key_cross_stack, key_hist, key_out = jax.random.split(key, 5)
        self.temporal_input = eqx.nn.Linear(per_time_dim, hidden_dim, key=key_temp_proj)
        self.temporal_encoder = TransformerStack(
            num_layers=max(1, temp_layers),
            model_dims=hidden_dim,
            num_heads=max(1, num_heads),
            key=key_temp_stack,
        )
        self.encoder = TransformerStack(
            num_layers=max(1, num_layers // 2),
            model_dims=hidden_dim,
            num_heads=max(1, num_heads),
            key=key_cross_stack,
        )
        self.history_embedding = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim,
            width_size=hidden_dim * 2,
            depth=4,
            key=key_hist,
        )
        self.output_projection = eqx.nn.Linear(hidden_dim, 1, key=key_out)

    def __call__(self, x, memory=None, key=None):
        """
        x: 输入特征, shape [B, K_pad, H, per_time_dim]
        memory: [B, hidden_dim] or None
        返回: weights [B, K_pad], current_memory [B, hidden_dim]
        """
        # x shape: [B, K_pad, H, per_time_dim]
        
        # 步骤1: 对每个元素的每个时间步应用线性投影
        x_proj = jax.vmap(jax.vmap(jax.vmap(self.temporal_input)))(x)
        # x_proj shape: [B, K_pad, H, hidden_dim]
        
        # 步骤2: 批量编码所有元素的时间序列
        B, K, H, D = x_proj.shape
        x_flat = jnp.reshape(x_proj, (B * K, H, D))

        def encode_sequence(seq):
            """对单个元素的时间序列进行编码。"""
            #jax.debug.print("encode_sequence: seq shape={}", seq.shape)
            seq_expanded = jnp.expand_dims(seq, axis=1)  # [H, 1, hidden_dim]
            #jax.debug.print("encode_sequence: seq_expanded shape={}", seq_expanded.shape)
            encoded = self.temporal_encoder(seq_expanded, key=None)  # [H, 1, hidden_dim]
            #jax.debug.print("encode_sequence: encoded shape={}", encoded.shape)
            return jnp.squeeze(encoded, axis=1)

        temp_encoded_flat = jax.vmap(encode_sequence)(x_flat)  # [B*K, H, hidden_dim]
        temp_encoded = jnp.reshape(temp_encoded_flat, (B, K, H, D))  # [B, K_pad, H, hidden_dim]
        
        elem_emb = jnp.mean(temp_encoded, axis=2)  # [B, K_pad, hidden_dim]
        
        # 步骤4: 跨元素编码
        encoder_in = jnp.transpose(elem_emb, (1, 0, 2))  # [K_pad, B, hidden_dim]
        encoded = self.encoder(encoder_in, key=key)  # [K_pad, B, hidden_dim]
        encoded = jnp.transpose(encoded, (1, 0, 2))  # [B, K_pad, hidden_dim]
        
        # 步骤5: 融合历史信息
        if memory is not None:
            history_embed = self.history_embedding(memory)  # [B, hidden_dim]
            decoded = encoded + history_embed[:, None, :]  # [B, K_pad, hidden_dim]
        else:
            decoded = encoded
        
        # 步骤6: 计算当前状态的汇总表示
        current_memory = jnp.mean(decoded, axis=1)  # [B, hidden_dim]
        
        # 步骤7: 生成权重
        weights = jax.vmap(jax.vmap(self.output_projection))(decoded)  # [B, K_pad, 1]
        weights = jnp.squeeze(weights, axis=-1)  # [B, K_pad]
        weights = jax.nn.softplus(weights)
        
        return weights, current_memory

# 原来的Seq2SeqTransformer（用于35维特征）
class Seq2SeqTransformerOld(eqx.Module):
    """序列到序列Transformer，支持状态传递（用于35维特征）"""
    encoder: eqx.Module
    input_proj: eqx.nn.Linear
    history_embedding: eqx.Module
    output_projection: eqx.nn.Linear
    
    def __init__(self, feature_dim, hidden_dim, num_layers, num_heads, key):
        # 分割随机密钥
        key_in, key_enc, key_hist, key_out = jax.random.split(key, 4)
        
        # 特征编码器 - 处理元素特征
        self.input_proj = eqx.nn.Linear(feature_dim, hidden_dim, key=key_in)
        self.encoder = TransformerStack(
            num_layers=max(1, num_layers // 2),
            model_dims=hidden_dim,
            num_heads=max(1, num_heads),
            key=key_enc,
        )
        
        # 历史信息嵌入
        self.history_embedding = eqx.nn.MLP(
            in_size=hidden_dim, 
            out_size=hidden_dim,
            width_size=hidden_dim * 2,
            depth=4,
            key=key_hist
        )
        
        # 输出投影 - 生成权重
        self.output_projection = eqx.nn.Linear(hidden_dim, 1, key=key_out)
    
    def __call__(self, x, memory=None, key=None):
        """
        x: [batch_size, seq_len, feature_dim] - 当前帧的元素特征
        memory: [batch_size, hidden_dim] - 上一时刻的历史信息
        """
        single_sample = (x.ndim == 2)
        if single_sample:
            x = jnp.expand_dims(x, 0)
        x_proj = self.input_proj(x)
        encoder_in = jnp.transpose(x_proj, (1, 0, 2))
        encoded = self.encoder(encoder_in, key=key)
        encoded = jnp.transpose(encoded, (1, 0, 2))
        if memory is not None:
            if memory.ndim == 1:
                memory = jnp.expand_dims(memory, 0)
            history_embed = self.history_embedding(memory)
            decoded = encoded + history_embed[:, None, :]
        else:
            decoded = encoded
        current_memory = jnp.mean(decoded, axis=1)
        weights = jax.vmap(jax.vmap(self.output_projection))(decoded)
        weights = jnp.squeeze(weights, axis=-1)
        if single_sample:
            return jnp.squeeze(weights, axis=0), jnp.squeeze(current_memory, axis=0)
        return weights, current_memory

def pad_batch(batch_inputs, batch_targets, use_time_encoder=True):
    """
    将可变长度的批处理样本对齐到统一长度
    返回: (features_padded, targets_padded, mask)
    """
    if not batch_inputs:
        return None, None, None
    
    # 找到最大的序列长度
    max_k = max(item['features'].shape[0] for item in batch_inputs)
    batch_size = len(batch_inputs)
    
    if use_time_encoder:
        # 时间编码器模式: [B, K_pad, H, 1]
        H = batch_inputs[0]['features'].shape[1]
        features_shape = (batch_size, max_k, H, 1)
    else:
        # 35维特征模式: [B, K_pad, 35]
        feature_dim = batch_inputs[0]['features'].shape[1]
        features_shape = (batch_size, max_k, feature_dim)
    
    # 初始化填充后的数组
    features_padded = jnp.zeros(features_shape, dtype=jnp.float32)
    targets_padded = jnp.zeros((batch_size, max_k), dtype=jnp.float32)
    mask = jnp.zeros((batch_size, max_k), dtype=jnp.bool_)
    
    # 填充每个样本
    for i, (inp, tgt) in enumerate(zip(batch_inputs, batch_targets)):
        k_i = inp['features'].shape[0]
        features_padded = features_padded.at[i, :k_i].set(inp['features'])
        targets_padded = targets_padded.at[i, :k_i].set(tgt['weights'])
        mask = mask.at[i, :k_i].set(True)
    
    return features_padded, targets_padded, mask

def train_seq2seq_model(
    dataloader, 
    feature_extractor, 
    model_params, 
    model_static,
    optimizer,
    opt_state, 
    train_batches,
    num_epochs=100,
    top_k: int = 0
):
    """使用滑动窗口进行序列到序列训练"""
    
    @eqx.filter_jit
    def loss_fn(params, static, features, targets, mask, memory=None, key=None):
        """计算一个批次的损失"""
        model = eqx.combine(params, static)
        
        # 前向传播 - 模型已支持批处理输入
        pred_weights, new_memory = model(features, memory, key=key)
        
        # 计算损失 (使用掩码处理填充)
        def single_loss(pred, target, sample_mask):
            mask_f = sample_mask.astype(jnp.float32)
            valid_count = jnp.sum(mask_f)
            safe_count = jnp.maximum(valid_count, 1.0)

            valid_pred = pred * mask_f
            valid_target = target * mask_f

            pred_sum = jnp.sum(valid_pred) + 1e-8
            target_sum = jnp.sum(valid_target) + 1e-8

            pred_norm = valid_pred / pred_sum
            target_norm = valid_target / target_sum

            diff_sq = ((pred_norm - target_norm) ** 2) * mask_f
            mse_loss = jnp.sum(diff_sq) / safe_count

            l1_reg = 1e-4 * jnp.sum(jnp.abs(valid_pred))
            rel_error = jnp.abs(pred_sum - target_sum) / target_sum

            total = mse_loss + 0.1 * rel_error + l1_reg
            return jnp.where(valid_count > 0, total, 0.0)

        # 对每个样本计算损失
        losses = jax.vmap(single_loss)(pred_weights, targets, mask)
        
        # 只对非空样本计算平均损失
        valid_samples = jnp.sum(mask, axis=1) > 0
        total_loss = jnp.where(jnp.sum(valid_samples) > 0, 
                              jnp.sum(losses * valid_samples) / jnp.sum(valid_samples),
                              0.0)
            
        return total_loss, new_memory
    
    @eqx.filter_jit
    def train_step(params, static, opt_state, features, targets, mask, memory=None, key=None):
        """执行一步训练"""
        (loss, new_memory), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(params, static, features, targets, mask, memory, key)
        
        # 梯度裁剪
        grad_norm = jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y ** 2),
            grads,
            0.0
        ) ** 0.5
        
        max_norm = 1.0
        scale = jnp.where(grad_norm > max_norm, max_norm / (grad_norm + 1e-8), 1.0)
        grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        
        # if grad_norm > max_norm:
        #     grads = jax.tree_util.tree_map(
        #         lambda g: g * max_norm / (grad_norm + 1e-8),
        #         grads
        #     )
        
        # 更新参数
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        
        return params, opt_state, loss, new_memory
    
    best_loss = float('inf')
    best_params = model_params
    key = jax.random.PRNGKey(42)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        # 对每个序列按顺序训练
        for batch_idx, batch_windows in enumerate(train_batches):
            key, subkey = jax.random.split(key)
            
            # 提取特征
            batch_inputs, batch_targets = feature_extractor.extract_batch_features(batch_windows)
            if not batch_inputs:
                continue
            
            # 对齐批处理数据
            features_padded, targets_padded, mask = pad_batch(
                batch_inputs, batch_targets, feature_extractor.use_time_encoder
            )
            
            if features_padded is None:
                continue
                
            # 初始无状态
            memory = None
            
            # 执行训练步骤
            model_params, opt_state, loss, memory = train_step(
                model_params, model_static, opt_state,
                features_padded, targets_padded, mask, memory, subkey
            )
            
            epoch_losses.append(float(loss))
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_batches)}: loss={float(loss):.6f}")
        
        # 计算epoch平均损失
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"avg_loss={avg_epoch_loss:.6f}, time={epoch_time:.2f}s")
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_params = model_params
            
    return best_params, opt_state

def main():
    parser = argparse.ArgumentParser(description="基于真正滑动窗口的序列到序列Transformer训练")
    parser.add_argument("--initial_options_dir", type=str, 
                       default=os.path.join(ROOT_DIR, "bunny_models/initial_cubature"), 
                       help="包含indices和weights文件的目录")
    parser.add_argument("--system_name", type=str, default="fem", help="系统名称")
    parser.add_argument("--problem_name", type=str, default="bunny", help="问题名称")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join(ROOT_DIR, "bunny_models/seq2seq_transformer"), 
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--history_len", type=int, default=4, help="历史窗口长度")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_time_encoder", dest="use_time_encoder", action="store_true",
                       help="使用time encoder提取时序特征")
    group.add_argument("--no_time_encoder", dest="use_time_encoder", action="store_false",
                       help="禁用time encoder，改用35维特征输入")
    parser.set_defaults(use_time_encoder=True)

    parser.add_argument("--top_k", type=int, default=0, help="启用 top-K 损失计算（0 表示禁用）")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Building FEM System")
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)
    fem_total_elements = system.mesh['E'].shape[0]
    print(f"FEM Network has {fem_total_elements} cubature elements")
    
    print("Initializing Sequential DataLoader...")
    dataloader = SequentialDataLoader(
        args.initial_options_dir, 
        args.history_len
    )
    
    feature_extractor = SlidingWindowFeatureExtractor(
        dataloader, 
        fem_total_elements,
        use_time_encoder=args.use_time_encoder
    )
    
    print("Generating training and validation batches...")
    train_batches, valid_batches = dataloader.generate_sliding_windows(
        batch_size=args.batch_size, 
        valid_ratio=0.2
    )
    
    print("Initializing Seq2Seq Transformer model...")
    key = jax.random.PRNGKey(42)
    if args.use_time_encoder:
        per_time_dim = 1
        model = Seq2SeqTransformer(
            per_time_dim=per_time_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            key=key,
            temp_layers=1
        )
    else:
        feature_dim = 35
        model = Seq2SeqTransformerOld(
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            key=key
        )
    
    model_params, model_static = eqx.partition(model, eqx.is_array)
    
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(model_params)
    
    print("Starting training...")
    best_params, final_opt_state = train_seq2seq_model(
        dataloader,
        feature_extractor,
        model_params,
        model_static,
        optimizer,
        opt_state,
        train_batches,
        num_epochs=args.epochs,
        top_k=args.top_k
    )
    
    print("Saving trained model...")
    final_model = eqx.combine(best_params, model_static)
    model_path = os.path.join(args.output_dir, "seq2seq_model.eqx")
    with open(model_path, "wb") as f:
        eqx.tree_serialise_leaves(f, final_model)
    
    config_dict = {
        'use_time_encoder': args.use_time_encoder,
        'top_k': args.top_k,
        'feature_dim': 35 if not args.use_time_encoder else None,
        'per_time_dim': 1 if args.use_time_encoder else None,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'history_len': args.history_len,
        'fem_total_elements': fem_total_elements,
    }
    
    config_path = os.path.join(args.output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Model trained and saved at {model_path}")

if __name__ == "__main__":
    main()
