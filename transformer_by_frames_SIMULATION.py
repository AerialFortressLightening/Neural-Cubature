import os
import sys
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import polyscope as ps
import polyscope.imgui as psim

script_dir = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(script_dir, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

model_dir = os.path.join(ROOT_DIR, "bunny_models", "seq2seq_transformer")
model_path = os.path.join(model_dir, "seq2seq_model.eqx")
config_path = os.path.join(model_dir, "model_config.json")

import config
import integrators
import minimize
import fem_model as fem
from transformer_by_frames import (
    Seq2SeqTransformer, Seq2SeqTransformerOld,
    SequentialDataLoader, SlidingWindowFeatureExtractor,
    pad_batch
)

# ========== 核心工具函数 ==========

def compute_element_contrib(system, system_def, pos):
    """计算全量元素能量贡献（禁用 cubature）"""
    sd = dict(system_def)
    sd.pop('cubature_inds', None)
    sd.pop('cubature_weights', None)
    return np.asarray(
        fem.fem_energy_element(sd, system.mesh, system.material_energy, jnp.array(pos))
    )

def rbf_weights(verts, center_id, radius):
    """径向基函数权重"""
    center = verts[center_id]
    d = np.linalg.norm(verts - center[None, :], axis=1)
    w = np.exp(- (d / (radius + 1e-8)) ** 2)
    w[d > radius] = 0.0
    return w / (w.max() + 1e-12) if w.max() > 0 else w

def update_highlight_volume_mesh(name, pos, tets, contrib_bin, contrib_score, alpha=0.15):
    """更新高亮体积网格"""
    try:
        vm = ps.get_volume_mesh(name)
        vm.update_vertex_positions(np.asarray(pos))
    except ValueError:
        vm = ps.register_volume_mesh(name, np.asarray(pos), np.asarray(tets))
        vm.set_transparency(alpha)
    vm.add_scalar_quantity("contrib_mask", contrib_bin, defined_on='cells', enabled=True)
    vm.add_scalar_quantity("contrib_score", contrib_score, defined_on='cells', enabled=False)

# ========== 滑动窗口推理逻辑 ==========

class SlidingWindowInference:
    """滑动窗口模型在线推理器"""
    
    def __init__(self, model, model_config, dataloader, fem_total_elements):
        self.model = model
        self.config = model_config
        self.dataloader = dataloader
        self.fem_total_elements = fem_total_elements
        self.history_len = model_config['history_len']
        
        # 特征提取器（与训练时完全一致）
        self.feature_extractor = SlidingWindowFeatureExtractor(
            dataloader, 
            fem_total_elements,
            use_time_encoder=model_config['use_time_encoder']
        )
        
        # 历史缓存
        self.frame_history = []  # 存储最近 history_len 帧的帧号
        self.memory = None       # 模型的隐状态
        
    def update_and_predict(self, current_frame, target_indices=None):
        """
        更新历史窗口并预测当前帧的权重
        
        Args:
            current_frame: 当前帧号
            target_indices: 如果提供，只预测这些元素的权重；否则预测全部
        
        Returns:
            predicted_weights: [K] or [M]，预测的权重
            selected_indices: [K]，对应的元素索引
        """
        # 更新历史窗口
        self.frame_history.append(current_frame)
        if len(self.frame_history) > self.history_len:
            self.frame_history.pop(0)
        
        # 检查是否有足够的历史
        if len(self.frame_history) < self.history_len:
            print(f"[Warning] 历史不足 ({len(self.frame_history)}/{self.history_len})，跳过预测")
            return None, None
        
        # 构造输入窗口（使用与训练时完全相同的逻辑）
        input_window = self.frame_history[:-1]  # 前 H-1 帧
        target_frame = self.frame_history[-1]   # 第 H 帧
        
        # 提取特征
        batch_inputs, batch_targets = self.feature_extractor.extract_batch_features(
            [(input_window, target_frame)]
        )
        
        if not batch_inputs:
            print(f"[Warning] 帧 {target_frame} 数据缺失")
            return None, None
        
        # 获取目标元素索引
        input_features = batch_inputs[0]['features']
        indices = batch_inputs[0]['indices']
        
        # 模型推理
        features_batched = jnp.expand_dims(input_features, axis=0)  # [1, K, H, 1] or [1, K, 35]

        memory_in = None
        if self.memory is not None:
            memory_in = jnp.asarray(self.memory)
            if memory_in.ndim == 1:
                memory_in = memory_in[None, :]
            if memory_in.shape[0] != features_batched.shape[0]:
                if memory_in.shape[0] == 1:
                    memory_in = jnp.repeat(memory_in, features_batched.shape[0], axis=0)
                else:
                    memory_in = memory_in[:features_batched.shape[0], :]
            memory_in = memory_in.astype(features_batched.dtype)

        pred_weights, new_memory = self.model(features_batched, memory_in, key=None)
        self.memory = jnp.asarray(new_memory)
        if self.memory.ndim == 1:
            self.memory = self.memory[None, :]
        self.memory = self.memory.astype(features_batched.dtype)

        pred_weights = jnp.squeeze(pred_weights, axis=0)  # [K]
        return pred_weights, indices
    
    def reset(self):
        """重置历史和状态"""
        self.frame_history = []
        self.memory = None

# ========== 主可视化循环 ==========

def main():
    # ---------- 1. 加载系统和模型 ----------
    print("=== 加载 FEM 系统 ===")
    _ = jnp.zeros(())
    system, system_def = config.construct_system_from_name("fem", "bunny")
    verts0 = np.asarray(system.mesh["Vrest"])
    tets = np.asarray(system.mesh["E"])
    fem_total_elements = tets.shape[0]
    
    print("=== 加载 Seq2Seq Transformer 模型 ===")
    model_dir = os.path.join(ROOT_DIR, "bunny_models", "seq2seq_transformer")
    model_path = os.path.join(model_dir, "seq2seq_model.eqx")
    config_path = os.path.join(model_dir, "model_config.json")
    
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # 重建模型结构
    key = jax.random.PRNGKey(0)
    if model_config['use_time_encoder']:
        model = Seq2SeqTransformer(
            per_time_dim=model_config['per_time_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            key=key,
            temp_layers=1
        )
    else:
        model = Seq2SeqTransformerOld(
            feature_dim=model_config['feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            key=key
        )
    
    # 加载训练好的权重
    with open(model_path, "rb") as f:
        model = eqx.tree_deserialise_leaves(f, model)
    
    print("=== 初始化数据加载器 ===")
    initial_options_dir = os.path.join(ROOT_DIR, "bunny_models", "initial_cubature")
    dataloader = SequentialDataLoader(initial_options_dir, model_config['history_len'])
    
    # 创建推理器
    inference_engine = SlidingWindowInference(
        model, model_config, dataloader, fem_total_elements
    )
    
    # ---------- 2. 设置交互参数 ----------
    total_frames = 60
    k_spring = 80.0
    _scale = np.asarray(system.mesh.get("scale", np.array([1.0])))
    grab_radius = float(_scale.max()) / 4.0
    lift_vec = np.array([0.00, 0.20, 0.05])
    t_start, t_end = 5, 35
    
    top_y_id = int(np.argmax(verts0[:, 1]))
    mask_full = np.zeros((verts0.shape[0],), dtype=float)
    
    system_def['k'] = k_spring
    system_def['interaction_mask'] = jnp.asarray(mask_full.copy())
    system_def['target'] = jnp.asarray(verts0.copy())
    system_def['grab_r'] = grab_radius
    
    # ---------- 3. 初始化积分器 ----------
    int_opts = {'timestep_h': 0.01}
    int_state = {}
    integrators.initialize_integrator(int_opts, int_state, "implicit-proximal")
    
    int_state['q_t'] = system_def['init_pos']
    int_state['q_tm1'] = int_state['q_t']
    int_state['qdot_t'] = jnp.zeros_like(int_state['q_t'])
    int_state['potential'] = system.potential_energy(system_def, int_state['q_t'])
    
    # ---------- 4. Polyscope 初始化 ----------
    ps.init()
    ps.set_ground_plane_mode('none')
    ps.look_at((2., 1., 2.), (0., 0., 0.))
    
    pos_full = np.asarray(system.get_full_position(system, system_def, int_state['q_t']))
    ps_mesh = system.visualize(system_def, int_state['q_t'], prefix="")
    
    # ---------- 5. UI 状态 ----------
    running = False
    run_fixed_steps = 0
    top_percent = 10.0
    use_sliding_window_pred = True
    apply_to_solver = False
    frame = 0
    last_frame = 0
    last_t = time.time()
    
    # 可视化缓存
    predicted_weights_cache = None
    predicted_indices_cache = None
    
    def ui_loop():
        nonlocal running, run_fixed_steps, top_percent, frame, last_frame, last_t
        nonlocal pos_full, use_sliding_window_pred, apply_to_solver
        nonlocal predicted_weights_cache, predicted_indices_cache
        
        if psim.TreeNode("Sliding Window Transformer Visualization"):
            psim.TextUnformatted(f"frame: {frame} / {total_frames}")
            _, running = psim.Checkbox("run", running)
            psim.SameLine()
            if psim.Button("single step"):
                run_fixed_steps = 1
            psim.SameLine()
            if psim.Button(f"run {total_frames}"):
                run_fixed_steps = total_frames
            
            _, top_percent = psim.SliderFloat("top % highlight", top_percent, 1.0, 50.0)
            _, use_sliding_window_pred = psim.Checkbox("Use Sliding Window Prediction", use_sliding_window_pred)
            _, apply_to_solver = psim.Checkbox("Apply to solver", apply_to_solver)
            
            if psim.Button("Reset Model Memory"):
                inference_engine.reset()
                print("[UI] 模型历史已重置")
            
            psim.TreePop()
        
        do_step = running or (run_fixed_steps > 0)
        if do_step and frame < total_frames:
            # ---------- 滑动窗口预测 ----------
            if use_sliding_window_pred:
                pred_w, pred_idx = inference_engine.update_and_predict(frame)
                
                if pred_w is not None and pred_idx is not None:
                    predicted_weights_cache = pred_w
                    predicted_indices_cache = pred_idx
                    
                    nsel = int(pred_idx.shape[0])
                    avg_weight = float(jnp.mean(pred_w))
                    print(f"[Frame {frame}] 预测 {nsel} 个元素，平均权重={avg_weight:.4f}")
                    
                    # 应用到求解器
                    if apply_to_solver and nsel > 0:
                        # 稀疏化：只保留 Top-K
                        topk = min(220, nsel)
                        sel_topk = jnp.argsort(pred_w)[::-1][:topk]
                        
                        system_def['cubature_inds'] = pred_idx[sel_topk].astype(jnp.int32)
                        system_def['cubature_weights'] = pred_w[sel_topk].astype(jnp.float32)
                        print(f"    应用到求解器：{topk} 个 cubature 点")
                    else:
                        system_def.pop('cubature_inds', None)
                        system_def.pop('cubature_weights', None)
                else:
                    system_def.pop('cubature_inds', None)
                    system_def.pop('cubature_weights', None)
            else:
                system_def.pop('cubature_inds', None)
                system_def.pop('cubature_weights', None)
            
            # ---------- 交互控制 ----------
            if frame == t_start:
                w = rbf_weights(pos_full, center_id=top_y_id, radius=grab_radius)
                system_def['interaction_mask'] = jnp.asarray(w)
                system_def['target'] = jnp.asarray(pos_full) + jnp.asarray(lift_vec)
                system_def['k'] = k_spring
            elif frame == t_end:
                system_def['interaction_mask'] = jnp.zeros_like(system_def['interaction_mask'])
                system_def['target'] = jnp.asarray(pos_full)
            
            # ---------- 物理仿真步进 ----------
            int_state_local, solver_info = integrators.timestep(
                system, system_def, int_state, int_opts
            )
            int_state.update(int_state_local)
            
            pos_full = np.asarray(system.get_full_position(system, system_def, int_state['q_t']))
            system.visualize(system_def, int_state['q_t'])
            
            # ---------- 能量贡献可视化 ----------
            contrib = compute_element_contrib(system, system_def, pos_full)
            perc = np.percentile(contrib, 100.0 - float(top_percent))
            contrib_mask = (contrib >= perc).astype(float)
            
            # 叠加预测结果高亮
            if predicted_weights_cache is not None and predicted_indices_cache is not None:
                pred_mask = np.zeros(fem_total_elements, dtype=float)
                pred_mask[np.asarray(predicted_indices_cache)] = np.asarray(predicted_weights_cache)
                # 归一化到 [0, 1] 用于颜色映射
                pred_mask = pred_mask / (pred_mask.max() + 1e-8)
                
                update_highlight_volume_mesh(
                    name="bunny_vol_prediction",
                    pos=pos_full,
                    tets=tets,
                    contrib_bin=pred_mask > 0.1,  # 二值化
                    contrib_score=pred_mask,      # 连续值
                    alpha=0.25
                )
            
            update_highlight_volume_mesh(
                name="bunny_vol_highlight",
                pos=pos_full,
                tets=tets,
                contrib_bin=contrib_mask,
                contrib_score=contrib,
                alpha=0.18
            )
            
            minimize.print_solver_info(solver_info)
            
            frame += 1
            run_fixed_steps -= 1
            
            fps = (frame - last_frame) * int_opts['timestep_h'] / (time.time() - last_t + 1e-9)
            psim.TextUnformatted(f"sim-to-real time rate: {fps:.3f}x")
            last_frame = frame
            last_t = time.time()
    
    ps.set_user_callback(ui_loop)
    ps.show()

if __name__ == "__main__":
    main()
