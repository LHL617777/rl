# 离线仿真脚本：加载Checkpoint复现OCCT环境策略
# 适配：TwoCarrierEnv (Spline Tracking 版本)
# 功能：加载模型 -> 运行仿真 -> 【重点】统计各项Reward构成 -> 生成视频
# 【兼容性】完全匹配 run_simulation.bat 的参数接口
from __future__ import annotations
import sys
import os
import warnings
import pickle
import numpy as np
import torch
import argparse
import omegaconf
from omegaconf import DictConfig
from tqdm import tqdm
from collections import defaultdict

# ===================== 预处理 =====================
torch.serialization.add_safe_globals([DictConfig, omegaconf.dictconfig.DictConfig])
warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path: sys.path.insert(0, current_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)

try:
    from utils_ppo_occt import make_ppo_models
    from occt_2d2c import TwoCarrierEnv
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ===================== 工具函数 =====================

def create_raw_env(cfg, render_mode, config_path, enable_visualization):
    # 优先使用传入的 config_path，其次尝试从 cfg 中获取
    env_config_path = getattr(cfg.env, 'config_path', None) if hasattr(cfg, 'env') else None
    final_config_path = config_path if config_path is not None else env_config_path
    
    print(f"🛠️ 创建环境 | Render: {render_mode} | Vis: {enable_visualization} | Config: {final_config_path}")
    
    env = TwoCarrierEnv(
        render_mode=render_mode,
        config_path=final_config_path,
        enable_visualization=enable_visualization
    )
    return env

def extract_deterministic_action(model_output, device):
    # 如果输出是 TensorDict (TorchRL 常见输出)，直接取 loc 或 action
    if hasattr(model_output, "get"):
        return model_output.get("action")
    
    # 如果是元组，通常 (action, log_prob)
    if isinstance(model_output, tuple):
        return model_output[0]
        
    return model_output

# ===================== 核心逻辑 =====================

def load_checkpoint(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"📥 Loading checkpoint: {ckpt_path}")
    map_location = torch.device(device)
    
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False, pickle_module=pickle)
    except Exception:
        with torch.serialization.safe_globals([DictConfig]):
            ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
            
    cfg = ckpt["cfg"]
    actor, _ = make_ppo_models(cfg.env.env_name, device=map_location)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    
    print(f"✅ Model loaded. Frames: {ckpt.get('collected_frames', 'N/A')}")
    return actor, cfg

def run_simulation(args):
    # 处理设备参数
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    
    actor, cfg = load_checkpoint(args.ckpt_path, device)
    
    # 根据参数决定渲染模式
    render_mode = "rgb_array" if args.enable_visualization else None
    
    env = create_raw_env(
        cfg, 
        render_mode,
        args.config_path,
        args.enable_visualization
    )
    

    # 视频保存逻辑
    video_dir = args.video_dir if args.video_dir else os.path.join(os.path.dirname(args.ckpt_path), "videos")
    # 注意：这里严格遵循 args.save_video 开关
    if args.save_video and args.enable_visualization:
        os.makedirs(video_dir, exist_ok=True)
        env.clear_render_frames()

    # --- 统计数据容器 ---
    global_stats = {
        "rewards": [],
        "lengths": [],
        "hinge_forces": [],
        "reward_details": defaultdict(list)
    }

    # 你的 Reward 关注列表
    TARGET_REWARD_KEYS = [
        "reward_r_progress",      
        "reward_r_force",         
        "reward_r_align_rear",    
        "reward_r_align_front",   
        "reward_r_stability",     
        "reward_r_smooth"         
    ]

    print(f"\n🚀 Starting Simulation for {args.num_episodes} episodes...")
    
    for ep in range(args.num_episodes):
        obs, _ = env.reset(seed=42 + ep, options={"clear_frames": True})
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        
        ep_reward = 0
        ep_steps = 0
        ep_hinge_forces = []
        ep_reward_breakdown = defaultdict(float)
        
        terminated = False
        truncated = False
        
        pbar = tqdm(total=1024, desc=f"Episode {ep+1}/{args.num_episodes}", leave=False)
        
        while not (terminated or truncated):
            with torch.no_grad():
                model_out = actor(obs_tensor.unsqueeze(0))
                action = extract_deterministic_action(model_out, device)
                action_np = action.squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            ep_reward += reward
            ep_steps += 1
            
            hf = info.get('Fh2', [0, 0])
            hf_mag = np.hypot(hf[0], hf[1])
            ep_hinge_forces.append(hf_mag)
            
            # 累计各项细分 Reward
            for key in TARGET_REWARD_KEYS:
                val = info.get(key, 0.0)
                if isinstance(val, (np.ndarray, list)):
                    val = float(val) if np.ndim(val)==0 else float(val[0])
                elif hasattr(val, 'item'):
                    val = val.item()
                ep_reward_breakdown[key] += val
            
            pbar.update(1)
            pbar.set_postfix({
                "Total": f"{ep_reward:.0f}", 
                "Prog": f"{ep_reward_breakdown['reward_r_progress']:.0f}",
                "Force": f"{ep_reward_breakdown['reward_r_force']:.0f}"
            })

        pbar.close()
        
        # 打印单轮报告
        stats_str = f"  Episode {ep+1} Report:\n"
        stats_str += f"    Total Reward: {ep_reward:.2f} (Steps: {ep_steps})\n"
        stats_str += f"    ---------------- Reward Breakdown ----------------\n"
        for key in TARGET_REWARD_KEYS:
            val = ep_reward_breakdown[key]
            global_stats["reward_details"][key].append(val)
            stats_str += f"    {key.replace('reward_', ''):<15}: {val:.2f}\n"
        stats_str += f"    --------------------------------------------------\n"
        stats_str += f"    Avg Hinge Force: {np.mean(ep_hinge_forces):.2f} N\n"
        print(stats_str)

        global_stats["rewards"].append(ep_reward)
        global_stats["lengths"].append(ep_steps)
        global_stats["hinge_forces"].extend(ep_hinge_forces)

        # 视频保存逻辑：仅当 .bat 开启了 SAVE_VIDEO 时才执行
        if args.save_video and args.enable_visualization:
            video_name = f"ep{ep+1}_r{int(ep_reward)}.mp4"
            save_path = env.save_eval_video(eval_round=f"offline_{ep+1}", video_save_dir=video_dir)
            if save_path:
                try:
                    new_path = os.path.join(video_dir, video_name)
                    if os.path.exists(new_path): os.remove(new_path)
                    os.rename(save_path, new_path)
                except OSError:
                    pass

        env.mark_sim_finished()

    env.close()
    
    # 最终报告
    print("\n" + "="*50)
    print("📊 FINAL SIMULATION REPORT (Average over episodes)")
    print("="*50)
    print(f"Total Reward:   {np.mean(global_stats['rewards']):.2f} ± {np.std(global_stats['rewards']):.2f}")
    print("-" * 50)
    print("Reward Components Breakdown (Mean):")
    for key in TARGET_REWARD_KEYS:
        vals = global_stats["reward_details"][key]
        if len(vals) > 0:
            print(f"  {key.replace('reward_', ''):<15}: {np.mean(vals):.2f}")
    print("-" * 50)
    print(f"Hinge Force:    Mean={np.mean(global_stats['hinge_forces']):.1f}, Max={np.max(global_stats['hinge_forces']):.1f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCCT Offline Simulation (Reward Analysis)")
    
    # === 参数定义严格对应 .bat 文件 ===
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda:0)")
    parser.add_argument("--enable_visualization", action="store_true", help="Enable rendering")
    parser.add_argument("--save_video", action="store_true", help="Enable video saving")
    parser.add_argument("--video_dir", type=str, default="", help="Directory to save videos")
    parser.add_argument("--config_path", type=str, default=None, help="Path to 2d2c.yaml")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        args.device = "cpu"
        # print("⚠️ CUDA not available, switching to CPU.")

    run_simulation(args)