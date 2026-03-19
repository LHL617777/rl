import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from model import Model2D2C
from occt_2d2c import TwoCarrierEnv
from scipy.interpolate import CubicSpline  # 用于复原样条曲线
import pandas as pd

# 设置 Matplotlib 样式
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def get_data_from_td(td, key_name):
    """智能查找数据 (兼容 next/info/key 和 next/key)"""
    path_1 = ("next", "info", key_name)
    path_2 = ("next", key_name)
    
    target = None
    if path_1 in td.keys(include_nested=True):
        target = td[path_1]
    elif path_2 in td.keys(include_nested=True):
        target = td[path_2]
    
    if target is None:
        return None

    # 处理 Batch 维度
    if target.ndim == 3 and target.shape[0] == 1:
        target = target.squeeze(0)
    elif target.ndim == 2 and target.shape[0] == 1:
        target = target.squeeze(0)
    
    return target.numpy()

def analyze_and_replay(pt_path, config_path=None, output_dir=None):
    print(f"📂 正在处理数据: {pt_path}")
    try:
        td = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return

    # ================= 1. 准备数据 =================
    full_states = get_data_from_td(td, "full_state")
    u1_seq = get_data_from_td(td, "u1")
    u2_seq = get_data_from_td(td, "u2_original")
    fh_vals = get_data_from_td(td, "Fh2")

    if full_states is None:
        print("❌ 无法生成视频：缺少 'full_state' 数据。")
        return

    steps = full_states.shape[0]
    print(f"⏱️ 轨迹长度: {steps} 帧")

    # 准备输出目录
    file_stem = os.path.splitext(os.path.basename(pt_path))[0]
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pt_path), f"analysis_{file_stem}")
    os.makedirs(output_dir, exist_ok=True)

    print("🗺️ 正在从数据中复原参考路径...")
    path_key_x = get_data_from_td(td, "path_key_x")
    path_key_y = get_data_from_td(td, "path_key_y")
    
    dense_x, dense_y = None, None
    if path_key_x is not None and path_key_y is not None:
        kx = path_key_x[0] if path_key_x.ndim > 1 else path_key_x
        ky = path_key_y[0] if path_key_y.ndim > 1 else path_key_y
        
        if np.any(kx != 0) or np.any(ky != 0):
            num_points = len(kx)
            ks = np.zeros(num_points)
            for i in range(1, num_points):
                ks[i] = ks[i-1] + np.hypot(kx[i] - kx[i-1], ky[i] - ky[i-1])
                
            spline_x = CubicSpline(ks, kx)
            spline_y = CubicSpline(ks, ky)
            
            path_s = np.linspace(0, ks[-1], 1000)
            dense_x = spline_x(path_s)
            dense_y = spline_y(path_s)
            print("✅ 参考路径复原成功！")

            # 【新增】导出参考路径到 CSV
            ref_path_df = pd.DataFrame({
                's_arc_length': path_s,
                'target_x': dense_x,
                'target_y': dense_y
            })
            ref_csv_path = os.path.join(output_dir, f"{file_stem}_target_path.csv")
            ref_path_df.to_csv(ref_csv_path, index=False)
            print(f"✅ 参考路径已保存至 CSV: {ref_csv_path}")

    # ================= 2. 先生成视频 (防止被删除) =================
    print("🎥 正在渲染视频...")
    
    env_kwargs = {"enable_visualization": False}
    if config_path: env_kwargs["config_path"] = config_path
    temp_env = TwoCarrierEnv(**env_kwargs)
    model = temp_env.model

    model.T = steps * model.dt
    model.N = steps
    model.x_arch = np.zeros((steps, model.N_x))
    model.u_arch = np.zeros((steps, model.N_u))
    model.Fh_arch = np.zeros((steps, 2 * model.N_c))

    model.x_arch = full_states
    if u1_seq is not None: model.u_arch[:, 0:4] = u1_seq
    if u2_seq is not None: model.u_arch[:, 4:8] = u2_seq
    if fh_vals is not None:
        model.Fh_arch[:, 2] = fh_vals[:, 0]
        model.Fh_arch[:, 3] = fh_vals[:, 1]
    
    if dense_x is not None and dense_y is not None:
        model.path_x = dense_x
        model.path_y = dense_y

    video_name = f"{file_stem}_replay.mp4"
    print(f"=== 调试信息 ===")
    print(f"轨迹长度 steps: {steps}")
    print(f"model.N: {model.N}, model.T: {model.T}, model.dt: {model.dt}")
    print(f"x_arch形状: {model.x_arch.shape}, u_arch形状: {model.u_arch.shape}")
    print(f"计算理论步数: {model.T / model.dt}")
    try:
        model.generateVideo(output_dir, video_name)
        print(f"✅ 视频已生成: {os.path.join(output_dir, video_name)}")
    except Exception as e:
        print(f"❌ 视频生成失败: {e}")
        import traceback
        traceback.print_exc()

    print("📝 正在提取并保存实际行驶轨迹至 CSV...")
    front_x, front_y = [], []
    rear_x, rear_y = [], []
    cargo_x = full_states[:, 0]
    cargo_y = full_states[:, 1]
    
    for i in range(steps):
        fx, fy = model.getXYi(full_states[i], 0)
        rx, ry = model.getXYi(full_states[i], 1)
        front_x.append(fx)
        front_y.append(fy)
        rear_x.append(rx)
        rear_y.append(ry)
        
    actual_traj_df = pd.DataFrame({
        'step': np.arange(steps),
        'time_s': np.arange(steps) * model.dt,
        'front_car_x': front_x,
        'front_car_y': front_y,
        'cargo_x': cargo_x,
        'cargo_y': cargo_y,
        'rear_car_x': rear_x,
        'rear_car_y': rear_y
    })
    traj_csv_path = os.path.join(output_dir, f"{file_stem}_actual_trajectory.csv")
    actual_traj_df.to_csv(traj_csv_path, index=False)
    print(f"✅ 实际轨迹已保存至 CSV: {traj_csv_path}")
    
    # ================= 3. 后生成图表 (安全区域) =================
    print("📊 正在生成分析图表...")
    
    # 定义要抓取的数据
    reward_map = {
        "Force Reward": "reward_r_force",
        "Align Rear": "reward_r_align_rear", 
        "Align Front": "reward_r_align_front",
        "Progress": "reward_r_progress",
        "Stability": "reward_r_stability",
        "Total Reward": "episode_reward"
    }
    
    val_map = {
        "Force Mag (N)": "reward_val_force",
        "Rear Angle (rad)": "reward_val_delta_psi_rear",
        "Front Angle (rad)": "reward_val_delta_psi_front"
    }

    # 提取有效数据
    rewards_data = {}
    for label, key in reward_map.items():
        val = get_data_from_td(td, key)
        if val is not None: rewards_data[label] = val

    vals_data = {}
    for label, key in val_map.items():
        val = get_data_from_td(td, key)
        if val is not None: vals_data[label] = val

    # --- 图1: 奖励拆分图 (每个奖励一张子图) ---
    if rewards_data:
        n_plots = len(rewards_data)
        # 动态调整高度，每个子图高3英寸
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
        if n_plots == 1: axes = [axes]
        
        for ax, (label, data) in zip(axes, rewards_data.items()):
            ax.plot(data, linewidth=1.5, color='#1f77b4') # 统一蓝色
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_ylabel("Reward")
            
            # 如果是 Progress，画一条 0 线
            if "Progress" in label:
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rewards_breakdown.png"), dpi=150)
        plt.close()

    # --- 图2: 物理数值图 (保持独立) ---
    if vals_data:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        
        # 1. Force
        if "Force Mag (N)" in vals_data:
            axes[0].plot(vals_data["Force Mag (N)"], color='orange')
            axes[0].set_title("Hinge Force Magnitude")
            axes[0].set_ylabel("Force (N)")
        
        # 2. Rear Angle
        if "Rear Angle (rad)" in vals_data:
            axes[1].plot(vals_data["Rear Angle (rad)"], color='green')
            axes[1].axhline(0.4, color='r', ls='--', label='Limit')
            axes[1].axhline(-0.4, color='r', ls='--')
            axes[1].set_title("Rear Alignment (Agent)")
            axes[1].set_ylabel("Rad")
            axes[1].legend()

        # 3. Front Angle
        if "Front Angle (rad)" in vals_data:
            axes[2].plot(vals_data["Front Angle (rad)"], color='purple')
            axes[2].axhline(0.4, color='r', ls='--', label='Limit')
            axes[2].axhline(-0.4, color='r', ls='--')
            axes[2].set_title("Front Alignment (System)")
            axes[2].set_ylabel("Rad")
            axes[2].legend()
        
        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "physics_breakdown.png"), dpi=150)
        plt.close()
    
    print(f"✅ 所有文件已保存至: {output_dir}")
    
    # 自动打开文件夹
    if os.name == 'nt':
        try:
            os.startfile(output_dir)
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze rewards AND generate video from .pt file")
    parser.add_argument("--pt", type=str, required=True, help="Path to the saved .pt trajectory file")
    parser.add_argument("--config", type=str, default=None, help="Optional path to 2d2c.yaml")
    args = parser.parse_args()
    
    analyze_and_replay(args.pt, args.config)