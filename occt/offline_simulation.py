# 离线仿真脚本：加载Checkpoint复现OCCT环境策略
# 完全适配TwoCarrierEnv源码版：精准匹配环境初始化/交互/可视化逻辑
from __future__ import annotations
import sys
import os
import warnings
import pickle
import numpy as np
import torch
import argparse
import omegaconf
from omegaconf import DictConfig, OmegaConf

# ===================== 预处理：屏蔽无关警告 + 环境配置 =====================
# 1. 允许OmegaConf.DictConfig被反序列化
torch.serialization.add_safe_globals([DictConfig, omegaconf.dictconfig.DictConfig])
# 2. 屏蔽所有无关警告
warnings.filterwarnings("ignore")
# 3. 检查CUDA环境
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"🔍 CUDA available: {CUDA_AVAILABLE}")
if not CUDA_AVAILABLE:
    print(f"   → 强制使用CPU加载Checkpoint和运行仿真")

# ===================== 核心：添加你的代码根目录 =====================
sys.path.insert(0, "E:\\rl")  

# ===================== 导入必需模块（基于TwoCarrierEnv源码） =====================
from utils_ppo_occt import make_ppo_models  # 仅导入模型创建
from occt_2d2c import TwoCarrierEnv  # 直接导入原始环境

# ===================== 工具函数：创建原始TwoCarrierEnv环境（精准匹配源码） =====================
def create_raw_env(cfg, render_mode, config_path, enable_visualization, vecnorm_frozen):
    """
    精准创建TwoCarrierEnv环境（完全匹配源码初始化参数）
    :param cfg: 配置字典/对象
    :param render_mode: 渲染模式 "human"/"rgb_array"/None
    :param config_path: 2d2c.yaml配置文件路径
    :param enable_visualization: 是否启用可视化
    :param vecnorm_frozen: 是否冻结VecNorm统计量
    :return: TwoCarrierEnv实例
    """
    # 从cfg中提取config_path（优先使用cfg中的配置）
    env_config_path = None
    if hasattr(cfg, 'env') and hasattr(cfg.env, 'config_path'):
        env_config_path = cfg.env.config_path
    
    # 优先级：显式传入的config_path > cfg中的config_path > None（使用默认）
    final_config_path = config_path if config_path is not None else env_config_path
    
    # 精准初始化TwoCarrierEnv（完全匹配源码参数）
    env = TwoCarrierEnv(
        render_mode=render_mode,
        config_path=final_config_path,
        enable_visualization=enable_visualization,
        vecnorm_frozen=vecnorm_frozen
    )
    return env

# ===================== 工具函数：提取模型输出的动作张量（适配4维动作空间） =====================
def extract_action_from_model_output(model_output, device):
    """
    从模型输出中提取4维动作张量，并确保归一化到[-1,1]区间
    :param model_output: 模型输出（张量/元组）
    :param device: 运行设备
    :return: 4维动作张量（归一化到[-1,1]）
    """
    # 提取动作张量
    if isinstance(model_output, torch.Tensor):
        action_tensor = model_output
    elif isinstance(model_output, tuple):
        action_tensor = model_output[0]  # PPO模型取第一个元素（动作）
    else:
        action_tensor = torch.tensor(model_output, device=device)
    
    # 确保是4维动作（匹配环境的动作空间）
    if action_tensor.shape[-1] != 4:
        raise ValueError(f"模型输出动作维度错误，期望4维，实际{action_tensor.shape[-1]}维")
    
    # 裁剪到[-1,1]（确保符合环境的归一化动作空间要求）
    action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
    
    return action_tensor

# ===================== 命令行参数配置（适配环境特性） =====================
def parse_args():
    parser = argparse.ArgumentParser(description="Offline Simulation for OCCT TwoCarrierEnv Policy")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint文件路径")
    parser.add_argument("--num_episodes", type=int, default=5, help="仿真轮数")
    parser.add_argument("--device", type=str, default="cuda:0" if CUDA_AVAILABLE else "cpu", help="运行设备")
    parser.add_argument("--enable_visualization", action="store_true", help="启用可视化")
    parser.add_argument("--render_mode", type=str, default="rgb_array", choices=["human", "rgb_array"], 
                        help="渲染模式：human(实时显示)/rgb_array(保存视频)")
    parser.add_argument("--save_video", action="store_true", help="保存视频（需启用可视化+rgb_array模式）")
    parser.add_argument("--video_dir", type=str, default="", help="视频保存目录")
    parser.add_argument("--config_path", type=str, default=None, help="2d2c.yaml配置文件路径")
    return parser.parse_args()

# ===================== 核心函数：加载Checkpoint（包含VecNorm状态） =====================
def load_checkpoint(ckpt_path, device):
    """
    加载Checkpoint，包含模型权重和VecNorm状态
    :param ckpt_path: Checkpoint路径
    :param device: 运行设备
    :return: actor模型, cfg配置, vecnorm_state归一化状态
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint不存在: {ckpt_path}")
    
    map_location = torch.device(device) if CUDA_AVAILABLE else torch.device("cpu")
    print(f"📥 加载Checkpoint到设备: {map_location}")
    
    # 加载Checkpoint（兼容PyTorch 2.6+）
    try:
        ckpt_dict = torch.load(
            ckpt_path, 
            map_location=map_location,
            weights_only=False,
            pickle_module=pickle
        )
    except Exception as e:
        print(f"⚠️ 初始加载失败，尝试降级方案：{e}")
        with torch.serialization.safe_globals([DictConfig]):
            ckpt_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    
    # 恢复模型
    cfg = ckpt_dict["cfg"]
    actor, _ = make_ppo_models(cfg.env.env_name, device=map_location)
    actor.load_state_dict(ckpt_dict["actor_state_dict"])
    actor.eval()
    
    # 提取VecNorm状态（兼容不同的存储方式）
    vecnorm_state = ckpt_dict.get("vecnorm_state", {
        "vecnorm_mean": ckpt_dict.get("vecnorm_mean", np.zeros(12, dtype=np.float64)),
        "vecnorm_var": ckpt_dict.get("vecnorm_var", np.ones(12, dtype=np.float64) * 1e-4),
        "vecnorm_count": ckpt_dict.get("vecnorm_count", 0),
        "vecnorm_decay": ckpt_dict.get("vecnorm_decay", 0.99999),
        "vecnorm_eps": ckpt_dict.get("vecnorm_eps", 1e-2),
        "vecnorm_frozen": ckpt_dict.get("vecnorm_frozen", True)
    })
    
    print(f"✅ Checkpoint加载成功!")
    print(f"  - 训练帧数: {ckpt_dict.get('collected_frames', 'unknown')}")
    print(f"  - VecNorm均值（前5维）: {vecnorm_state['vecnorm_mean'][:5].round(6)}")
    print(f"  - VecNorm模式: {'评测模式（冻结）' if vecnorm_state['vecnorm_frozen'] else '训练模式（解冻）'}")
    
    return actor, cfg, vecnorm_state

# ===================== 核心函数：运行离线仿真（完全适配TwoCarrierEnv） =====================
def run_offline_simulation(actor, cfg, vecnorm_state, args):
    final_device = torch.device(args.device) if CUDA_AVAILABLE else torch.device("cpu")
    print(f"\n🚀 开始离线仿真（TwoCarrierEnv）")
    print(f"  运行设备: {final_device}")
    print(f"  仿真轮数: {args.num_episodes}")
    print(f"  可视化: {'启用' if args.enable_visualization else '禁用'}")
    print(f"  渲染模式: {args.render_mode if args.enable_visualization else 'None'}")
    print(f"  视频保存: {'启用' if (args.save_video and args.enable_visualization and args.render_mode == 'rgb_array') else '禁用'}")
    
    # 1. 创建原始环境（精准匹配TwoCarrierEnv初始化参数）
    sim_env = create_raw_env(
        cfg=cfg,
        render_mode=args.render_mode if args.enable_visualization else None,
        config_path=args.config_path,
        enable_visualization=args.enable_visualization,
        vecnorm_frozen=vecnorm_state["vecnorm_frozen"]
    )
    print(f"✅ 创建原始环境成功: {type(sim_env)}")
    
    # 2. 加载并设置VecNorm状态（完全匹配环境的set_vecnorm_state方法）
    try:
        sim_env.set_vecnorm_state(vecnorm_state)
        sim_env.freeze_vecnorm()  # 强制冻结（评测模式）
        print(f"✅ VecNorm状态加载并冻结完成")
    except Exception as e:
        print(f"⚠️ VecNorm状态加载失败: {e}")
    
    # 3. 视频保存配置
    video_save_dir = None
    if args.save_video and args.enable_visualization and args.render_mode == "rgb_array":
        video_save_dir = args.video_dir if args.video_dir else os.path.dirname(args.ckpt_path)
        os.makedirs(video_save_dir, exist_ok=True)
        print(f"✅ 视频保存目录: {video_save_dir}")
        sim_env.clear_render_frames()  # 清空帧缓存
    else:
        print(f"ℹ️ 视频保存条件未满足（需同时启用可视化+rgb_array模式+save_video参数）")
    
    # 4. 多轮仿真（标准Gym接口，完全匹配TwoCarrierEnv）
    episode_rewards = []
    episode_lengths = []
    episode_hinge_forces = []  # 记录每轮铰接力
    
    with torch.no_grad():  # 禁用梯度计算（评测模式）
        for episode_idx in range(args.num_episodes):
            print(f"\n--- Episode {episode_idx+1} ---")
            
            # 重置环境（精准匹配reset参数）
            obs, info = sim_env.reset(
                seed=42 + episode_idx,  # 固定种子，保证可复现
                options={"clear_frames": True},
                clear_frames=True
            )
            # 转换obs为Tensor（模型输入需要）
            obs = torch.tensor(obs, dtype=torch.float32, device=final_device)
            
            # 清空帧缓存（仅当启用可视化时）
            if args.enable_visualization:
                sim_env.clear_render_frames()
            
            ep_reward = 0.0
            ep_length = 0
            ep_hinge_forces = []  # 记录本轮每步铰接力
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # a. 准备模型输入（添加batch维度: [12] → [1, 12]）
                obs_batch = obs.unsqueeze(0)
                
                # b. 模型预测动作（4维，归一化到[-1,1]）
                model_output = actor(obs_batch)
                action_tensor = extract_action_from_model_output(model_output, final_device)
                # print(f" Step{ep_length :4d} | Action (normalized): {action_tensor.squeeze(0).cpu().numpy().round(3)}")
                # 移除batch维度 + 转换为numpy数组（环境接受numpy）
                action_np = action_tensor.squeeze(0).cpu().numpy()
                action_np[0] = np.clip(
                    action_np[0], 
                    -0.2, 0.2  # 限制u2动作范围
                )
                action_np[1] = np.clip(
                    action_np[1], 
                    -0.2, 0.2  # 限制u2动作范围
                )
                # action_np = np.array([-1, -1, 1, 0])
                # print(f" Predicted action (normalized): {action_np.round(3)}")
                # c. 环境step（标准Gym接口，环境自动反归一化动作）
                obs, reward, terminated, truncated, info = sim_env.step(action_np)

                # print(f" Step{ep_length :4d} | u1_original: {info['u1'].round(3)}")
                # print(f" Step{ep_length :4d} | u2_original: {info['u2_original'].round(3)}")
                # print(f" Step{ep_length :4d} | u2_normalized: {info['u2_normalized'].round(3)}")
                # print(f" Step{ep_length :4d} | x: {info['x'].round(3)}")

                # d. 转换新obs为Tensor（下一轮模型输入）
                obs = torch.tensor(obs, dtype=torch.float32, device=final_device)
                
                # e. 累计数据
                ep_reward += reward
                ep_length += 1
                ep_hinge_forces.append(info['Fh2'])  # 记录铰接力
                
                # f. 打印进度（每100步）
                if ep_length % 100 == 0:
                    fh2_mag = np.hypot(info['Fh2'][0], info['Fh2'][1])  # 铰接力大小
                    pos_error = info.get('pos_error', 0.0)
                    print(f"  Step {ep_length} | 当前奖励: {ep_reward:.2f} | 铰接力大小: {fh2_mag:.2f} | 位置误差: {pos_error:.2f}")
            
            # 5. 记录本轮结果
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            episode_hinge_forces.append(ep_hinge_forces)
            
            # 计算本轮铰接力统计
            ep_hinge_array = np.array(ep_hinge_forces)
            ep_hinge_mag = np.hypot(ep_hinge_array[:, 0], ep_hinge_array[:, 1])
            avg_hinge_force = np.mean(ep_hinge_mag)
            max_hinge_force = np.max(ep_hinge_mag)
            
            print(f"  Episode {episode_idx+1:2d} 统计:")
            print(f"    总奖励: {ep_reward:.2f} | 总步数: {ep_length}")
            print(f"    平均铰接力: {avg_hinge_force:.2f} | 最大铰接力: {max_hinge_force:.2f}")
            
            # 6. 保存本轮视频（完全匹配环境的save_eval_video方法）
            if args.save_video and args.enable_visualization and args.render_mode == "rgb_array":
                try:
                    video_path = sim_env.save_eval_video(
                        eval_round=f"offline_ep{episode_idx+1}",
                        video_save_dir=video_save_dir
                    )
                    if video_path:
                        print(f"    ✅ 视频已保存: {video_path}")
                except Exception as e:
                    print(f"    ⚠️ 视频保存失败: {e}")
            
            # 标记仿真结束（避免reset时清空帧）
            sim_env.mark_sim_finished()
    
    # 5. 整体统计结果
    print(f"\n📊 仿真结果汇总（{args.num_episodes}轮）:")
    print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  平均步数: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    # 铰接力统计
    all_hinge_forces = []
    for ep_hf in episode_hinge_forces:
        ep_hf_array = np.array(ep_hf)
        all_hinge_forces.extend(np.hypot(ep_hf_array[:, 0], ep_hf_array[:, 1]))
    if all_hinge_forces:
        print(f"  平均铰接力: {np.mean(all_hinge_forces):.2f} ± {np.std(all_hinge_forces):.2f}")
        print(f"  最大铰接力: {np.max(all_hinge_forces):.2f}")
    
    print(f"  最大奖励轮次: 第{np.argmax(episode_rewards)+1}轮 ({np.max(episode_rewards):.2f})")
    print(f"  最小奖励轮次: 第{np.argmin(episode_rewards)+1}轮 ({np.min(episode_rewards):.2f})")
    
    # 6. 关闭环境（触发视频生成）
    sim_env.close()
    
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_hinge_forces": episode_hinge_forces,
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_hinge_force": np.mean(all_hinge_forces) if all_hinge_forces else 0.0
    }

# ===================== 主函数 =====================
def main():
    args = parse_args()
    
    CKPT_PATH="E:\rl\occt\outputs\2026-01-06\19-06-31\checkpoints_occt\checkpoint_1024000_frames.pt"
    DEVICE="cpu"
    # 加载Checkpoint
    actor, cfg, vecnorm_state = load_checkpoint(args.ckpt_path, args.device)
    
    # 运行仿真
    sim_results = run_offline_simulation(actor, cfg, vecnorm_state, args)
    
    print(f"\n🎉 离线仿真完成!")
    print(f"  结果已保存，平均奖励: {sim_results['avg_reward']:.2f}")

if __name__ == "__main__":
    main()