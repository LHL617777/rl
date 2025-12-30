# -------------------------- 【完整导入】适配原生Gymnasium环境 --------------------------
import os
import torch
import numpy as np
import gymnasium as gym
from torchrl.envs import ExplorationType, set_exploration_type
from utils_ppo_occt import make_ppo_models
from occt_2d2c import TwoCarrierEnv

# -------------------------- 【配置不变】 --------------------------
ENV_NAME = "TwoCarrierEnv-v0"
CHECKPOINT_DIR = "./checkpoints_occt/"
MAX_SIM_STEPS = 1000
RENDER_MODE = "human"
ENABLE_VISUALIZATION = True
SEED = 42

# -------------------------- 【辅助函数不变】 --------------------------
def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print(f"错误：Checkpoint 目录 {checkpoint_dir} 不存在，请先完成训练")
        return None
    
    ckpt_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_") and filename.endswith("_frames.pt"):
            try:
                frame_count = int(filename.split("_")[1])
                ckpt_files.append((frame_count, filename))
            except (IndexError, ValueError):
                continue
    
    if not ckpt_files:
        print(f"错误：{checkpoint_dir} 下无有效 Checkpoint 文件")
        return None
    
    ckpt_files.sort(reverse=True, key=lambda x: x[0])
    latest_frame_count, latest_filename = ckpt_files[0]
    latest_ckpt_path = os.path.join(checkpoint_dir, latest_filename)
    
    print(f"成功找到最新 Checkpoint：")
    print(f"  文件名：{latest_filename}")
    print(f"  训练帧数：{latest_frame_count}")
    print(f"  文件路径：{latest_ckpt_path}")
    
    return latest_ckpt_path

# -------------------------- 【核心修正】仿真流程 --------------------------
def run_occt_simulation():
    # 1. 设备初始化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 设备初始化完成：{device} ===")

    # 2. 加载最新 Checkpoint
    latest_ckpt_path = './outputs/checkpoints_occt/checkpoint_225280_frames.pt'
    if latest_ckpt_path is None:
        return
    ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
    print(f"\n=== 成功加载 Checkpoint 配置 ===")

    # 3. 构建模型并加载参数
    print(f"\n=== 构建 PPO 模型（与训练结构一致） ===")
    actor, _ = make_ppo_models(ENV_NAME, device=device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    print("=== 模型参数加载完成，已进入评估模式 ===")

    # 4. 创建原生Gymnasium OCCT环境
    print(f"\n=== 创建原生Gymnasium OCCT环境（{RENDER_MODE} 模式） ===")
    env = gym.make(
        ENV_NAME,
        render_mode=RENDER_MODE,
        enable_visualization=ENABLE_VISUALIZATION
    )

    # 5. 重置原生环境（正常解包）
    print(f"\n=== 重置环境，开始仿真 ===")
    obs, info = env.reset(seed=SEED)
    total_reward = 0.0
    step_count = 0
    done = False

    # 6. 仿真循环（修正动作提取逻辑）
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        while not done and step_count < MAX_SIM_STEPS:
            # 原生观测 → 模型张量输入（增加批次维度）
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # 模型预测动作（修正：兼容元组/张量返回值，无字符串索引）
            actor_output = actor(obs_tensor)
            if isinstance(actor_output, tuple):
                action_tensor = actor_output[0]  # 提取元组中的动作张量
            else:
                action_tensor = actor_output  # 直接接收动作张量
            
            # 模型张量 → 原生numpy动作（删除批次维度）
            action = action_tensor.cpu().numpy().squeeze()

            # 原生环境step()交互
            obs, reward, terminated, truncated, info = env.step(action)

            # 更新仿真状态
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # 打印核心指标
            if step_count % 50 == 0 or done:
                Fh2_x, Fh2_y = info["Fh2"]
                pos_error = info["pos_error"]
                print(f"【第 {step_count:04d} 步】")
                print(f"  累计奖励：{total_reward:.6f} | 铰接力(Fh2)：({Fh2_x:.2f}, {Fh2_y:.2f})")
                print(f"  双车位置误差：{pos_error:.4f}m | 终止状态：{terminated}")
                print(f"  " + "-" * 50)

    # 7. 仿真结束总结
    print(f"\n=== 仿真结束 ===")
    print(f"  总步数：{step_count}")
    print(f"  累计奖励：{total_reward:.6f}")
    print(f"  仿真终止原因：{'模型完成仿真' if terminated else '达到最大步数'}")

    # 8. 关闭环境并生成视频（若开启）
    if RENDER_MODE == "rgb_array" and ENABLE_VISUALIZATION:
        print(f"\n=== 正在生成仿真视频 ===")
    env.close()
    if RENDER_MODE == "rgb_array" and ENABLE_VISUALIZATION:
        print(f"  视频生成完成（可在 Checkpoint 或 ./output 目录查找）")

# -------------------------- 【主入口不变】 --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("        OCCT 双车运载超大件系统 PPO 模型仿真（原生Gymnasium）")
    print("=" * 60)
    run_occt_simulation()
    print("\n=== 仿真脚本执行完毕 ===")