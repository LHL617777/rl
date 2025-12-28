import gymnasium as gym
from gymnasium.utils.env_checker import check_env
# 导入你的自定义环境类
from occt_2d2c import TwoCarrierEnv
import numpy as np

# 1. 修正环境校验代码（消除前两个警告）
print("=" * 60)
print("开始环境合规性校验（消除包装器与废弃参数警告）...")

# 方式1：直接创建原始环境实例（无包装器，推荐用于校验）
raw_env = TwoCarrierEnv(render_mode=None, enable_visualization=False)

# 方式2：若通过gym.make创建，先解包再校验
# env = gym.make("TwoCarrierEnv-v0", enable_visualization=False)
# raw_env = env.unwrapped  # 解包获取原始环境

# 执行校验（删除warn参数，使用原始环境）
try:
    check_env(raw_env, skip_render_check=True)  # 移除warn=True，使用原始环境
    print("✅ 环境通过Gymnasium所有合规性校验！无功能性错误")
except Exception as e:
    print(f"❌ 环境存在合规性问题：{e}")
finally:
    raw_env.close()
print("=" * 60)

# 1. 查看已注册环境（确认你的环境在注册表中）
print("=" * 60)
print("已注册的Gymnasium环境（包含你的自定义环境）：")
# 筛选包含你的环境ID的条目
env_id = "TwoCarrierEnv-v0"
registered_envs = [id for id in gym.registry.keys() if env_id in id]
for id in registered_envs:
    print(f"  - {id}")
print("=" * 60)


# 2. 验证向量环境创建（用于并行训练，指南提及`make_vec`）
try:
    vec_env = gym.make_vec(env_id, num_envs=2, enable_visualization=False)
    print(f"✅ 向量环境创建成功，并行环境数量：{vec_env.num_envs}")
    vec_env.reset(seed=42)
    vec_env.close()
except Exception as e:
    print(f"向量环境创建提示：{e}（若报错，需确保环境支持向量化，不影响单机训练）")
print("=" * 60)