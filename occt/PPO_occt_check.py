''' 使用主流 RL 库Stable Baselines3（常用 RL 工具）
    用 PPO 算法（支持连续动作空间）进行简单训练
    验证环境能否用于 RL 训练。'''

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from occt_2d2c import TwoCarrierEnv
from stable_baselines3.common.monitor import Monitor

# 1. 创建向量环境（PPO算法推荐使用向量环境加速训练）
env_id = "TwoCarrierEnv-v0"
vec_env = make_vec_env(
    env_id,
    n_envs=2,  # 2个并行环境
    env_kwargs={"enable_visualization": False}  # 关闭可视化，加速训练
)

# 2. 初始化PPO模型（适配连续动作空间，你的环境动作空间是Box，PPO天然支持）
model = PPO(
    "MlpPolicy",  # 多层感知器策略，适用于连续状态+连续动作
    vec_env,
    verbose=1,  # 打印训练日志
    seed=42,
    learning_rate=3e-4,
    n_steps=2048,
)

# 3. 进行短时间训练（验证能否正常训练，无需训练到收敛）
print("=" * 60)
print("开始简单训练（验证环境与RL算法兼容性）...")
model.learn(total_timesteps=10000)  # 训练10000步，快速验证
print("✅ 训练完成，环境可正常与RL算法交互！")
print("=" * 60)

# 4. 保存模型（验证模型能否正常保存，基于环境状态）
model.save("ppo_two_carrier")
print("✅ 模型已保存为：ppo_two_carrier.zip")

# 5. 加载模型并评估性能（验证训练成果，说明环境可用于效果评估）
loaded_model = PPO.load("ppo_two_carrier")
eval_env = gym.make(env_id, enable_visualization=False)
eval_env = Monitor(eval_env)  # 使用Monitor记录评估数据
mean_reward, std_reward = evaluate_policy(
    loaded_model,
    eval_env,
    n_eval_episodes=5,  # 评估5个回合
    deterministic=True  # 确定性策略评估
)

print("=" * 60)
print(f"模型评估结果：")
print(f"  平均奖励：{mean_reward:.4f}")
print(f"  奖励标准差：{std_reward:.4f}")
print("✅ RL算法兼容性测试完成，环境可用于强化学习训练与评估！")
print("=" * 60)

# 关闭所有环境
vec_env.close()
eval_env.close()