# 模型训练和评估所需的工具函数
from __future__ import annotations

import torch.nn
import torch.optim
import os  # 补充必要导入

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder
from torchrl.envs.gym_like import default_info_dict_reader

# 导入自定义环境（确保能识别TwoCarrierEnv）
from occt_2d2c import TwoCarrierEnv

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(
    env_name="TwoCarrierEnv-v1",
    device="cpu",
    from_pixels: bool = False,
    render_mode=None,  # 自定义参数：渲染模式
    enable_visualization: bool = False,  # 自定义参数：可视化开关
    vecnorm_frozen: bool = False,  # 新增：VecNorm冻结标记，训练=False（默认），评测=True
    vecnorm_mean=None,
    vecnorm_var=None,
    shared_w_force=None
):
    """
    创建环境（传env_name字符串，直接透传自定义参数，解决未知关键字参数错误）
    适配：移除TorchRL VecNormV2，使用环境内部自定义VecNorm，支持冻结统计量
    """
    # 第一步：针对自定义环境，构造需要透传的参数（直接作为GymEnv的关键字参数）
    gymnasium_kwargs = {}
    if env_name == "TwoCarrierEnv-v1":
        # 直接构造自定义环境的__init__参数，新增vecnorm_frozen透传
        gymnasium_kwargs = {
            "render_mode": render_mode,
            "enable_visualization": enable_visualization,
            # 新增：将VecNorm冻结标记传递给TwoCarrierEnv
            "vecnorm_frozen": vecnorm_frozen,
            "vecnorm_mean": vecnorm_mean,
            "vecnorm_var": vecnorm_var,
            "shared_w_force": shared_w_force
        }
    
    # 第二步：传env_name字符串，直接透传自定义参数给GymEnv
    # 关键：将gymnasium_kwargs解包，作为GymEnv的顶层关键字参数传递
    # 此时GymEnv会自动将这些参数透传给gymnasium.make，进而传递给TwoCarrierEnv
    base_env = GymEnv(
        env_name=env_name,  # 保留传env_name字符串的习惯
        from_pixels=from_pixels,
        pixels_only=False,
        device=device,
        disable_env_checker=True,  # 新增：关闭被动环境检查器，避免观测空间警告
        **gymnasium_kwargs  # 解包传递自定义参数（含vecnorm_frozen）
    )

    # =============== 【核心修复 2/2】 =================
    # set_info_dict_reader 需要一个函数，而不是列表。
    # 使用 default_info_dict_reader(["key"]) 来生成这个函数。
    # 这会告诉 TorchRL："请生成一个函数，这个函数专门去 info 里抓取 'reward_details' 字段"
    reward_keys = [
        "reward_r_force", 
        "reward_r_align_rear", 
        "reward_r_align_front", 
        # "reward_r_smooth", 
        "reward_r_progress", 
        "reward_r_stability",
        "reward_val_force",
        "reward_val_delta_psi_rear",
        "reward_val_delta_psi_front"
    ]
    base_env.set_info_dict_reader(
        info_dict_reader=default_info_dict_reader(reward_keys)
    )
    # ==================================================

    # 第三步：调整TorchRL环境变换逻辑，移除双重归一化，保留必要变换
    env = TransformedEnv(base_env)
    
    # 调整：ClipTransform范围，适配环境内归一化观测的合理范围（与环境observation_space一致）
    env.append_transform(ClipTransform(
        in_keys=["observation"],
        low=-1000.0,  # 与环境内obs_norm_low一致
        high=1000.0   # 与环境内obs_norm_high一致
    ))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    
    env = env.to(device)
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_models_state(proof_environment, device):
    """构建PPO Actor/Critic模型，适配12维归一化观测"""
    # Define input shape（自动适配环境返回的12维归一化观测，无需修改）
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec_unbatched.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low.to(device),
        "high": proof_environment.action_spec_unbatched.space.high.to(device),
        "tanh_loc": False,
    }

    # Define policy architecture（输入维度自动适配12维，无需修改）
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=[64, 64],
        device=device,
    )

    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(
            proof_environment.action_spec_unbatched.shape[-1], scale_lb=1e-8
        ).to(device),
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=proof_environment.full_action_spec_unbatched.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value architecture（输入维度自动适配12维，无需修改）
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
        device=device,
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module


def make_ppo_models(env_name, device):
    """创建PPO模型，适配归一化观测环境"""
    # 构建验证环境（默认vecnorm_frozen=False，不影响模型输入维度）
    proof_environment = make_env(env_name, device=device)
    actor, critic = make_ppo_models_state(proof_environment, device=device)
    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def eval_model(actor, test_env, num_episodes=3, eval_round=None):
    """
    评估模型性能（启用 auto_reset=True，简洁高效，保留独立视频生成）
    适配：确保评测环境VecNorm已冻结，使用环境内部归一化观测
    :param actor: 待评估的 PPO Actor 模型
    :param test_env: TorchRL 封装后的测试环境（已冻结VecNorm）
    :param num_episodes: 每轮评测的 episode 数量
    :param eval_round: 外层评测轮次（来自 PPO_occt_check.py 的 eval_round_counter）
    :return: 所有 episode 奖励的均值
    """
    test_rewards = []
    
    # 提前解除环境包装，获取原始 TwoCarrierEnv 实例（验证VecNorm状态+视频生成）
    try:
        raw_test_env = test_env.unwrapped
        # 优化：精准获取TwoCarrierEnv实例，验证VecNorm冻结状态
        while not isinstance(raw_test_env, TwoCarrierEnv) and raw_test_env is not None:
            raw_test_env = getattr(raw_test_env, "_env", raw_test_env.unwrapped)
        
        if raw_test_env is not None:
            # 验证：打印评测环境VecNorm状态，确保已冻结
            print(f"✅ 成功获取原始 TwoCarrierEnv 实例，VecNorm冻结状态：{raw_test_env.vecnorm_frozen}")
            if not raw_test_env.vecnorm_frozen:
                print("⚠️ 评测环境VecNorm未冻结，强制设置为True以保证一致性")
                raw_test_env.vecnorm_frozen = True
        else:
            print("❌ 未获取到原始 TwoCarrierEnv 实例，无法生成本地视频")
    except Exception as e:
        raw_test_env = None
        print(f"❌ 获取原始环境实例失败，无法生成本地视频：{type(e).__name__}: {e}")
    
    for episode_idx in range(num_episodes):
        # 步骤1：执行 TorchRL rollout（启用 auto_reset=True，无需手动传 tensordict）
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True, 
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000
        )
        
        # 步骤2：提取并缓存本轮 episode 奖励（保持原有逻辑，不变）
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
        
        # # 步骤3：自定义视频生成（每 episode 结束后保存独立视频，保持原有逻辑）
        # if raw_test_env is not None and raw_test_env.enable_visualization and raw_test_env.render_mode == "rgb_array":
        #     # 构造视频命名标识（评测轮次+episode 索引，避免重名）
        #     video_identifier = f"{eval_round or 'unknown'}_episode_{episode_idx+1}"
        #     # 保存该 episode 的独立视频（基于当前缓存的 render_frames）
        #     video_filepath = raw_test_env.save_eval_video(
        #         eval_round=video_identifier,
        #         video_save_dir=getattr(raw_test_env, "checkpoint_dir", None)
        #     )
        #     # 关键适配：视频保存后，手动清空帧列表（实现帧隔离，避免叠加到下一个 episode）
        #     raw_test_env.clear_render_frames()
        
        # 步骤4：释放 td_test 内存，避免内存泄漏（不变）
        del td_test
    
    # 步骤5：计算并返回所有 episode 奖励的均值（保持原有逻辑，不变）
    if len(test_rewards) > 0:
        return torch.cat(test_rewards, 0).mean()
    else:
        return torch.tensor(0.0, dtype=torch.float32)