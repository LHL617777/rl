# 模型训练和评估所需的工具函数
from __future__ import annotations

import torch.nn
import torch.optim

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNormV2,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(
    env_name="TwoCarrierEnv-v0",
    device="cpu",
    from_pixels: bool = False,
    render_mode=None,  # 自定义参数：渲染模式
    enable_visualization: bool = False  # 自定义参数：可视化开关
):
    """
    创建环境（传env_name字符串，直接透传自定义参数，解决未知关键字参数错误）
    """
    # 第一步：针对自定义环境，构造需要透传的参数（直接作为GymEnv的关键字参数）
    gymnasium_kwargs = {}
    if env_name == "TwoCarrierEnv-v0":
        # 直接构造自定义环境的__init__参数，不封装gym_kwargs
        gymnasium_kwargs = {
            "render_mode": render_mode,
            "enable_visualization": enable_visualization
        }
    
    # 第二步：传env_name字符串，直接透传自定义参数给GymEnv
    # 关键：将gymnasium_kwargs解包，作为GymEnv的顶层关键字参数传递
    # 此时GymEnv会自动将这些参数透传给gymnasium.make，进而传递给TwoCarrierEnv
    base_env = GymEnv(
        env_name=env_name,  # 保留传env_name字符串的习惯
        from_pixels=from_pixels,
        pixels_only=False,
        **gymnasium_kwargs  # 解包传递自定义参数，无gym_kwargs顶层参数
    )
    
    # 第三步：保留原有TorchRL环境变换逻辑，确保训练/评测流程不变
    env = TransformedEnv(base_env)
    env.append_transform(VecNormV2(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_models_state(proof_environment, device):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec_unbatched.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "low": proof_environment.action_spec_unbatched.space.low.to(device),
        "high": proof_environment.action_spec_unbatched.space.high.to(device),
        "tanh_loc": False,
    }

    # Define policy architecture
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

    # Define value architecture
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
    :param actor: 待评估的 PPO Actor 模型
    :param test_env: TorchRL 封装后的测试环境
    :param num_episodes: 每轮评测的 episode 数量
    :param eval_round: 外层评测轮次（来自 PPO_occt_check.py 的 eval_round_counter）
    :return: 所有 episode 奖励的均值
    """
    test_rewards = []
    
    # 提前解除环境包装，获取原始 TwoCarrierEnv 实例（仅执行一次，提升效率）
    try:
        raw_test_env = test_env.unwrapped
        while not hasattr(raw_test_env, "render_frames") and not isinstance(raw_test_env, type(None)):
            raw_test_env = getattr(raw_test_env, "_env", None)
        if raw_test_env is not None:
            print("成功获取原始 TwoCarrierEnv 实例，准备捕获帧并生成视频")
        else:
            print("未获取到原始 TwoCarrierEnv 实例，无法生成本地视频")
    except Exception as e:
        raw_test_env = None
        print(f"获取原始环境实例失败，无法生成本地视频：{type(e).__name__}: {e}")
    
    for episode_idx in range(num_episodes):
        # 步骤1：执行 TorchRL rollout（启用 auto_reset=True，无需手动传 tensordict）
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True, 
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000
            # 删除 tensordict 参数，无需手动传入初始环境状态
        )
        
        # 步骤2：提取并缓存本轮 episode 奖励（保持原有逻辑，不变）
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
        
        # 步骤3：自定义视频生成（每 episode 结束后保存独立视频，保持原有逻辑）
        if raw_test_env is not None and raw_test_env.enable_visualization and raw_test_env.render_mode == "rgb_array":
            # 构造视频命名标识（评测轮次+episode 索引，避免重名）
            video_identifier = f"{eval_round or 'unknown'}_episode_{episode_idx+1}"
            # 保存该 episode 的独立视频（基于当前缓存的 render_frames）
            video_filepath = raw_test_env.save_eval_video(
                eval_round=video_identifier,
                video_save_dir=getattr(raw_test_env, "checkpoint_dir", None)
            )
            # 关键适配：视频保存后，手动清空帧列表（实现帧隔离，避免叠加到下一个 episode）
            raw_test_env.clear_render_frames()
        
        # 步骤4：释放 td_test 内存，避免内存泄漏（不变）
        del td_test
    
    # 步骤5：计算并返回所有 episode 奖励的均值（保持原有逻辑，不变）
    if len(test_rewards) > 0:
        return torch.cat(test_rewards, 0).mean()
    else:
        return torch.tensor(0.0, dtype=torch.float32)
