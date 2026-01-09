# 模型训练脚本：PPO算法在OCCT环境中的实现
"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm.
适配：环境内部自定义VecNorm观测归一化，移除双重归一化，增强Checkpoint兼容性
"""
from __future__ import annotations
import sys
import os
import numpy as np

# 添加本地自定义torchrl的根目录（E:\rl\torchrl\的上一级目录，即E:\rl\）
sys.path.insert(0, "E:\\rl")  # insert(0)表示将该路径放在搜索优先级第1位
# sys.path.insert(0, "/home/yons/Graduation/rl")  # Linux系统中的路径
# TIspdizNBNWYgfoUxNl86    swanlab API Key

import warnings

import hydra
import torchrl
from torchrl._utils import compile_with_warmup
from torchrl.record.loggers.swanlab import SwanLabLogger
import gymnasium as gym
from occt_2d2c import TwoCarrierEnv
from omegaconf import DictConfig
from torchrl.collectors import Collector


@hydra.main(config_path="", config_name="config_occt", version_base="1.1")
def main(cfg: DictConfig):

    import torch.optim
    import tqdm

    from tensordict import TensorDict
    from tensordict.nn import CudaGraphModule

    from torchrl._utils import timeit
    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import ClipPPOLoss, group_optimizers
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record import VideoRecorder
    from torchrl.record.loggers import generate_exp_name, get_logger
    from utils_ppo_occt import eval_model, make_env, make_ppo_models

    torch.set_float32_matmul_precision("high")

    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        (cfg.collector.total_frames // cfg.collector.frames_per_batch)
        * cfg.loss.ppo_epochs
        * num_mini_batches
    )

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create models (适配12维归一化观测，无需修改，由make_ppo_models自动适配)
    actor, critic = make_ppo_models(cfg.env.env_name, device=device)

    # Create collector：修改点1 - 封装训练环境，设置vecnorm_frozen=False（不冻结，自动更新统计量）
    collector = Collector(
        create_env_fn=lambda: make_env(  # 用lambda封装，传递VecNorm状态参数
            cfg.env.env_name,
            device,
            vecnorm_frozen=False  # 训练环境：不冻结VecNorm，统计量随训练更新
        ),
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        max_frames_per_traj=-1,
        compile_policy={"mode": compile_mode, "warmup": 1} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            cfg.collector.frames_per_batch,
            compilable=cfg.compile.compile,
            device=device,
        ),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,
        compilable=cfg.compile.compile,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
        device=device,
        vectorized=not cfg.compile.compile,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coeff=cfg.loss.entropy_coeff,
        critic_coeff=cfg.loss.critic_coeff,
        normalize_advantage=True,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=torch.tensor(cfg.optim.lr, device=device), eps=1e-5
    )
    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=torch.tensor(cfg.optim.lr, device=device), eps=1e-5
    )
    optim = group_optimizers(actor_optim, critic_optim)
    del actor_optim, critic_optim

    # Create logger：无需修改，仅新增VecNorm相关日志（可选）
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo",
            experiment_name=exp_name,
            swanlab_kwargs={
                "config": cfg,
                "group": cfg.logger.group_name,
                "project": cfg.logger.project_name
                or f"swanlab_{cfg.env.scenario_name}",
            },
        )
        logger_video = False
    else:
        logger_video = False

    # Create test environment：修改点2 - 评测环境设置vecnorm_frozen=True（冻结统计量，保证一致性）
    test_env = make_env(
        cfg.env.env_name,
        device,
        from_pixels=logger_video,
        render_mode=None,  # 自定义rgb_array渲染模式
        enable_visualization=False,  # 自定义可视化功能
        vecnorm_frozen=True  # 评测环境：冻结VecNorm，不更新统计量，保证评测结果稳定
    )
    test_env.eval()

    # Update函数：无需修改，适配归一化观测输入
    def update(batch, num_network_updates):
        optim.zero_grad(set_to_none=True)
        # Linearly decrease the learning rate and clip epsilon
        alpha = torch.ones((), device=device)
        if cfg.optim.anneal_lr:
            alpha = 1 - (num_network_updates / total_network_updates)
            for group in optim.param_groups:
                group["lr"] = cfg.optim.lr * alpha
        if cfg.loss.anneal_clip_epsilon:
            loss_module.clip_epsilon.copy_(cfg.loss.clip_epsilon * alpha)
        num_network_updates = num_network_updates + 1

        # Forward pass PPO loss
        loss = loss_module(batch)
        critic_loss = loss["loss_critic"]
        actor_loss = loss["loss_objective"] + loss["loss_entropy"]
        total_loss = critic_loss + actor_loss

        # Backward pass
        total_loss.backward()

        # Update the networks
        optim.step()
        return loss.detach().set("alpha", alpha), num_network_updates

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)
        adv_module = compile_with_warmup(adv_module, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)
        adv_module = CudaGraphModule(adv_module)

    # Main loop
    collected_frames = 0
    num_network_updates = torch.zeros((), dtype=torch.int64, device=device)
    pbar = tqdm.tqdm(
        total=cfg.collector.total_frames,
        desc="Training",
        leave=True,
        dynamic_ncols=True
    )

    eval_round_counter = 0  # 初始化评测轮次，每次触发评测自增

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = torch.tensor(cfg.optim.lr, device=device)
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_logger_test_interval = cfg.logger.test_interval
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes
    losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    # ===================== 修改点3：增强Checkpoint，保存VecNorm统计量 =====================
    def save_checkpoint(current_frames):
        """封装Checkpoint保存逻辑，适配cfg配置，新增VecNorm统计量保存"""
        # 关键：提取训练环境的原始TwoCarrierEnv实例，获取VecNorm统计量
        raw_train_env = None
        vecnorm_mean = np.zeros(12, dtype=np.float64)
        vecnorm_var = np.ones(12, dtype=np.float64) * 1e-4  # 与环境默认最小方差一致
        vecnorm_frozen = False

        try:
            # 解包torchrl Collector的环境实例，获取原始TwoCarrierEnv
            train_env_instance = collector.env
            raw_train_env = train_env_instance.unwrapped
            while not isinstance(raw_train_env, TwoCarrierEnv) and raw_train_env is not None:
                raw_train_env = getattr(raw_train_env, "_env", raw_train_env.unwrapped)
            
            if raw_train_env is not None:
                # 提取VecNorm统计量
                vecnorm_mean = raw_train_env.vecnorm_mean.copy()
                vecnorm_var = raw_train_env.vecnorm_var.copy()
                vecnorm_frozen = raw_train_env.vecnorm_frozen
        except Exception as e:
            print(f"⚠️ 获取训练环境VecNorm统计量失败（不影响模型保存）：{e}")

        # 构造Checkpoint字典，新增VecNorm相关内容
        ckpt_dict = {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "cfg": cfg,
            "collected_frames": current_frames,
            # 新增：VecNorm统计量，用于后续加载时恢复归一化分布
            "vecnorm_mean": vecnorm_mean,
            "vecnorm_var": vecnorm_var,
            "vecnorm_frozen": vecnorm_frozen,
        }
        save_dir = cfg.checkpoint.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"checkpoint_{current_frames}_frames.pt")
        torch.save(ckpt_dict, save_path)
        print(f"\n✅ Checkpoint saved to: {save_path}")
        # 打印VecNorm统计量，监控训练收敛情况
        if raw_train_env is not None:
            print(f"✅ 附带VecNorm均值前12维：{vecnorm_mean[:12].round(6)}")
            print(f"✅ 附带VecNorm方差前12维：{vecnorm_var[:12].round(6)}")

    # ===================== 新增：Checkpoint加载函数（可选，用于后续继续训练/评测） =====================
    def load_checkpoint(ckpt_path, target_env=None):
        """加载Checkpoint，恢复模型参数与VecNorm统计量"""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint文件不存在：{ckpt_path}")
        
        # 加载Checkpoint
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        
        # 恢复模型与优化器参数
        actor.load_state_dict(ckpt_dict["actor_state_dict"])
        critic.load_state_dict(ckpt_dict["critic_state_dict"])
        optim.load_state_dict(ckpt_dict["optim_state_dict"])
        
        # 恢复VecNorm统计量（若传入目标环境）
        if target_env is not None:
            try:
                raw_env = target_env.unwrapped
                while not isinstance(raw_env, TwoCarrierEnv) and raw_env is not None:
                    raw_env = getattr(raw_env, "_env", raw_env.unwrapped)
                
                if raw_env is not None:
                    raw_env.vecnorm_mean = np.asarray(ckpt_dict["vecnorm_mean"], dtype=np.float64)
                    raw_env.vecnorm_var = np.asarray(ckpt_dict["vecnorm_var"], dtype=np.float64)
                    raw_env.vecnorm_frozen = ckpt_dict["vecnorm_frozen"]
                    print(f"✅ 已从Checkpoint恢复VecNorm统计量，冻结状态：{raw_env.vecnorm_frozen}")
            except Exception as e:
                print(f"⚠️ 恢复VecNorm统计量失败：{e}")
        
        # 返回配置与已训练帧数
        return ckpt_dict["cfg"], ckpt_dict["collected_frames"]

    # 示例：加载预训练Checkpoint继续训练（按需启用，需在cfg中配置load_ckpt_path）
    # if hasattr(cfg.checkpoint, "load_ckpt_path") and cfg.checkpoint.load_ckpt_path:
    #     cfg, start_frames = load_checkpoint(cfg.checkpoint.load_ckpt_path, collector.env)
    #     collected_frames = start_frames
    #     last_saved_frames = start_frames
    #     print(f"✅ 已加载预训练Checkpoint，从 {start_frames} 帧继续训练")

    # 提取保存配置（从cfg读取，避免硬编码）
    save_interval = cfg.checkpoint.save_interval
    last_saved_frames = 0  # 记录上一次保存的帧数，避免重复保存

    collector_iter = iter(collector)
    total_iter = len(collector)
    for i in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collecting"):
            data = next(collector_iter)

        metrics_to_log = {}
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # ===== 触发Checkpoint保存（逻辑不变，增强后自动携带VecNorm统计量） =====
        if save_interval > 0:
            if collected_frames // save_interval > last_saved_frames // save_interval:
                save_checkpoint(collected_frames)
                last_saved_frames = collected_frames

        # Get training rewards and episode lengths：无需修改
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            metrics_to_log.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        # PPO_occt_check.py 的 main 函数循环中

        # ================= 修改点5：提取并统计 Reward 分项 =================
        try:
            # 定义我们要记录的 key (与环境和 utils 中一致)
            reward_keys = [
                "reward_r_force", 
                "reward_r_align", 
                "reward_r_smooth", 
                "reward_r_progress", 
                "reward_val_force"
            ]
            
            # TorchRL 通常会将 info 中的 scalar 数据提升到 "next" 的一级 key
            # 或者在 ("next", "info", key)
            
            # 优先检查 data["next"] 下是否有这些 key
            next_td = data["next"]
            
            for key in reward_keys:
                # 检查 key 是否存在于 TensorDict
                if key in next_td.keys():
                    val_mean = next_td[key].float().mean().item()
                    metrics_to_log[f"reward_parts/{key}"] = val_mean
                # 备选：有时候会在 info 嵌套下
                elif ("info", key) in next_td.keys(include_nested=True):
                    val_mean = next_td["info", key].float().mean().item()
                    metrics_to_log[f"reward_parts/{key}"] = val_mean

        except Exception as e:
            if i == 0: print(f"⚠️ 提取 Reward Details 失败: {e}")
        # ===================================================================

        with timeit("training"):
            for j in range(cfg_loss_ppo_epochs):

                # Compute GAE
                with torch.no_grad(), timeit("adv"):
                    torch.compiler.cudagraph_mark_step_begin()
                    data = adv_module(data)
                    if compile_mode:
                        data = data.clone()

                with timeit("rb - extend"):
                    # Update the data buffer
                    data_reshape = data.reshape(-1)
                    data_buffer.extend(data_reshape)

                for k, batch in enumerate(data_buffer):
                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss, num_network_updates = update(
                            batch, num_network_updates=num_network_updates
                        )
                        loss = loss.clone()
                    num_network_updates = num_network_updates.clone()
                    losses[j, k] = loss.select(
                        "loss_critic", "loss_entropy", "loss_objective"
                    )

        # Get training losses and times：无需修改
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            metrics_to_log.update({f"train/{key}": value.item()})
        metrics_to_log.update(
            {
                "train/lr": loss["alpha"] * cfg_optim_lr,
                "train/clip_epsilon": loss["alpha"] * cfg_loss_clip_epsilon
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
            }
        )

        # Get test rewards：修改点4 - 验证评测环境VecNorm状态，简化视频逻辑
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
                i * frames_in_batch
            ) // cfg_logger_test_interval:
                eval_round_counter += 1  # 自增评测轮次
                actor.eval()
                print(f"\n============= 开始第 {eval_round_counter} 轮评测 =============")

                # ===== 新增：步骤1 - 提取训练环境的最新VecNorm统计量 =====
                train_vecnorm_mean = np.zeros(12, dtype=np.float64)
                train_vecnorm_var = np.ones(12, dtype=np.float64) * 1e-4
                try:
                    # 解包训练环境，获取最新的mean/var（复用你现有Checkpoint中的提取逻辑）
                    raw_train_env = collector.env.unwrapped
                    while not isinstance(raw_train_env, TwoCarrierEnv) and raw_train_env is not None:
                        raw_train_env = getattr(raw_train_env, "_env", raw_train_env.unwrapped)
                    
                    if raw_train_env is not None:
                        train_vecnorm_mean = raw_train_env.vecnorm_mean.copy()
                        train_vecnorm_var = raw_train_env.vecnorm_var.copy()
                        print(f"✅ 提取到训练环境最新VecNorm：均值前5维 {train_vecnorm_mean[:5].round(6)}，方差前5维 {train_vecnorm_var[:5].round(6)}")
                except Exception as e:
                    print(f"⚠️ 提取训练环境VecNorm失败，将使用默认值：{e}")

                # ===== 新增：步骤2 - 同步到test_env，并确保冻结 =====
                try:
                    raw_test_env = test_env.unwrapped
                    while not isinstance(raw_test_env, TwoCarrierEnv) and raw_test_env is not None:
                        raw_test_env = getattr(raw_test_env, "_env", raw_test_env.unwrapped)
                    
                    if raw_test_env is not None:
                        # 覆盖test_env的初始mean/var为训练环境的最新值
                        raw_test_env.vecnorm_mean = train_vecnorm_mean
                        raw_test_env.vecnorm_var = train_vecnorm_var
                        # 强制确认冻结，避免意外更新
                        raw_test_env.vecnorm_frozen = True
                        print(f"✅ 已将训练环境VecNorm同步到test_env，且保持冻结状态")
                except Exception as e:
                    print(f"⚠️ 同步VecNorm到test_env失败：{e}")
                
                # 1. 执行原有评测逻辑（test_env已冻结VecNorm，eval_model内部验证状态）
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg_logger_num_test_episodes, eval_round=eval_round_counter
                )
                
                # 2. 更新评测指标：修正test_rewards已为均值，无需再调用.mean()
                metrics_to_log.update(
                    {
                        "eval/reward": test_rewards.item() if isinstance(test_rewards, torch.Tensor) else test_rewards,
                        "eval/round": eval_round_counter,
                    }
                )
                
                actor.train()
                print(f"============= 第 {eval_round_counter} 轮评测结束 =============")

        if logger:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            for key, value in metrics_to_log.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()

    # 训练结束后保存最终Checkpoint
    if save_interval > 0:
        save_checkpoint(cfg.collector.total_frames)
    
    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()
    pbar.close()
    print("\n🎉 训练完成，所有Checkpoint已保存（含VecNorm统计量）！")


if __name__ == "__main__":
    main()