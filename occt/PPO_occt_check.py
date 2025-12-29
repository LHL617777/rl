# 模型训练脚本：PPO算法在OCCT环境中的实现

"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm.
"""
from __future__ import annotations
import sys
import os
# 添加本地自定义torchrl的根目录（E:\rl\torchrl\的上一级目录，即E:\rl\）
sys.path.insert(0, "E:\\rl")  # insert(0)表示将该路径放在搜索优先级第1位
# sys.path.insert(0, "/home/yons/Graduation/rl")  # Linux系统中的路径

import warnings

import hydra
import torchrl
from torchrl._utils import compile_with_warmup
from torchrl.record.loggers.swanlab import SwanLabLogger
import gymnasium as gym
from occt_2d2c import TwoCarrierEnv
# 补全缺失的导入
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

    # Create models (check utils_mujoco.py)
    actor, critic = make_ppo_models(cfg.env.env_name, device=device)

    # Create collector
    collector = Collector(
        create_env_fn=make_env(cfg.env.env_name, device),
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

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo",
            experiment_name=exp_name,
            # wandb_kwargs={
            #     "config": dict(cfg),
            #     "project": cfg.logger.project_name,
            #     "group": cfg.logger.group_name,
            # },
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

    # Create test environment
    # 1. 传递可视化参数给make_env
    test_env = make_env(
        cfg.env.env_name, 
        device, 
        from_pixels=logger_video,
        render_mode="rgb_array",  # 开启自定义rgb_array渲染模式
        enable_visualization=True  # 开启自定义可视化功能
    )
    test_env.eval()

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

    # ===================== 替换：手动保存Checkpoint =====================
    def save_checkpoint(current_frames):
        """封装Checkpoint保存逻辑，适配cfg配置"""
        ckpt_dict = {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "cfg": cfg,
            "collected_frames": current_frames,
        }
        save_dir = cfg.checkpoint.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"checkpoint_{current_frames}_frames.pt")
        torch.save(ckpt_dict, save_path)
        print(f"\n✅ Checkpoint saved to: {save_path}")

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

        # ===== 新增：手动判断是否达到保存间隔，触发Checkpoint保存 =====
        if save_interval > 0:
            if collected_frames // save_interval > last_saved_frames // save_interval:
                save_checkpoint(collected_frames)
                last_saved_frames = collected_frames

        # Get training rewards and episode lengths
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

        # Get training losses and times
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

        # Get test rewards
        with torch.no_grad(), set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), timeit("eval"):
            if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
                i * frames_in_batch
            ) // cfg_logger_test_interval:
                eval_round_counter += 1  # 自增评测轮次
                actor.eval()
                print(f"\n============= 开始第 {eval_round_counter} 轮评测 =============")
                
                # 1. 执行原有评测逻辑
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg_logger_num_test_episodes, eval_round=eval_round_counter
                )
                
                # 2. 解除环境包装，获取原始TwoCarrierEnv实例
                try:
                    raw_test_env = test_env.unwrapped
                    while not isinstance(raw_test_env, TwoCarrierEnv):
                        raw_test_env = raw_test_env.unwrapped
                except Exception as e:
                    print(f"获取原始环境实例失败，无法保存/上传视频：{e}")
                    raw_test_env = None
                
                # 3. 调用自定义方法保存单轮评测视频
                video_filepath = None
                if raw_test_env is not None:
                    # 保存视频（传入评测轮次，便于命名区分）
                    video_filepath = raw_test_env.save_eval_video(
                        eval_round=eval_round_counter,
                        video_save_dir=cfg.checkpoint.checkpoint_dir
                    )
                    # 清空帧列表，为下一轮评测做准备
                    raw_test_env.clear_render_frames()
                    # 标记仿真结束，避免reset清空帧列表（可选）
                    raw_test_env.mark_sim_finished()
                
                # 4. 更新评测指标（保留原有奖励记录）
                metrics_to_log.update(
                    {
                        "eval/reward": test_rewards.mean(),
                        "eval/round": eval_round_counter,
                    }
                )
                
                # 5. 复用swanlab.py，将本地视频上传到SwanLab日志（核心复用点）
                if logger and video_filepath is not None and os.path.exists(video_filepath):
                    # 构造视频日志名称
                    video_log_name = f"eval_video/round_{eval_round_counter}"
                    # 调用修正后的swanlab.log_video（或直接调用swanlab.log）
                    # 复用SwanLabLogger的log_video方法（已修正）
                    logger.log_video(
                        name=video_log_name,
                        video_path=video_filepath,
                        step=collected_frames  # 关联训练累计帧数，便于日志对齐
                    )
                    print(f"第 {eval_round_counter} 轮评测视频已上传至SwanLab日志")
                
                actor.train()
                print(f"============= 第 {eval_round_counter} 轮评测结束 =============")

        if logger:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            for key, value in metrics_to_log.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()

    # ===================== 新增：训练结束后保存最终Checkpoint =====================
    if save_interval > 0:
        save_checkpoint(cfg.collector.total_frames)
    
    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()
    pbar.close()
    print("\n🎉 训练完成，所有Checkpoint已保存！")


if __name__ == "__main__":
    main()