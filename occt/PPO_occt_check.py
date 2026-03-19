# 模型训练脚本：PPO算法在OCCT环境中的实现
"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm.
适配：环境内部自定义VecNorm观测归一化，移除双重归一化，增强Checkpoint兼容性
并行版：支持 Windows/Linux，固定 VecNorm 统计量
"""
from __future__ import annotations
import sys
import os
import numpy as np
import functools # 新增：用于构造可序列化的环境函数
import torch.multiprocessing as mp # 新增：多进程管理

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
from torchrl.envs import ParallelEnv # 新增：并行环境包装器

# 导入工具函数
from utils_ppo_occt import eval_model, make_env, make_ppo_models


# =========================================================================
# 关键修改：定义模块级辅助函数 (必须在 main 之外，否则 Windows 报错)
# =========================================================================
def make_train_env_wrapper(env_name, device, shared_w_force):
    """
    用于在子进程中创建环境的包装函数。
    """
    return make_env(
        env_name,
        device=device,  # 子进程通常建议用 CPU，由 Collector 统一传到 GPU
        shared_w_force=shared_w_force,
        capture_video_data=False  # 训练时不捕获视频数据
    )


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
    from torchrl.record.loggers import generate_exp_name, get_logger

    torch.set_float32_matmul_precision("high")

    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # 并行环境数量配置 (从cfg读取，默认为4)
    num_envs = getattr(cfg.env, "num_envs", 4)
    # num_envs = 1
    print(f"🚀 启动并行训练 | 并行环境数: {num_envs} | 平台: {os.name}")

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

    # Create models
    actor, critic = make_ppo_models(cfg.env.env_name, device=device)

    # ================= [Curriculum Learning 设置] =================
    # 定义受力惩罚的“课程表”
    # 初始权重 (Phase 1): 很小，让它先学会跑
    W_FORCE_START = 1
    # [cite_start]最终权重 (Phase 2): 较大，迫使它优化受力
    W_FORCE_END = 5
    # 课程开始的帧数 (前20%的时间先不加压)
    CURRICULUM_START_FRAME = 0 
    # 课程结束的帧数 (在训练结束前达到最大权重)
    CURRICULUM_END_FRAME = cfg.collector.total_frames * 0.8 

    # 创建跨进程共享变量 (类型 'd' 代表 double/float)
    # 这个变量的值可以在主进程修改，所有子进程环境会自动读取到新值
    # shared_w_force = mp.Value('d', W_FORCE_START)
    manager = mp.Manager()
    shared_w_force = manager.Value('d', W_FORCE_START)
    # ==============================================================

    # # 固定统计量定义
    # FIXED_MEAN = [3.267685660834517, -0.3385894488464295, 1.9635006395349606, -0.1718040936161826, -0.12279842830682108, 0.007645809435244238, 0.011146266809907063, -0.0359171949460023, 0.02129872767178046, 339.37157534109247, 8.429046335233613, 0.0]
    # FIXED_VAR = [10.854110464128235, 1.1635751967064711, 4.684817522924974, 0.9177030245326943, 0.16467684585450817, 0.1034575355366116, 0.027449087149541657, 0.010220331556196624, 0.016829017219362707, 500327.52070539613, 1705366.736736945, 0.0]
    # FIXED_MEAN = None
    # FIXED_VAR = None
    # ================= 修改点1：使用 functools.partial + ParallelEnv 创建 Collector =================
    # 使用 partial 固定参数，确保 Windows 下可以被 pickle 序列化
    # 建议子环境使用 "cpu"，避免多进程竞争 GPU 资源，数据会由 Collector 统一转到 device
    train_env_factory = functools.partial(
        make_train_env_wrapper,
        env_name=cfg.env.env_name,
        device="cpu", 
        # fixed_mean=FIXED_MEAN,
        # fixed_var=FIXED_VAR,
        shared_w_force=shared_w_force
    )

    collector = Collector(
        create_env_fn=lambda: ParallelEnv(
            num_workers=num_envs,
            create_env_fn=train_env_factory,
            serial_for_single=True, # 如果 num_envs=1 自动切回串行
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

    def save_checkpoint(current_frames):
        """仅保存模型和配置值"""
        ckpt_dict = {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "cfg": cfg,
            "collected_frames": current_frames
        }
        save_dir = cfg.checkpoint.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"checkpoint_{current_frames}_frames.pt")
        torch.save(ckpt_dict, save_path)
        print(f"\n✅ Checkpoint saved to: {save_path}")

    # load_checkpoint 函数 (保持不变，无需修改)
    def load_checkpoint(ckpt_path, target_env=None):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint文件不存在：{ckpt_path}")
        ckpt_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt_dict["actor_state_dict"])
        critic.load_state_dict(ckpt_dict["critic_state_dict"])
        optim.load_state_dict(ckpt_dict["optim_state_dict"])
        return ckpt_dict["cfg"], ckpt_dict["collected_frames"]
    
    # ================= [新增] 断点重训/预训练加载逻辑 =================
    # 假设你在 config_occt.yaml 中添加了字段: checkpoint.load_path (默认为 null)
    # 或者直接在这里硬编码测试
    load_path = cfg.checkpoint.get("load_path", None)  
    # 示例： load_path = "checkpoints/checkpoint_1000000_frames.pt" 
    
    if load_path and os.path.exists(load_path):
        print(f"\n🔄 正在加载预训练模型: {load_path}")
        try:
            # 复用你已有的 load_checkpoint 函数，但我们需要微调一下调用方式
            # 注意：这里我们还没创建 Collector，所以暂时传 None 给 target_env
            # 如果 VecNorm 是固定的 (FIXED_MEAN)，则不需要从 checkpoint 恢复 env 统计量
            loaded_cfg, loaded_frames = load_checkpoint(load_path, target_env=None)
            
            print(f"✅ 模型权重已恢复 (原训练步数: {loaded_frames})")
            
            # 【策略选择】
            # 选项 A: 彻底的 "Resume" (恢复优化器状态，继续之前的训练流)
            # 适用于：机器断电了，想接着跑
            # -------------------------------------------------------------
            # collected_frames = loaded_frames 
            # print("   -> 模式: Resume (继承历史步数和优化器状态)")

            # 选项 B: "Finetune" / 课程学习 (重置优化器，重置步数，只保留网络权重)
            # 适用于：你的场景（第一阶段结束，改变超参如 max_steps，进入第二阶段）
            # -------------------------------------------------------------
            # 如果是课程学习，我们通常希望学习率重新开始衰减，或者使用较小的恒定学习率
            # 所以我们只加载网络权重，不加载 optim_state_dict (除非你想保持动量)
            # 这里我做一个折中：加载所有状态，但允许你手动重置 collected_frames
            
            # 如果你想让进度条从 0 开始 (适合课程学习的第二阶段):
            collected_frames = 0 
            print("   -> 模式: Curriculum Finetune (步数重置为0，在新配置下继续训练)")
            
        except Exception as e:
            print(f"❌ 加载 Checkpoint 失败: {e}")
            raise e
    else:
        print("\n🆕 未指定 load_path 或文件不存在，开始从头训练")
    # ====================================================================

    # Create logger
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

    # Create test environment：保持串行即可，使用固定统计量
    test_env = make_env(
        cfg.env.env_name,
        device,
        from_pixels=logger_video,
        render_mode=None,
        enable_visualization=False,
        shared_w_force=shared_w_force,
        capture_video_data=True  # 评测时捕获视频数据
    )
    test_env.eval()

    # Update函数
    def update(batch, num_network_updates):
        optim.zero_grad(set_to_none=True)
        alpha = torch.ones((), device=device)
        if cfg.optim.anneal_lr:
            alpha = 1 - (num_network_updates / total_network_updates)
            for group in optim.param_groups:
                group["lr"] = cfg.optim.lr * alpha
        if cfg.loss.anneal_clip_epsilon:
            loss_module.clip_epsilon.copy_(cfg.loss.clip_epsilon * alpha)
        num_network_updates = num_network_updates + 1

        loss = loss_module(batch)
        critic_loss = loss["loss_critic"]
        actor_loss = loss["loss_objective"] + loss["loss_entropy"]
        total_loss = critic_loss + actor_loss

        total_loss.backward()
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

    eval_round_counter = 0

    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_lr = torch.tensor(cfg.optim.lr, device=device)
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon # 修正变量名引用
    cfg_logger_test_interval = cfg.logger.test_interval
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes
    losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])


    save_interval = cfg.checkpoint.save_interval
    last_saved_frames = 0

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

        if save_interval > 0:
            if collected_frames // save_interval > last_saved_frames // save_interval:
                save_checkpoint(collected_frames)
                last_saved_frames = collected_frames

        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            metrics_to_log.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item() / len(episode_length),
                }
            )

        # 提取并统计 Reward 分项
        try:
            reward_keys = [
                "reward_r_force", "reward_r_align_rear", "reward_r_align_front",
                # "reward_r_smooth", 
                "reward_r_progress", "reward_r_stability",
                "reward_val_force", "reward_val_delta_psi_rear", "reward_val_delta_psi_front"
            ]
            next_td = data["next"]
            for key in reward_keys:
                if key in next_td.keys():
                    val_mean = next_td[key].float().mean().item()
                    metrics_to_log[f"reward_parts/{key}"] = val_mean
                elif ("info", key) in next_td.keys(include_nested=True):
                    val_mean = next_td["info", key].float().mean().item()
                    metrics_to_log[f"reward_parts/{key}"] = val_mean
        except Exception as e:
            if i == 0: print(f"⚠️ 提取 Reward Details 失败: {e}")

        # --- 核心：计算并更新当前权重 ---
        # 1. 计算当前进度 (0.0 ~ 1.0)
        progress = (collected_frames - CURRICULUM_START_FRAME) / (CURRICULUM_END_FRAME - CURRICULUM_START_FRAME)
        progress = np.clip(progress, 0.0, 1.0) # 限制在 0~1 之间

        # 2. 线性插值计算当前权重
        current_w_force = W_FORCE_START + (W_FORCE_END - W_FORCE_START) * progress
        
        # 3. 更新共享变量 (所有并行环境会立即生效)
        shared_w_force.value = current_w_force

        # 4. 记录到 TensorBoard
        if i % 10 == 0:
            print(f"Frame {collected_frames}: w_force updated to {current_w_force:.5f}")
        metrics_to_log["train/w_force"] = current_w_force  # 记录到 wandb/tensorboard 方便观察
        # ------------------------------------

        with timeit("training"):
            for j in range(cfg_loss_ppo_epochs):
                with torch.no_grad(), timeit("adv"):
                    torch.compiler.cudagraph_mark_step_begin()
                    data = adv_module(data)
                    if compile_mode:
                        data = data.clone()

                with timeit("rb - extend"):
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

        # ================= 修改点3：简化评测 (无需同步 VecNorm) =================
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC), timeit("eval"):
            if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
                i * frames_in_batch
            ) // cfg_logger_test_interval:
                eval_round_counter += 1
                actor.eval()
                print(f"\n============= 开始第 {eval_round_counter} 轮评测 =============")
                test_rewards = eval_model(
                    actor, test_env, num_episodes=cfg_logger_num_test_episodes, eval_round=eval_round_counter
                )
                
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

    if save_interval > 0:
        save_checkpoint(cfg.collector.total_frames)
    
    collector.shutdown()
    if not test_env.is_closed:
        test_env.close()
    pbar.close()
    print("\n🎉 训练完成，所有Checkpoint已保存！")

# ================= 修改点4：Windows多进程入口保护 =================
if __name__ == "__main__":
    # 强制使用 spawn，解决 Windows pickling 问题及 CUDA 兼容性
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()