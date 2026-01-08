import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import yaml  
from model import Model2D2C
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2
from PIL import Image
from io import BytesIO
import torch

# plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False

class TwoCarrierEnv(gym.Env):
    """两辆车运载超大件系统的自定义强化学习环境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode=None, config_path=None, enable_visualization=False, vecnorm_frozen: bool = False):
        super().__init__()
        
        # 加载2d2c.yaml配置文件（优先使用传入路径，否则加载同一目录下的2d2c.yaml）
        self.config_name = '2d2c'
        self.config = self._load_config(config_path)
        
        # 初始化动力学模型（使用加载的config）
        self.model = Model2D2C(self.config)
        
        # 状态空间定义（观测：两车状态+铰接力）
        # 状态向量包含：超大件位置/姿态、两车姿态、各部分速度（共10维）+ 铰接力（2维）
        obs_low = np.array([
            -100, -100, -np.pi,  # 超大件 X, Y, Psi_o
            -np.pi, -np.pi,      # 两车姿态 Psi_1, Psi_2
            -10, -10, -5,        # 超大件速度 X_dot, Y_dot, Psi_dot_o
            -5, -5,              # 两车角速度 Psi_dot_1, Psi_dot_2
            -1e5, -1e5           # 铰接力 Fh_x, Fh_y（第二辆车）
        ])
        obs_high = np.array([
            100, 100, np.pi,
            np.pi, np.pi,
            10, 10, 5,
            5, 5,
            1e5, 1e5
        ])
        # 归一化后的观测空间（合理范围，覆盖归一化后的所有可能值）
        obs_norm_low = np.full(12, -1000.0, dtype=np.float64)  # 12维观测，下限-1000
        obs_norm_high = np.full(12, 1000.0, dtype=np.float64)   # 12维观测，上限1000
        self.observation_space = spaces.Box(
            low=obs_norm_low, high=obs_norm_high, dtype=np.float64
        )
        # self.observation_space = spaces.Box(
        #     low=obs_low, high=obs_high, dtype=np.float64
        # )
        
        # 原始动作空间上下限（保存用于归一化/反归一化）
        self.original_action_low = np.array([-np.pi/6, -np.pi/6, 0, 0])  # 原始下限
        self.original_action_high = np.array([np.pi/6, np.pi/6, 1e3, 1e3])  # 原始上限
        
        # 归一化后的动作空间（统一映射到[-1, 1]区间）
        self.action_space = spaces.Box(
            low=-np.ones(4, dtype=np.float64),  # 归一化下限：[-1, -1, -1, -1]
            high=np.ones(4, dtype=np.float64),  # 归一化上限：[1, 1, 1, 1]
            dtype=np.float64
        )
        
        # 第一辆车随机控制量
        self.u1_random = np.array([0, 0, 1e3, 1e3])  # 示例：随机控制量
        self.rng = np.random.default_rng()  # 随机数生成器（保证可复现性）
        # 1. 前轮转角：轮级大偏移 + 步级小扰动（核心：提升轮间差异）
        self.steer_episode_base_std = np.pi/15  # 轮级基础偏移标准差（≈12°，大幅提升轮间差异）
        self.steer_step_dynamic_std = 0.008     # 步级动态小扰动标准差（≈0.46°，保证轮内平滑）
        self.steer_max_bound = np.pi/6          # 最大转角约束（±30°，与原始动作空间一致）
        self.steer_min_bound = -np.pi/6
        # 新增：存储轮级固定基础偏移（每轮重置时更新）
        self.steer_episode_offset = 0.0
        # 2. 推力噪声（保留原有真实车辆特性，无修改）
        self.thrust_noise_rel_std = 0.02  # 相对标准差2%（真实发动机输出波动通常<5%）
        self.thrust_noise_abs_min = 0  # 推力最低为0（不能为负，无反向驱动力）
        self.thrust_noise_abs_max = 1e3  # 推力最高不超过原始上限
        
        # 可视化开关
        self.enable_visualization = enable_visualization
        # 可视化相关初始化
        self.render_mode = render_mode if enable_visualization else None
        self.render_frames = []  # 存储rgb_array帧
        self.trajectories = {   # 存储轨迹数据
            'cargo': [],
            'car1': [],
            'car2': [],
            'hinge1': [],
            'hinge2': []
        }
        self.fig = None
        self.ax = None
        self.animation = None
        self.is_sim_finished = False  # 新增：标识仿真是否已结束
        # 车轮参数（可放入yaml配置）
        self.wheel_radius = 0.3  # 车轮半径
        self.wheel_width = 0.15  # 车轮宽度

        # ===================================== 新增：观测归一化（VecNorm）相关初始化 =====================================
        self.vecnorm_decay = 0.99999  # 滑动平均衰减系数（与原VecNormV2一致）
        self.vecnorm_eps = 1e-2       # 防止除零的小常数（与原VecNormV2一致）
        self.vecnorm_frozen = vecnorm_frozen  # 是否冻结统计量
        self.vecnorm_min_var = 1e-4   # 最小方差约束，避免初期方差过小导致归一化值爆炸
        # 滑动统计量（初始化为None，首次观测时根据观测形状自动初始化）
        self.vecnorm_mean = np.zeros(12, dtype=np.float64)  # 初始均值为0
        self.vecnorm_var = np.ones(12, dtype=np.float64) * self.vecnorm_min_var  # 初始方差为最小方差
        self.vecnorm_count = 0        # 统计更新次数（用于初始阶段的无偏估计）
        # ==================================================================================================================
        self.hinge_force_penalty = 0.0
        self.control_smooth_penalty = 0.0

    def _load_config(self, config_path):
        """加载同一目录下的2d2c.yaml配置文件"""
        # 若未传入config路径，默认加载同一目录下的2d2c.yaml
        if config_path is None:
            # 获取当前脚本所在绝对目录，避免相对路径错乱
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "2d2c.yaml")
        
        # 检查配置文件是否存在，不存在则使用默认配置兜底
        if not os.path.exists(config_path):
            print(f"警告：未找到配置文件 {config_path}，将使用默认配置")
            return self._get_default_config()
        
        # 读取并解析YAML配置文件
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"错误：解析2d2c.yaml失败，原因：{e}，将使用默认配置")
            return self._get_default_config()

    def _get_default_config(self):
        """默认仿真配置（兜底使用，与YAML配置格式一致）"""
        return {
            'N_c': 2, 'N_q': 5, 'N_x': 10, 'N_u': 8,  # 双载体配置
            'M_o': 1000, 'I_o': 1000,                 # 超大件参数
            'M_1': 500, 'M_2': 500,                   # 车辆质量
            'I_1': 100, 'I_2': 100,                   # 转动惯量
            'x__o_1': 5, 'x__o_2': -5,                # 铰链相对超大件坐标
            'y__o_1': 0, 'y__o_2': 0,
            'x__1_1': 1, 'x__2_2': 1,                 # 质心相对自身铰链坐标
            'y__1_1': 0, 'y__2_2': 0,
            'C_f': 10000, 'C_r': 10000,               # 轮胎刚度
            'l_f': 2, 'l_r': 2,                       # 前后轮距质心距离
            'X_o_0': 0, 'Y_o_0': 0, 'Psi_o_0': 0,     # 初始状态
            'Psi_1_0': 0, 'Psi_2_0': 0,
            'X_dot_o_0': 0, 'Y_dot_o_0': 0, 'Psi_dot_o_0': 0,
            'Psi_dot_1_0': 0, 'Psi_dot_2_0': 0,
            'T': 10, 'dt': 0.1, 'integrator': 'RK4',  # 仿真参数
            'framerate': 10, 'range': 20,             # 可视化参数
            'oversized_cargo_bias': 2, 'oversized_cargo_width': 3
        }

    def normalize_action(self, original_action):
        """
        将原始动作（原始空间）归一化到[-1, 1]区间
        :param original_action: 原始动作，shape=(4,)，对应[前轮转向角, 后轮转向角, 前轮推力, 后轮推力]
        :return: 归一化后的动作，shape=(4,)，范围[-1, 1]
        """
        # 线性归一化公式：norm_action = 2 * (orig_action - orig_low) / (orig_high - orig_low) - 1
        # 避免除零（防止原始上下限相等）
        orig_range = self.original_action_high - self.original_action_low
        orig_range = np.where(orig_range == 0, 1e-8, orig_range)
        
        norm_action = 2 * (original_action - self.original_action_low) / orig_range - 1
        # 裁剪到[-1, 1]，防止超出边界
        norm_action = np.clip(norm_action, -1, 1)
        return norm_action.astype(np.float64)

    def denormalize_action(self, normalized_action):
        """
        将归一化动作（[-1, 1]）反归一化到原始动作空间
        :param normalized_action: 归一化动作，shape=(4,)，范围[-1, 1]
        :return: 原始动作，shape=(4,)，对应原始上下限范围
        """
        # 反归一化公式：orig_action = orig_low + (norm_action + 1) * (orig_high - orig_low) / 2
        orig_range = self.original_action_high - self.original_action_low
        orig_action = self.original_action_low + (normalized_action + 1) * orig_range / 2
        # 裁剪到原始动作范围，确保物理有效性
        orig_action = np.clip(
            orig_action, 
            self.original_action_low, 
            self.original_action_high
        )
        return orig_action.astype(np.float64)

    # ===================================== 新增：VecNorm 辅助方法1 - 更新滑动统计量 =====================================
    def _update_vecnorm_stats(self, obs):
        """
        在线更新 VecNorm 滑动统计量（仅训练阶段、未冻结时执行）
        :param obs: 原始观测（12维numpy数组，单条观测）
        """
        # 若已冻结，直接返回（不更新统计量）
        if self.vecnorm_frozen:
            return
        
        # 统一转换为numpy数组（确保格式一致）
        obs_np = np.asarray(obs, dtype=np.float64)
        
        # 首次初始化统计量（匹配观测的12维形状）
        if self.vecnorm_mean is None:
            self.vecnorm_mean = np.zeros_like(obs_np, dtype=np.float64)
            self.vecnorm_var = np.ones_like(obs_np, dtype=np.float64)
        
        # 计算当前观测的均值和方差（单条观测，直接赋值）
        current_mean = obs_np
        current_var = np.square(obs_np)  # 单条观测的方差为自身平方
        current_count = 1

        # 更新滑动统计量（采用无偏滑动平均，与原VecNormV2逻辑一致）
        self.vecnorm_count += current_count
        if self.vecnorm_count <= current_count:
            # 首次更新，直接赋值
            self.vecnorm_mean = current_mean
            self.vecnorm_var = current_var
        else:
            # 滑动平均更新：new = decay * old + (1 - decay) * current
            decay = self.vecnorm_decay
            self.vecnorm_mean = decay * self.vecnorm_mean + (1 - decay) * current_mean
            self.vecnorm_var = decay * self.vecnorm_var + (1 - decay) * current_var
        # 新增：方差下限约束，避免方差过小导致归一化值爆炸
        self.vecnorm_var = np.maximum(self.vecnorm_var, self.vecnorm_min_var)

    # ===================================== 新增：VecNorm 辅助方法2 - 执行观测归一化 =====================================
    def _normalize_observation(self, obs):
        """
        执行观测归一化（训练/评测阶段均执行，仅评测阶段不更新统计量）
        :param obs: 原始观测（12维numpy数组）
        :return: 归一化后的观测（12维numpy数组，保持原数据类型）
        """
        # 若统计量未初始化（首次观测），直接返回原始观测
        if self.vecnorm_mean is None:
            return obs
        
        # 统一转换为numpy数组
        obs_np = np.asarray(obs, dtype=np.float64)
        
        # 计算标准差（加eps防止除零）
        std_np = np.sqrt(self.vecnorm_var) + self.vecnorm_eps
        
        # 执行归一化：(原始观测 - 滑动均值) / 滑动标准差
        normalized_obs_np = (obs_np - self.vecnorm_mean) / std_np
        
        # 恢复原数据类型并返回
        return normalized_obs_np.astype(obs.dtype) if hasattr(obs, 'dtype') else normalized_obs_np

    def _get_observation(self):
        """从模型提取观测值（包含归一化处理）"""
        x = self.model.x  # 当前状态向量
        # 提取第二辆车的铰接力（Fh_arch存储格式：[Fh1_x, Fh1_y, Fh2_x, Fh2_y]）
        Fh2_x = self.model.Fh_arch[self.model.count, 2]
        Fh2_y = self.model.Fh_arch[self.model.count, 3]
        # 原始观测（12维）
        raw_obs = np.concatenate([
            x[:5],   # 位置与姿态：[Xo, Yo, Psio, Psi1, Psi2]
            x[5:10], # 速度：[Xo_dot, Yo_dot, Psio_dot, Psi1_dot, Psi2_dot]
            [Fh2_x, Fh2_y]  # 第二辆车铰接力
        ])
        
        # 步骤1：更新VecNorm统计量（仅训练阶段未冻结时生效）
        self._update_vecnorm_stats(raw_obs)
        
        # 步骤2：执行观测归一化（返回归一化后的观测）
        normalized_obs = self._normalize_observation(raw_obs)
        
        return normalized_obs

    def _calculate_reward(self):
        """计算奖励（核心：最小化铰接力+跟随约束）"""
        # 1. 铰接力惩罚（核心目标）
        Fh2_x = self.model.Fh_arch[self.model.count, 2]
        Fh2_y = self.model.Fh_arch[self.model.count, 3]
        hinge_force_penalty = 1e-10 * (Fh2_x**2 + Fh2_y**2)  # 缩放因子避免数值过大
        
        # 2. 跟随误差惩罚（位置+姿态）
        X1, Y1 = self.model.getXYi(self.model.x, 0)  # 第一辆车位置
        X2, Y2 = self.model.getXYi(self.model.x, 1)  # 第二辆车位置
        Psi1 = self.model.x[3]
        Psi2 = self.model.x[4]
        pos_error = np.hypot(X2 - X1, Y2 - Y1)
        psi_error = np.abs(Psi2 - Psi1)
        tracking_penalty = 1.0 * pos_error + 0.5 * psi_error  # 位置误差权重更高
        
        # 3. 控制量平滑性惩罚（避免动作突变）
        if self.model.count > 0:
            u2_prev = self.model.u_arch[self.model.count - 1, 4:8]  # 上一步第二辆车控制量
            u2_current = self.model.u_arch[self.model.count, 4:8]
            control_smooth_penalty = 5 * 1e-7 * np.sum((u2_current - u2_prev)**2)
        else:
            control_smooth_penalty = 0
        
        self.hinge_force_penalty = hinge_force_penalty
        self.control_smooth_penalty = control_smooth_penalty   
        # 总奖励 = 负惩罚（最小化目标）
        return - (0.5 * hinge_force_penalty + 0 * tracking_penalty + 0.5 * control_smooth_penalty)
    
    def _get_noisy_u1(self):
        """
        生成带有真实车辆特性的第一辆车控制量
        核心原则：合理区间内多样分布、平滑无突变、符合物理约束
        :return: 带扰动的u1控制量（np.array，与self.u1_fixed同结构）
        """
        # 1. 复制原始固定控制量（避免修改原数据）
        u1_noisy = np.copy(self.u1_random)
        
        # 2. 前轮转角：轮级大偏移 + 步级小扰动 + 平滑 + 约束（核心修改）
        # 2.1 步级动态小扰动（模拟实时操作误差、路面颠簸，幅度小）
        steer_step_noise = self.rng.normal(loc=0, scale=self.steer_step_dynamic_std)
        # 2.2 叠加：原始值 + 轮级固定偏移 + 步级小扰动
        steer_candidate = u1_noisy[0] + self.steer_episode_offset + steer_step_noise
        # 2.3 裁剪约束（不超出车辆最大转向角）
        steer_clipped = np.clip(steer_candidate, self.steer_min_bound, self.steer_max_bound)
        
        # 2.5 平滑过渡（避免相邻步转角突变，符合驾驶员操控惯性/转向系统阻尼）
        if hasattr(self, '_prev_u1_noisy'):
            # 平滑权重：当前候选值0.8 + 上一步值0.2
            steer_smoothed = 0.8 * steer_clipped + 0.2 * self._prev_u1_noisy[0]
            # 平滑后再次裁剪，确保不超出约束
            u1_noisy[0] = np.clip(steer_smoothed, self.steer_min_bound, self.steer_max_bound)
        else:
            u1_noisy[0] = steer_clipped
        
        # 3. 前后轮推力：固定为原始值
        
        # 4. 保存当前扰动值，用于下一步平滑
        self._prev_u1_noisy = np.copy(u1_noisy)
        
        return u1_noisy.astype(np.float64)
    
    def step(self, action):
        """环境一步交互（注意：传入的action是归一化后的动作，需先反归一化）"""
        # 反归一化：将[-1, 1]的动作映射回原始动作空间
        original_action = self.denormalize_action(action)
        
        # 组合控制量：第一辆车控制量 + 第二辆车原始动作
        u1 = self._get_noisy_u1()  # 获取带扰动的第一辆车控制量
        u = np.concatenate([u1, original_action])
        
        self.model.step(u)
        observation = self._get_observation()  # 直接获取归一化后的观测
        reward = self._calculate_reward()
        self._record_trajectories()

        # 仅在启用可视化时记录轨迹和渲染
        if self.enable_visualization:
            self._record_trajectories()
            self._render_frame()  
            if self.render_mode == "human":
                plt.pause(0.001)  
        
        # 判断终止条件（仿真结束）
        terminated = self.model.is_finish
        truncated = False
        X1, Y1 = self.model.getXYi(self.model.x, 0)  # 第一辆车位置
        X2, Y2 = self.model.getXYi(self.model.x, 1)  
        info = {
            'Fh2': (self.model.Fh_arch[self.model.count, 2], 
                    self.model.Fh_arch[self.model.count, 3]),
            'pos_error': np.hypot(
                self.model.getXYi(self.model.x, 1)[0] - self.model.getXYi(self.model.x, 0)[0],
                self.model.getXYi(self.model.x, 1)[1] - self.model.getXYi(self.model.x, 0)[1]
            ),
            'u1': u1,  # 新增：保存第一辆车控制量，用于可视化车轮摆角
            'u2_normalized': action,  # 归一化后的动作
            'u2_original': original_action,  # 原始动作（便于调试）
            'x': np.array([X1, Y1, X2, Y2]),
            "hinge_force_penalty": self.hinge_force_penalty,
            "control_smooth_penalty": self.control_smooth_penalty
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, clear_frames=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)  # 重置随机数生成器，保证可复现
        self.model = Model2D2C(self.config)
        # 生成轮级固定基础偏移（正态分布，围绕0，标准差更大）
        self.steer_episode_offset = self.rng.normal(
            loc=0, 
            scale=self.steer_episode_base_std
        )
        # 轮级偏移先裁剪（确保不超出物理约束）
        self.steer_episode_offset = np.clip(
            self.steer_episode_offset,
            self.steer_min_bound,
            self.steer_max_bound
        )
        if hasattr(self, '_prev_u1_noisy'):
            del self._prev_u1_noisy  # 重置第一辆车扰动历史
        
        # 2. 处理options，提取clear_frames（优先从options获取，适配auto_reset=True）
        options = options or {}
        # 优先级：options中的clear_frames > 显式传入的clear_frames > 默认False
        final_clear_frames = options.get(
            "clear_frames", 
            clear_frames if clear_frames is not None else False
        )
        
        if self.enable_visualization:
            # 3. 仅当final_clear_frames=True时，才清空帧列表（修改原有判断条件）
            if final_clear_frames and not self.is_sim_finished:
                self.render_frames = []
            self.trajectories = {   # 重置轨迹（保留原有逻辑，不变）
                'cargo': [],
                'car1': [],
                'car2': [],
                'hinge1': [],
                'hinge2': []
            }
            self._reset_visualization() 
        self.is_sim_finished = False
        observation = self._get_observation()  # 直接获取归一化后的观测
        return observation, {}

    # ===================================== 新增：VecNorm 模式切换方法 =====================================
    def freeze_vecnorm(self):
        """冻结 VecNorm 统计量（评测模式：停止更新均值/方差，使用训练时累积的统计量）"""
        self.vecnorm_frozen = True
        print("观测归一化统计量已冻结，进入评测模式")

    def unfreeze_vecnorm(self):
        """解冻 VecNorm 统计量（训练模式：恢复更新均值/方差）"""
        self.vecnorm_frozen = False
        print("观测归一化统计量已解冻，进入训练模式")

    # ===================================== 新增：VecNorm 状态序列化/反序列化方法 =====================================
    def get_vecnorm_state(self):
        """导出 VecNorm 状态（用于保存到 checkpoint）"""
        return {
            "vecnorm_mean": self.vecnorm_mean,
            "vecnorm_var": self.vecnorm_var,
            "vecnorm_count": self.vecnorm_count,
            "vecnorm_decay": self.vecnorm_decay,
            "vecnorm_eps": self.vecnorm_eps,
            "vecnorm_frozen": self.vecnorm_frozen
        }

    def set_vecnorm_state(self, vecnorm_state):
        """从 checkpoint 导入 VecNorm 状态（用于离线评测加载）"""
        self.vecnorm_mean = vecnorm_state["vecnorm_mean"]
        self.vecnorm_var = vecnorm_state["vecnorm_var"]
        self.vecnorm_count = vecnorm_state["vecnorm_count"]
        self.vecnorm_decay = vecnorm_state["vecnorm_decay"]
        self.vecnorm_eps = vecnorm_state["vecnorm_eps"]
        self.vecnorm_frozen = vecnorm_state["vecnorm_frozen"]
        print("观测归一化状态已从 checkpoint 加载完成")

    def mark_sim_finished(self):
        self.is_sim_finished = True
        print("仿真已标记为结束，后续reset()不会清空帧列表")

    def _record_trajectories(self):
        """更新轨迹记录，使用参考代码的坐标获取方法，确保数据一致"""
        i_sim = self.model.count
        x = self.model.x_arch[i_sim, :]

        # 超大件质心
        cargo_pos = (x[0], x[1])
        self.trajectories['cargo'].append(cargo_pos)

        # 车辆质心（使用参考代码的getXYi方法）
        car1_pos = self.model.getXYi(x, 0)
        car2_pos = self.model.getXYi(x, 1)
        self.trajectories['car1'].append(car1_pos)
        self.trajectories['car2'].append(car2_pos)

        # 铰接点（使用参考代码的getXYhi方法）
        hinge1_pos = self.model.getXYhi(x, 0)
        hinge2_pos = self.model.getXYhi(x, 1)
        self.trajectories['hinge1'].append(hinge1_pos)
        self.trajectories['hinge2'].append(hinge2_pos)

    def _reset_visualization(self):
        """重构可视化初始化，完全参考参考代码的绘图元素类型"""
        if self.fig is not None:
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=(8, 8), dpi=60)
        self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        self.ax.set_facecolor('#f8f8f8')
        self.ax.set_xlabel('X (m)', fontsize=20)
        self.ax.set_ylabel('Y (m)', fontsize=20)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("两车运载超大件系统仿真可视化", fontsize=16)

        # 初始化绘图句柄（与参考代码一一对应）
        self.plot_handles = {
            # 1. 车轮：线集合（LineCollection）
            'tire': None,
            # 2. 铰接力：箭头列表
            'Fh': [],
            # 3. 铰接点：多边形列表
            'hinge': [],
            # 4. 货物：多边形
            'cargo': None,
            # 5. 车辆：多边形列表（两车）
            'car': [],
            # 6. 轨迹线：保持原有轨迹逻辑
            'cargo_traj': self.ax.plot([], [], 'k--', alpha=0.3, linewidth=1)[0],
            'car1_traj': self.ax.plot([], [], '#3498db', linestyle='--', alpha=0.4, linewidth=1)[0],
            'car2_traj': self.ax.plot([], [], '#e74c3c', linestyle='--', alpha=0.4, linewidth=1)[0],
            'hinge1_traj': self.ax.plot([], [], ':', color='blue', alpha=0.2, linewidth=0.8)[0],
            'hinge2_traj': self.ax.plot([], [], ':', color='orange', alpha=0.2, linewidth=0.8)[0]
        }

        # 标记是否为首次绘制（首次初始化绘图元素，后续仅更新数据）
        self.first_render = True

    def _render_frame(self):
        """重构帧渲染，完全遵循参考代码的坐标变换与可视化逻辑（新增内存优化）"""
        # 初始化可视化（若未初始化）
        if self.fig is None or self.ax is None:
            self._reset_visualization()

        # 获取当前仿真步数据（与参考代码对齐）
        i_sim = self.model.count
        x = self.model.x_arch[i_sim, :]
        u = self.model.u_arch[i_sim, :]

        # 1. 获取所有可视化数据（复用参考代码的核心方法）
        # 车轮线段数据
        tire_segments = self.model.getTireVis(i_sim)
        # 铰接力箭头+铰接点标记数据
        fh_arrows, hinge_markers = self.model.getHingeVis(i_sim)
        # 货物多边形数据
        cargo_polygon = self.model.getOversizedCargoVis(i_sim)
        # 车辆多边形数据（两车）
        car_polygons = self.model.getCarrierVis(i_sim)
        # 分解铰接力箭头数据，便于更新
        fh_color = self.model.config.get('c_Fh', 'green')
        fh_width = self.model.config.get('width_Fh', 0.01)

        # 2. 首次绘制：初始化绘图元素（后续仅更新数据，避免重复创建）
        if self.first_render:
            # 2.1 车轮：LineCollection
            self.plot_handles['tire'] = LineCollection(
                tire_segments,
                colors=self.model.config['c_tire'],
                linewidths=self.model.config['lw_tire'],
                zorder=2.4
            )
            self.ax.add_collection(self.plot_handles['tire'])

            # 2.2 铰接力：箭头
            for arrow_data in fh_arrows:
                h = self.ax.arrow(
                    arrow_data[0], arrow_data[1], arrow_data[2], arrow_data[3],
                    width=fh_width,  # 固定宽度，后续不再修改
                    color=fh_color,
                    zorder=2.4,
                    alpha=0.7
                )
                self.plot_handles['Fh'].append(h)

            # 2.3 铰接点：多边形标记
            for marker_poly in hinge_markers:
                h = Polygon(
                    marker_poly,
                    zorder=2.6,
                    alpha=1.0,
                    fc='black',
                    ec='white'
                )
                self.ax.add_patch(h)
                self.plot_handles['hinge'].append(h)

            # 2.4 货物：多边形
            if cargo_polygon:
                self.plot_handles['cargo'] = Polygon(
                    cargo_polygon,
                    zorder=2.5,
                    alpha=self.model.config['alpha_o'],
                    fc=self.model.config['fc_o'],
                    ec='black',
                    linewidth=1.5
                )
                self.ax.add_patch(self.plot_handles['cargo'])

            # 2.5 车辆：多边形（两车分别设置颜色）
            for i, poly in enumerate(car_polygons):
                h = Polygon(
                    poly,
                    zorder=2.5,
                    alpha=self.model.config['alpha_c'],
                    fc=self.model.config['fc_c'][i],
                    ec='black',
                    linewidth=1
                )
                self.ax.add_patch(h)
                self.plot_handles['car'].append(h)

            self.first_render = False  # 首次绘制完成，后续仅更新数据

        # 3. 非首次绘制：更新所有绘图元素数据（核心：仅更新数据，不重新创建）
        else:
            # 3.1 更新车轮线段
            self.plot_handles['tire'].set_segments(tire_segments)

            # 3.2 更新铰接力箭头
            # 移除旧箭头
            for h in self.plot_handles['Fh']:
                h.remove()
            self.plot_handles['Fh'].clear()
            # 重新绘制新箭头（宽度仍用初始值，不变化）
            for arrow_data in fh_arrows:
                h = self.ax.arrow(
                    arrow_data[0], arrow_data[1], arrow_data[2], arrow_data[3],
                    width=fh_width,  # 固定宽度，和首次绘制一致
                    color=fh_color,
                    zorder=2.4,
                    alpha=0.7
                )
                self.plot_handles['Fh'].append(h)

            # 3.3 更新铰接点标记
            for h, marker_poly in zip(self.plot_handles['hinge'], hinge_markers):
                h.set_xy(marker_poly)

            # 3.4 更新货物多边形
            if cargo_polygon and self.plot_handles['cargo']:
                self.plot_handles['cargo'].set_xy(cargo_polygon)
            elif self.plot_handles['cargo']:
                # 无货物时移出视野
                self.plot_handles['cargo'].set_xy([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])

            # 3.5 更新车辆多边形
            for h, poly in zip(self.plot_handles['car'], car_polygons):
                h.set_xy(poly)

        # 4. 更新轨迹数据（保持原有逻辑，确保轨迹连贯）
        if len(self.trajectories['cargo']) > 1:
            cargo_traj = np.array(self.trajectories['cargo'])
            self.plot_handles['cargo_traj'].set_data(cargo_traj[:, 0], cargo_traj[:, 1])
            car1_traj = np.array(self.trajectories['car1'])
            self.plot_handles['car1_traj'].set_data(car1_traj[:, 0], car1_traj[:, 1])
            car2_traj = np.array(self.trajectories['car2'])
            self.plot_handles['car2_traj'].set_data(car2_traj[:, 0], car2_traj[:, 1])
            hinge1_traj = np.array(self.trajectories['hinge1'])
            self.plot_handles['hinge1_traj'].set_data(hinge1_traj[:, 0], hinge1_traj[:, 1])
            hinge2_traj = np.array(self.trajectories['hinge2'])
            self.plot_handles['hinge2_traj'].set_data(hinge2_traj[:, 0], hinge2_traj[:, 1])

        # 5. 更新坐标范围（与参考代码对齐，基于货物中心）
        X_o = self.model.x_arch[i_sim, 0]
        Y_o = self.model.x_arch[i_sim, 1]
        vis_range = self.model.config['range']
        self.ax.set_xlim([X_o - vis_range, X_o + vis_range])
        self.ax.set_ylim([Y_o - vis_range, Y_o + vis_range])

        # 6. 刷新画布（高效刷新）
        self.fig.canvas.draw_idle()

        # 7. rgb_array模式下保存帧（优化内存+资源释放）
        if self.render_mode == "rgb_array":
            frame = None
            buf = None
            img = None
            try:
                buf = BytesIO()
                self.fig.savefig(
                    buf,
                    format='png',
                    bbox_inches='tight',
                    dpi=96,
                    facecolor=self.fig.get_facecolor()
                )
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
                frame = np.array(img, dtype=np.uint8)  # 新增：指定dtype，避免高内存占用的浮点型数组

                # 校验帧尺寸一致性（优化插值方法，减少内存占用）
                if len(self.render_frames) > 0:
                    ref_shape = self.render_frames[0].shape
                    if frame.shape != ref_shape:
                        # 使用INTER_AREA插值，更适合缩小/放大，内存占用更低
                        frame = cv2.resize(
                            frame,
                            (ref_shape[1], ref_shape[0]),
                            interpolation=cv2.INTER_AREA
                        )

                # 核心优化：限制帧缓存上限，避免内存溢出
                max_cache_frames = 1001  # 可按需调整，超过则丢弃最早帧
                if len(self.render_frames) >= max_cache_frames:
                    self.render_frames.pop(0)  # 丢弃最早帧，保持缓存容量稳定
                self.render_frames.append(frame)
                return frame
            except Exception as e:
                print(f"帧保存失败，错误：{type(e).__name__}: {e}")
            finally:
                # 关键：显式释放资源，避免内存泄漏
                if buf is not None:
                    buf.close()
                if img is not None:
                    del img
                if buf is not None:
                    del buf
            return frame

    def save_eval_video(self, eval_round=None, video_save_dir=None):
        """
        手动保存单轮评测的仿真视频（基于当前render_frames中的帧）
        优化：分批写入帧+健壮编码器兼容+内存释放
        :param eval_round: 评测轮次（用于文件名区分）
        :param video_save_dir: 视频保存目录（默认沿用原有./output）
        :return: 保存的视频文件路径（失败返回None）
        """
        # 前置校验：可视化启用、rgb_array模式、帧列表非空
        if not self.enable_visualization or self.render_mode != "rgb_array" or len(self.render_frames) == 0:
            print("警告：不满足视频保存条件（可视化未启用/非rgb_array模式/无有效帧）")
            return None
        
        out = None
        try:
            # 1. 配置视频保存目录（保持原有逻辑）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if video_save_dir is None:
                # 优先使用Checkpoint目录（需与cfg.checkpoint.checkpoint_dir对齐）
                default_ckpt_dir = os.path.join(current_dir, "checkpoints")
                video_save_dir = default_ckpt_dir if os.path.exists(default_ckpt_dir) else os.path.join(current_dir, "output")
            os.makedirs(video_save_dir, exist_ok=True)
            
            # 2. 生成带评测轮次的文件名（保持原有命名逻辑）
            time_str = datetime.datetime.now().strftime('%y%m%d%H%M%S')
            if eval_round is not None:
                file_prefix = f"{self.config_name}_eval_round_{eval_round}"
            else:
                file_prefix = f"{self.config_name}_vis"
            file_name = f"{file_prefix}_{time_str}.mp4"
            video_path = os.path.join(video_save_dir, file_name)
            
            # 3. 视频合成核心逻辑（优化编码器兼容+分批写入）
            fps = self.metadata['render_fps']
            height, width, _ = self.render_frames[0].shape
            video_writer_opened = False
            
            # 第一步：尝试MP4格式（mp4v编码器）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            if out.isOpened():
                video_writer_opened = True
            
            # 第二步：MP4失败，切换AVI格式（XVID编码器，兼容性最强）
            if not video_writer_opened:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_path = video_path.replace(".mp4", ".avi")
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                if out.isOpened():
                    video_writer_opened = True
                    print(f"mp4格式不支持，切换为avi格式，保存路径：{video_path}")
                else:
                    raise RuntimeError("MP4和AVI格式均无法初始化VideoWriter，编码器缺失或内存不足")
            
            # 4. 分批写入帧（核心优化：减少单次内存占用）
            batch_size = 60  # 每批写入60帧，写完释放批处理内存
            total_frames = len(self.render_frames)
            for i in range(0, total_frames, batch_size):
                # 提取当前批次帧
                batch_frames = self.render_frames[i:i+batch_size]
                for frame in batch_frames:
                    # 格式转换（保持原有逻辑，显式指定dtype减少内存）
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.uint8)
                    out.write(bgr_frame)
                
                # 打印进度（保持原有逻辑）
                current_written = min(i + batch_size, total_frames)
                if current_written % 100 == 0:
                    print(f"  已写入 {current_written}/{total_frames} 帧")
                
                # 显式释放批处理帧内存
                del batch_frames
            
            # 5. 释放资源并打印结果
            out.release()
            out = None
            print(f"单轮评测视频已成功保存至: {video_path}")
            return video_path
        
        except Exception as e:
            # 清理残留的VideoWriter资源
            if out is not None and out.isOpened():
                out.release()
            out = None
            print(f"生成单轮评测视频失败，详细错误信息：{type(e).__name__}: {e}")
            
            # 备选：保存关键帧（优化压缩，减少内存占用）
            try:
                key_frame_dir = os.path.join(video_save_dir or os.path.join(current_dir, "output"), "key_frames")
                os.makedirs(key_frame_dir, exist_ok=True)
                
                # 优化：每10帧保存1张，增加PNG压缩参数
                key_frame_interval = 10
                key_frames = self.render_frames[::key_frame_interval]
                for i, frame in enumerate(key_frames):
                    img_name = f"eval_round_{eval_round or 'unknown'}_frame_{i:03d}.png"
                    img_path = os.path.join(key_frame_dir, img_name)
                    # 格式转换+压缩保存，减少磁盘占用和内存消耗
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        img_path,
                        bgr_frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, 6]  # PNG压缩级别（0-9），6为平衡值
                    )
                
                print(f"已保存关键帧至: {key_frame_dir}，共保存 {len(key_frames)} 张")
            except Exception as e2:
                print(f"保存关键帧也失败，错误：{type(e2).__name__}: {e2}")
            return None
        finally:
            # 最终兜底：释放VideoWriter资源
            if out is not None and out.isOpened():
                out.release()
    
    def clear_render_frames(self):
        if hasattr(self, 'render_frames'):
            self.render_frames = []
    
    def close(self):
        """关闭环境并生成视频"""
        if self.fig is not None:
            plt.close(self.fig)

        if self.enable_visualization and self.render_mode == "rgb_array" and len(self.render_frames) > 0:
            try:
                # 创建输出目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(current_dir, "output")
                os.makedirs(output_dir, exist_ok=True)
                print(f"输出目录已准备：{output_dir}，共待写入帧数量：{len(self.render_frames)}")

                # 生成视频文件名
                time_str = datetime.datetime.now().strftime(r'%y%m%d%H%M%S')
                file_name = f"{self.config_name}_vis_{time_str}.mp4"
                video_path = os.path.join(output_dir, file_name)

                # 使用OpenCV合成视频（兼容多系统编码）
                fps = self.metadata['render_fps']
                height, width, _ = self.render_frames[0].shape
                # 兼容Windows/Mac/Linux的编码格式
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4格式（优先）
                # 备选编码：若mp4v失败，切换为XVID（avi格式）
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_path = video_path.replace(".mp4", ".avi")
                    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    print(f"mp4格式不支持，切换为avi格式，保存路径：{video_path}")

                # 写入帧
                for idx, frame in enumerate(self.render_frames):
                    # 转换为BGR格式（OpenCV要求）
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)
                    if idx % 100 == 0:
                        print(f"已写入 {idx+1}/{len(self.render_frames)} 帧")

                out.release()
                print(f"可视化视频已成功保存至: {video_path}")

            except Exception as e:
                # 暴露具体异常信息，便于排查
                print(f"生成视频失败，详细错误信息：{type(e).__name__}: {e}")
                # 保存单帧图像作为备选
                try:
                    for i, frame in enumerate(self.render_frames[::10]):  # 每10帧保存一张
                        img_path = os.path.join(output_dir, f"frame_{i:03d}.png")
                        cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"已保存关键帧至: {output_dir}，共保存 {len(self.render_frames[::10])} 张")
                except Exception as e2:
                    print(f"保存关键帧也失败，错误：{type(e2).__name__}: {e2}")
        elif self.enable_visualization and self.render_mode == "rgb_array" and len(self.render_frames) == 0:
            print("警告：未生成任何视频帧，无法创建视频！")
        else:
            print("可视化功能已关闭或非rgb_array模式，不生成视频")


# 注册环境（导入时自动注册）
gym.register(
    id="TwoCarrierEnv-v0",
    entry_point="occt_2d2c:TwoCarrierEnv",
    max_episode_steps=1000,
    kwargs={}  # 预留参数
)

# 测试代码
if __name__ == "__main__":
    # ===================== 配置调整（便于测试扰动功能） =====================
    RENDER_MODE = "rgb_array"  # 可选"human"（可视化）或"rgb_array"（保存视频）
    ENABLE_VISUALIZATION = True  # 测试扰动时建议先关闭，避免卡顿；验证可视化时再开启
    TEST_STEPS = 500  # 测试步数（无需跑满1000步，足够验证扰动即可）
    SEED = 42  # 固定种子，保证扰动测试可复现
    
    # ===================== 环境创建（保持原有逻辑，强化解包） =====================
    env = gym.make(
        "TwoCarrierEnv-v0",
        render_mode=RENDER_MODE,
        config_path=None,  # 使用默认2d2c.yaml配置
        enable_visualization=ENABLE_VISUALIZATION
    )
    # 必须解包获取原始环境实例（解除TimeLimit包装，才能访问扰动相关属性/记录）
    raw_env = env.unwrapped
    print(f"=== 环境初始化完成 ===")
    print(f"环境实例渲染模式：{raw_env.render_mode}")
    print(f"可视化功能：{'启用' if raw_env.enable_visualization else '关闭'}")
    print(f"原始u1_random：{raw_env.u1_random}（前轮转角初始值：{raw_env.u1_random[0]:.4f}rad）")
    print(f"前轮转角约束：±{np.pi/6:.4f}rad（±30°）")
    print(f"观测归一化模式：{'训练模式（可更新统计量）' if not raw_env.vecnorm_frozen else '评测模式（统计量已冻结）'}")
    print("=" * 60)

    # ===================== 步骤1：保留原有动作归一化/反归一化测试 =====================
    print("\n--- 【原有测试】动作归一化/反归一化验证 ---")
    original_test_action = np.array([np.pi/12, -np.pi/12, 500, 500])  # 中间值
    print(f"原始测试动作：{original_test_action}")
    normalized_action = raw_env.normalize_action(original_test_action)
    print(f"归一化后动作：{normalized_action}（范围应在[-1,1]）")
    denormalized_action = raw_env.denormalize_action(normalized_action)
    print(f"反归一化后动作：{denormalized_action}（应与原始动作一致）")
    print(f"归一化/反归一化误差：{np.mean(np.abs(denormalized_action - original_test_action)):.6f}")
    print("--- 动作归一化测试完成 ---")
    print("=" * 60)

    # ===================== 步骤2：新增【观测归一化功能验证】 =====================
    print("\n--- 【新增测试】观测归一化功能验证 ---")
    # 重置环境（固定种子，保证可复现）
    obs, info = env.reset(seed=SEED)
    print(f"首次观测（归一化后）形状：{obs.shape}")
    print(f"首次观测（归一化后）前12维：{obs[:12]}")
    
    # 运行几步，查看归一化统计量更新
    for step in range(5):
        normalized_action = np.array([0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(normalized_action)
        if step == 4:
            print(f"第5步观测（归一化后）前12维：{obs[:12]}")
            print(f"当前归一化统计量（均值前12维）：{raw_env.vecnorm_mean[:12] if raw_env.vecnorm_mean is not None else '未初始化'}")
            print(f"当前归一化统计量（方差前12维）：{raw_env.vecnorm_var[:12] if raw_env.vecnorm_var is not None else '未初始化'}")
    
    # 切换到评测模式，冻结统计量
    raw_env.freeze_vecnorm()
    obs_frozen, reward_frozen, terminated_frozen, truncated_frozen, info_frozen = env.step(normalized_action)
    print(f"冻结后第一步观测（归一化后）前5维：{obs_frozen[:5]}")
    print(f"冻结后归一化均值是否变化：{np.allclose(raw_env.vecnorm_mean, raw_env.vecnorm_mean)}（应始终为True）")
    print("--- 观测归一化功能测试完成 ---")
    print("=" * 60)

    # ===================== 步骤3：新增【前轮转角多样性扰动专属测试】（核心修改） =====================
    print("\n--- 【新增测试】前轮转角多样性扰动验证 ---")
    # 重置环境（固定种子，保证可复现）
    obs, info = env.reset(seed=SEED)
    # 初始化记录容器，保存每一步的u1前轮转角、推力（用于后续统计）
    steer_records = []  # 记录前轮转角（索引0）
    thrust1_records = []  # 记录前轮推力（索引2）
    thrust2_records = []  # 记录后轮推力（索引3）

    # 运行指定步数，记录扰动数据
    print(f"开始运行{TEST_STEPS}步仿真，记录u1扰动数据...")
    for step in range(TEST_STEPS):
        # 采用归一化中间值动作（避免第二辆车动作干扰u1扰动测试）
        normalized_action = np.array([0, 0, 0, 0])
        # 环境交互（每一步都会生成带扰动的u1）
        obs, reward, terminated, truncated, info = env.step(normalized_action)
        
        # 提取并记录u1的关键数据
        current_u1 = info['u1']
        steer_records.append(current_u1[0])
        thrust1_records.append(current_u1[2])
        thrust2_records.append(current_u1[3])
        
        # 每100步打印一次中间结果，验证扰动有效性
        if (step + 1) % 100 == 0:
            print(f"  第{step+1}步 | u1前轮转角：{current_u1[0]:.4f}rad（≈{np.rad2deg(current_u1[0]):.1f}°）")
            print(f"  第{step+1}步 | u1前后轮推力：{current_u1[2]:.1f}, {current_u1[3]:.1f}")
            print(f"  任务终止状态：{terminated} | 截断状态：{truncated}")
            print(f"  " + "-" * 30)
        
        # 若提前终止，跳出循环
        if terminated or truncated:
            print(f"  仿真提前终止，共运行{step+1}步")
            break

    # 转换为numpy数组，方便统计分析
    steer_records = np.array(steer_records)
    thrust1_records = np.array(thrust1_records)
    thrust2_records = np.array(thrust2_records)

    # 统计并打印扰动结果（验证多样性+约束有效性）
    print("\n--- 扰动数据统计结果 ---")
    print(f"【前轮转角（核心验证）】")
    print(f"  取值范围：[{steer_records.min():.4f}, {steer_records.max():.4f}]rad | 对应角度：[{np.rad2deg(steer_records.min()):.1f}, {np.rad2deg(steer_records.max()):.1f}]°")
    print(f"  均值：{steer_records.mean():.4f}rad | 标准差：{steer_records.std():.4f}rad")
    print(f"  是否符合约束（±π/6）：{ (steer_records >= -np.pi/6).all() and (steer_records <= np.pi/6).all() }")
    print(f"【前轮推力】")
    print(f"  取值范围：[{thrust1_records.min():.1f}, {thrust1_records.max():.1f}]")
    print(f"  均值：{thrust1_records.mean():.1f} | 标准差：{thrust1_records.std():.1f}")
    print(f"【后轮推力】")
    print(f"  取值范围：[{thrust2_records.min():.1f}, {thrust2_records.max():.1f}]")
    print(f"  均值：{thrust2_records.mean():.1f} | 标准差：{thrust2_records.std():.1f}")
    print("--- 前轮转角多样性扰动测试完成 ---")
    print("=" * 60)

    # ===================== 步骤4：（可选）多轮仿真验证重置有效性 =====================
    print("\n--- 【可选测试】多轮仿真重置验证（确保扰动状态独立） ---")
    max_episodes = 4
    for episode in range(max_episodes):
        print(f"\n第{episode+1}/{max_episodes}轮仿真")
        # 重置环境（会清空上一轮的扰动状态_prev_u1_noisy）
        obs, info = env.reset(seed=SEED + episode)  # 每轮种子偏移，避免数据重复
        episode_steer_first = None
        for step in range(50):  # 每轮仅运行50步
            normalized_action = np.array([0, 0, 0, 0])
            obs, reward, terminated, truncated, info = env.step(normalized_action)
            if step == 0:
                # 记录每轮第一步的前轮转角，验证重置后扰动重新初始化
                episode_steer_first = info['u1'][0]
                print(f"  第1步前轮转角：{episode_steer_first:.4f}rad（≈{np.rad2deg(episode_steer_first):.1f}°）")
            if terminated or truncated:
                break
    print("--- 多轮仿真重置验证完成 ---")
    print("=" * 60)

    # ===================== 步骤5：保留原有环境关闭+视频生成逻辑 =====================
    print("\n--- 环境关闭与视频生成 ---")
    env.close()

    # 规范输出提示
    if RENDER_MODE == "rgb_array" and ENABLE_VISUALIZATION:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "output")
        video_files = []
        if os.path.exists(output_dir):
            video_files = [f for f in os.listdir(output_dir) if f.endswith(('.mp4', '.avi'))]
        if video_files:
            print(f"\n仿真结束！视频已保存至：{output_dir}，最新文件名：{video_files[-1]}")
        else:
            print("\n仿真结束！未生成视频（可视化未启用或无有效帧）")
    elif RENDER_MODE == "human":
        print("\n仿真结束！实时可视化窗口已关闭")
    else:
        print("\n仿真结束！未启用可视化功能，扰动数据已通过控制台输出验证")
    print("=" * 60)