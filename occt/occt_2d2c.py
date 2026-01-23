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
from scipy.interpolate import CubicSpline  #用于生成平滑曲线
import math

# plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False

class TwoCarrierEnv(gym.Env):
    """两辆车运载超大件系统的自定义强化学习环境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode=None, config_path=None, enable_visualization=False, 
                 vecnorm_frozen: bool=False, vecnorm_mean=None, vecnorm_var=None, shared_w_force=None):
        super().__init__()
        
        # 加载2d2c.yaml配置文件
        self.config_name = '2d2c'
        self.config = self._load_config(config_path)

        self.rng = np.random.default_rng()
        self.model = Model2D2C(self.config)
        
        # =========================================================================================
        # [Paper Adaptation] 状态空间重构：完全局部化 (Fully Localized Observation)
        # 对应论文中的 Vehicle Coordinate System 转换 [cite: 140]
        # 维度映射 (12维):
        # [0]  Local_X_Cargo : 货物中心相对于后车的局部X坐标 (纵向距离)
        # [1]  Local_Y_Cargo : 货物中心相对于后车的局部Y坐标 (横向偏差)
        # [2]  V_Body_X      : 后车在自身车身坐标系下的纵向速度
        # [3]  V_Body_Y      : 后车在自身车身坐标系下的侧滑速度
        # [4]  Psi_o_2       : 相对角度 (货物 - 后车)
        # [5]  Psi_1_o       : 相对角度 (前车 - 货物)
        # [6]  Psi_dot_2     : 后车自身角速度 (Yaw Rate)
        # [7]  Psi_dot_o_2   : 相对角速度 (货物 - 后车)
        # [8]  Psi_dot_1_o   : 相对角速度 (前车 - 货物)
        # [9]  Fh_Long       : 铰接力在后车坐标系下的纵向分量
        # [10] Fh_Lat        : 铰接力在后车坐标系下的横向分量
        # [11] Placeholder   : 预留位 (如前车距离等，目前为0.0)
        # =========================================================================================
        
        # 定义物理边界（主要用于参考，VecNorm会处理实际数值范围）
        obs_low = np.array([
            -np.inf, -np.inf,       # Local Pos
            -20, -20,               # Body Vel
            -np.pi, -np.pi,         # Relative Angle
            -5, -10, -10,           # Omega / Relative Omega
            -1e5, -1e5,             # Body Force
            -1.0                    # Placeholder
        ])
        obs_high = np.array([
            np.inf, np.inf,
            20, 20,
            np.pi, np.pi,
            5, 10, 10,
            1e5, 1e5,
            1.0
        ])
        
        # 归一化后的观测空间
        obs_norm_low = np.full(12, -1000.0, dtype=np.float64)
        obs_norm_high = np.full(12, 1000.0, dtype=np.float64)
        self.observation_space = spaces.Box(
            low=obs_norm_low, high=obs_norm_high, dtype=np.float64
        )
        
        # 动作空间保持不变
        self.original_action_low = np.array([-np.pi/6, -np.pi/6, 0, 0])
        self.original_action_high = np.array([np.pi/6, np.pi/6, 1e3, 1e3])
        self.action_space = spaces.Box(
            low=-np.ones(4, dtype=np.float64),
            high=np.ones(4, dtype=np.float64),
            dtype=np.float64
        )
        
        self.u1_random = np.array([0, 0, 1e3, 1e3])
        self.steer_episode_base_std = np.pi/15
        self.steer_step_dynamic_std = 0.008
        self.steer_max_bound = np.pi/6
        self.steer_min_bound = -np.pi/6
        self.steer_episode_offset = 0.0
        self.thrust_noise_rel_std = 0.02 
        self.thrust_noise_abs_min = 0  
        self.thrust_noise_abs_max = 1e3 
        
        self.enable_visualization = enable_visualization
        self.render_mode = render_mode if enable_visualization else None
        self.render_frames = []
        self.trajectories = {
            'cargo': [], 'car1': [], 'car2': [], 'hinge1': [], 'hinge2': []
        }
        self.fig = None
        self.ax = None
        self.is_sim_finished = False
        
        # VecNorm 初始化
        self.vecnorm_decay = 0.99999
        self.vecnorm_eps = 1e-2
        self.vecnorm_frozen = vecnorm_frozen
        self.vecnorm_min_var = 1e-4
        self.vecnorm_count = 0
        
        if vecnorm_mean is not None and vecnorm_var is not None:
            self.vecnorm_mean = np.array(vecnorm_mean, dtype=np.float64)
            self.vecnorm_var = np.array(vecnorm_var, dtype=np.float64)
            self.vecnorm_var = np.maximum(self.vecnorm_var, self.vecnorm_min_var)
            self.vecnorm_frozen = True
            print(f"【TwoCarrierEnv】已加载固定归一化统计量，VecNorm 状态已冻结。")
        else:
            self.vecnorm_frozen = vecnorm_frozen
            self.vecnorm_mean = np.zeros(12, dtype=np.float64) 
            self.vecnorm_var = np.ones(12, dtype=np.float64) * self.vecnorm_min_var
        
        self.hinge_force_penalty = 0.0
        self.control_smooth_penalty = 0.0
        # 保存共享变量
        self.shared_w_force = shared_w_force
        # 默认初始权重 (如果没传共享变量，就用这个默认值)
        self.default_w_force = 0.005

    def _load_config(self, config_path):
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "2d2c.yaml")
        if not os.path.exists(config_path):
            return self._get_default_config()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        except Exception:
            return self._get_default_config()

    def _get_default_config(self):
        return {
            'N_c': 2, 'N_q': 5, 'N_x': 10, 'N_u': 8,
            'M_o': 1000, 'I_o': 1000,
            'M_1': 500, 'M_2': 500,
            'I_1': 100, 'I_2': 100,
            'x__o_1': 5, 'x__o_2': -5,
            'y__o_1': 0, 'y__o_2': 0,
            'x__1_1': 1, 'x__2_2': 1,
            'y__1_1': 0, 'y__2_2': 0,
            'C_f': 10000, 'C_r': 10000,
            'l_f': 2, 'l_r': 2,
            'X_o_0': 0, 'Y_o_0': 0, 'Psi_o_0': 0,
            'Psi_1_0': 0, 'Psi_2_0': 0,
            'X_dot_o_0': 0, 'Y_dot_o_0': 0, 'Psi_dot_o_0': 0,
            'Psi_dot_1_0': 0, 'Psi_dot_2_0': 0,
            'T': 10, 'dt': 0.1, 'integrator': 'RK4',
            'framerate': 10, 'range': 20,
            'oversized_cargo_bias': 2, 'oversized_cargo_width': 3
        }
    
    def _generate_random_spline_path(self):
        """
        [New Feature] 生成随机样条曲线路径供前车跟踪
        生成的路径保存在 self.spline_func (函数) 和 self.path_x/y (可视化数据)
        """
        # 1. 确定起点 (前车初始位置)
        # 注意：这里假设 Config 中的 X_o_0 是货物中心，我们需要估算前车大致位置
        # 根据 config: x_front ≈ X_o_0 + x__o_1
        start_x = self.config.get('X_o_0', 0) + self.config.get('x__o_1', 5)
        start_y = self.config.get('Y_o_0', 0) # 假设初始 Y 为 0

        # 2. 定义路径参数
        total_length = 30.0  # 路径总长 (覆盖 max_episode_steps * speed)
        num_points = 4        # 控制点数量 (控制点越多，弯道越复杂)
        
        # 3. 生成控制点 (Control Points)
        # X轴：均匀分布
        key_x = np.linspace(start_x, start_x + total_length, num_points)
        
        # Y轴：起点固定，后续点随机扰动
        key_y = [start_y]
        
        # 第2个点波动小一点，保证起步平稳
        key_y.append(start_y + self.rng.uniform(-2, 2))
        
        # 后续点产生较大波动 (模拟变道或S弯)
        for _ in range(num_points - 2):
            last_y = key_y[-1]
            # 随机偏移，但限制在视野范围内 (-30, 30)
            next_y = last_y + self.rng.uniform(-10, 10) 
            next_y = np.clip(next_y, -25, 25)
            key_y.append(next_y)
            
        key_y = np.array(key_y)
        
        # 4. 生成三次样条函数 y = f(x)
        self.spline_func = CubicSpline(key_x, key_y)
        
        # 5. 生成可视化用的离散点
        self.path_x = np.linspace(start_x, start_x + total_length, 500)
        self.path_y = self.spline_func(self.path_x)

    def _get_spline_tracking_u1(self):
        """
        [New Feature] 前车跟踪控制器
        计算前车为了跟踪样条曲线所需的控制量 u1
        """
        # 1. 获取前车当前状态
        # 状态索引依赖 model 定义，通常: x[0-2]cargo, x[3]front_psi, x[4]rear_psi
        # 我们使用 getXYi 获取前车中心坐标
        x_state = self.model.x
        X_front, Y_front = self.model.getXYi(x_state, 0) # idx 0 是前车
        Psi_front = x_state[3] # 前车航向角
        
        # 2. 预瞄 (Lookahead) 机制
        # 不看当前位置，看前方一点点，这样走线更顺滑
        lookahead_dist = 2.0 
        target_x = X_front + lookahead_dist
        
        # 3. 计算目标状态
        # 利用样条函数计算目标Y和目标斜率
        target_y = self.spline_func(target_x)
        target_dy_dx = self.spline_func(target_x, 1) # 求一阶导数
        
        # 目标航向角 (Desired Heading)
        target_psi = np.arctan(target_dy_dx)
        
        # 4. 计算误差
        # 横向误差 (Lateral Error): 当前Y 与 对应X处的路径Y 的差
        current_y_ref = self.spline_func(X_front)
        lat_error = Y_front - current_y_ref
        
        # 航向误差 (Heading Error)
        heading_error = self._normalize_angle(Psi_front - target_psi)
        
        # 5. PD 控制律
        # k_lat: 纠正偏离路线的力度
        # k_head: 纠正车头朝向的力度
        k_lat = 0.3   # 如果车摆动太大，减小这个值
        k_head = 1.5  # 主要靠这个跟踪方向
        
        # 计算前轮转角 (Steering Angle)
        # 注意符号：如果偏左(y大)，lat_error>0，需要负转角(向右)，所以是负号
        steer_cmd = -k_lat * lat_error - k_head * heading_error
        
        # 6. 约束与输出
        steer_cmd = np.clip(steer_cmd, self.steer_min_bound, self.steer_max_bound)
        
        # 组装 u1: [Steer, ?, Thrust, ?]
        # 保持推力恒定或稍微随机，主要改变转向
        u1 = np.copy(self.u1_random)
        u1[0] = steer_cmd # 覆盖转向角
        
        # 可选：根据转向角度适当减速 (模拟真实驾驶)
        if np.abs(steer_cmd) > 0.2:
             u1[2] *= 0.9 # 推力减小
             u1[3] *= 0.9
             
        return u1.astype(np.float64)

    def normalize_action(self, original_action):
        orig_range = self.original_action_high - self.original_action_low
        orig_range = np.where(orig_range == 0, 1e-8, orig_range)
        norm_action = 2 * (original_action - self.original_action_low) / orig_range - 1
        return np.clip(norm_action, -1, 1).astype(np.float64)

    def denormalize_action(self, normalized_action):
        orig_range = self.original_action_high - self.original_action_low
        orig_action = self.original_action_low + (normalized_action + 1) * orig_range / 2
        return np.clip(orig_action, self.original_action_low, self.original_action_high).astype(np.float64)

    def _update_vecnorm_stats(self, obs):
        if self.vecnorm_frozen:
            return
        obs_np = np.asarray(obs, dtype=np.float64)
        if self.vecnorm_mean is None:
            self.vecnorm_mean = np.zeros_like(obs_np, dtype=np.float64)
            self.vecnorm_var = np.ones_like(obs_np, dtype=np.float64)
        
        current_mean = obs_np
        current_var = np.square(obs_np)
        current_count = 1

        self.vecnorm_count += current_count
        if self.vecnorm_count <= current_count:
            self.vecnorm_mean = current_mean
            self.vecnorm_var = current_var
        else:
            decay = self.vecnorm_decay
            self.vecnorm_mean = decay * self.vecnorm_mean + (1 - decay) * current_mean
            self.vecnorm_var = decay * self.vecnorm_var + (1 - decay) * current_var
        self.vecnorm_var = np.maximum(self.vecnorm_var, self.vecnorm_min_var)

    def _normalize_observation(self, obs):
        if self.vecnorm_mean is None:
            return obs
        obs_np = np.asarray(obs, dtype=np.float64)
        std_np = np.sqrt(self.vecnorm_var) + self.vecnorm_eps
        normalized_obs_np = (obs_np - self.vecnorm_mean) / std_np
        return normalized_obs_np.astype(obs.dtype) if hasattr(obs, 'dtype') else normalized_obs_np

    def _transform_to_local(self, x_target, y_target, x_self, y_self, psi_self):
        """
        Paper Eq (5)-(6): Global to Local Transformation
        """
        dx = x_target - x_self
        dy = y_target - y_self
        # 旋转矩阵 R^T
        x_local = dx * np.cos(psi_self) + dy * np.sin(psi_self)
        y_local = -dx * np.sin(psi_self) + dy * np.cos(psi_self)
        return x_local, y_local
    
    def _get_observation(self):
        """
        Paper Implementation: Fully Localized Observation
        """
        x = self.model.x
        i_sim = self.model.count
        idx_rear = 1 
        
        # 1. 后车全局状态
        X_2, Y_2 = self.model.getXYi(x, idx_rear)
        Psi_2 = x[4]
        X_dot_2, Y_dot_2 = self.model.getXYdoti(x, idx_rear)
        Psi_dot_2 = x[9]
        
        # 2. 目标点(货物)全局状态
        X_cargo = x[0]
        Y_cargo = x[1]
        
        # --- 核心：坐标转换 ---
        # (1) 位置局部化
        local_x_cargo, local_y_cargo = self._transform_to_local(
            X_cargo, Y_cargo, X_2, Y_2, Psi_2
        )
        
        # (2) 速度局部化 (Body Frame Velocity)
        vx_body = X_dot_2 * np.cos(Psi_2) + Y_dot_2 * np.sin(Psi_2)
        vy_body = -X_dot_2 * np.sin(Psi_2) + Y_dot_2 * np.cos(Psi_2)

        # 3. 相对状态
        Psi_o = x[2]
        Psi_1 = x[3]
        Psi_o_2 = self._normalize_angle(Psi_o - Psi_2)
        Psi_1_o = self._normalize_angle(Psi_1 - Psi_o)
        
        Psi_dot_o = x[7]
        Psi_dot_1 = x[8]
        Psi_dot_o_2 = Psi_dot_o - Psi_dot_2
        Psi_dot_1_o = Psi_dot_1 - Psi_dot_o
        
        # 4. 铰接力局部化
        Fh2_x = self.model.Fh_arch[i_sim, 2]
        Fh2_y = self.model.Fh_arch[i_sim, 3]
        Fh_longitudinal = Fh2_x * np.cos(Psi_2) + Fh2_y * np.sin(Psi_2)
        Fh_lateral      = -Fh2_x * np.sin(Psi_2) + Fh2_y * np.cos(Psi_2)

        # 组装 12维 向量
        raw_obs = np.array([
            local_x_cargo, local_y_cargo, # [0-1] 位置 (局部)
            vx_body, vy_body,             # [2-3] 速度 (局部)
            Psi_o_2, Psi_1_o,             # [4-5] 角度 (相对)
            Psi_dot_2,                    # [6]   角速度
            Psi_dot_o_2, Psi_dot_1_o,     # [7-8] 角速度 (相对)
            Fh_longitudinal, Fh_lateral,  # [9-10] 力 (局部)
            0.0                           # [11] Placeholder
        ], dtype=np.float64)
        
        self._update_vecnorm_stats(raw_obs)
        return self._normalize_observation(raw_obs)
    
    def _calculate_reward(self):
        """
        最终修正版 Reward：双重门控机制 (Double Gating)
        解决：后车自己摆正了，却把前车推折叠了的问题。
        """
        x = self.model.x
        i_sim = self.model.count
        
        # --- 1. 获取状态 ---
        Fh2_x = self.model.Fh_arch[i_sim, 2]
        Fh2_y = self.model.Fh_arch[i_sim, 3]
        F_force_mag = np.hypot(Fh2_x, Fh2_y)
        F_safe = self.config.get('force_safe', 2000.0) 

        Psi_cargo = x[2]      # 货物航向
        Psi_front = x[3]      # 前车航向
        Psi_rear = x[4]       # 后车航向
        
        # --- 2. 计算两个关键夹角 ---
        # (1) 后车与货物（Agent 自己的姿态）
        delta_psi_rear = self._normalize_angle(Psi_rear - Psi_cargo)
        
        # (2) 前车与货物（系统的健康状况 - 你指出的那个大角）
        delta_psi_front = self._normalize_angle(Psi_front - Psi_cargo)

        # --- 3. Progress Reward (盲从协同) ---
        # 计算投影速度（同上）
        X_cargo, Y_cargo = x[0], x[1]
        X_front, Y_front = self.model.getXYi(x, 0)
        vec_fc_x = X_front - X_cargo
        vec_fc_y = Y_front - Y_cargo
        dist_fc = np.hypot(vec_fc_x, vec_fc_y) + 1e-6
        dir_x = vec_fc_x / dist_fc
        dir_y = vec_fc_y / dist_fc
        
        X_dot_cargo = x[self.config['N_q'] + 0]
        Y_dot_cargo = x[self.config['N_q'] + 1]
        v_effective = X_dot_cargo * dir_x + Y_dot_cargo * dir_y
        
        target_speed = 1.0
        r_progress = np.clip(v_effective, -0.5, target_speed)

        # =================================================================
        # 【核心修改】双重门控机制 (Double Gating)
        # =================================================================
        # 阈值设定：45度 (约0.8弧度)
        is_rear_folded = np.abs(delta_psi_rear) > 0.8
        is_front_folded = np.abs(delta_psi_front) > 0.8  # <--- 加入了你的观察
        
        if is_rear_folded or is_front_folded:
            # 只要任意一端折叠，不仅没收进度分，还要倒扣分！
            # 迫使 Agent 在看到前车折叠时，必须减速或停车
            r_progress = -2.0 
        
        # --- 4. Alignment Penalty (同时惩罚两端) ---
        # 后车不正，扣分（为了传力效率）
        r_align_rear = -1.0 * np.abs(delta_psi_rear)
        
        # 前车不正，也要扣后车的分（为了系统安全）
        # 告诉 Agent：前车歪了也是你的责任（因为是你推的）
        r_align_front = -1.0 * np.abs(delta_psi_front)

        # --- 5. Force Penalty (指数级) ---
        force_ratio = F_force_mag / F_safe
        r_force = -1.0 * (force_ratio ** 2)
        if force_ratio > 0.8:
            r_force -= 10.0 * (force_ratio - 0.8)

        # --- 6. Stability ---
        Psi_dot_rear = x[self.config['N_q'] + 4]
        r_stability = -1.0 * np.square(Psi_dot_rear)

        # --- 7. 权重配置 ---
        w_progress = 10.0
        if self.shared_w_force is not None:
             w_force = self.shared_w_force.value
        else:
             w_force = 50.0
        
        w_align = 20.0
        w_stability = 2.0
        
        # 总分包含 front 和 rear 的惩罚
        total_reward = (w_progress * r_progress) + \
                       (w_force * r_force) + \
                       (w_align * (r_align_rear + r_align_front)) + \
                       (w_stability * r_stability) + \
                       0.05

        self.reward_info = {
            "r_force": r_force * w_force,
            "r_align_rear": r_align_rear * w_align,
            "r_align_front": r_align_front * w_align,
            "r_progress": r_progress * w_progress,
            "r_stability": r_stability * w_stability,
            "val_force": F_force_mag,
            "val_delta_psi_rear": delta_psi_rear,
            "val_delta_psi_front": delta_psi_front
        }
        
        return total_reward

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _get_noisy_u1(self):
        u1_noisy = np.copy(self.u1_random)
        steer_step_noise = self.rng.normal(loc=0, scale=self.steer_step_dynamic_std)
        steer_candidate = u1_noisy[0] + self.steer_episode_offset + steer_step_noise
        steer_clipped = np.clip(steer_candidate, self.steer_min_bound, self.steer_max_bound)
        
        if hasattr(self, '_prev_u1_noisy'):
            steer_smoothed = 0.8 * steer_clipped + 0.2 * self._prev_u1_noisy[0]
            u1_noisy[0] = np.clip(steer_smoothed, self.steer_min_bound, self.steer_max_bound)
        else:
            u1_noisy[0] = steer_clipped
        
        self._prev_u1_noisy = np.copy(u1_noisy)
        return u1_noisy.astype(np.float64)
    
    def step(self, action):
        original_action = self.denormalize_action(action)
        # u1 = self._get_noisy_u1()
        u1 = self._get_spline_tracking_u1()
        u = np.concatenate([u1, original_action])
        
        self.model.step(u)
        observation = self._get_observation()
        reward = self._calculate_reward()
        self._record_trajectories()

        if self.enable_visualization:
            self._render_frame()  
            if self.render_mode == "human":
                plt.pause(0.001)  
        
        terminated = self.model.is_finish
        truncated = False
        X1, Y1 = self.model.getXYi(self.model.x, 0)
        X2, Y2 = self.model.getXYi(self.model.x, 1)  
        info = {
            "reward_r_force": np.array(self.reward_info.get("r_force", 0.0), dtype=np.float32),
            "reward_r_align_rear": np.array(self.reward_info.get("r_align_rear", 0.0), dtype=np.float32),
            "reward_r_align_front": np.array(self.reward_info.get("r_align_front", 0.0), dtype=np.float32),
            # "reward_r_smooth": np.array(self.reward_info.get("r_smooth", 0.0), dtype=np.float32),
            "reward_r_progress": np.array(self.reward_info.get("r_progress", 0.0), dtype=np.float32),
            "reward_r_stability": np.array(self.reward_info.get("r_stability", 0.0), dtype=np.float32),
            "reward_val_force": np.array(self.reward_info.get("val_force", 0.0), dtype=np.float32),
            "reward_val_delta_psi_rear": np.array(self.reward_info.get("val_delta_psi_rear", 0.0), dtype=np.float32),
            "reward_val_delta_psi_front": np.array(self.reward_info.get("val_delta_psi_front", 0.0), dtype=np.float32),
            'Fh2': (self.model.Fh_arch[self.model.count, 2], 
                    self.model.Fh_arch[self.model.count, 3]),
            'pos_error': np.hypot(X2 - X1, Y2 - Y1),
            'u1': u1,
            'u2_normalized': action,
            'u2_original': original_action,
            'x': np.array([X1, Y1, X2, Y2]),
            "hinge_force_penalty": self.hinge_force_penalty,
            "control_smooth_penalty": self.control_smooth_penalty
        }
        
        # # 物理熔断
        # Fh2_x = self.model.Fh_arch[self.model.count, 2]
        # Fh2_y = self.model.Fh_arch[self.model.count, 3]
        # current_force = np.hypot(Fh2_x, Fh2_y)
        # FORCE_TERMINATE_THRESHOLD = 10000.0 
        
        # if current_force > FORCE_TERMINATE_THRESHOLD:
        #     terminated = True
        #     reward -= 2000.0 
        #     info['termination_reason'] = 'force_limit'
        # else:
        #     info['termination_reason'] = 'time_limit'

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, clear_frames=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
            
        # ================== 【新增】在这里实现每回合随机速度 ==================
        # 这样每次环境重置，速度都是新的（例如第一把 0.3，第二把 0.8）
        v_init = self.rng.uniform(0.2, 1.0) 
        psi_init = self.config['Psi_o_0'] 
        
        vx_init = v_init * np.cos(psi_init)
        vy_init = v_init * np.sin(psi_init)
        
        # 更新 Config，这样重新生成 Model 时会用到新参数
        self.config['X_dot_o_0'] = vx_init
        self.config['Y_dot_o_0'] = vy_init
        # ================================================================

        # ================== 【新增】生成随机路径 ==================
        self._generate_random_spline_path()
        # ========================================================

        # 重新初始化模型（或者仅重置状态，取决于你的实现偏好）
        # 推荐保留这行，确保模型参数彻底更新
        self.model = Model2D2C(self.config)
        self.steer_episode_offset = self.rng.normal(
            loc=0, 
            scale=self.steer_episode_base_std
        )
        self.steer_episode_offset = np.clip(
            self.steer_episode_offset,
            self.steer_min_bound,
            self.steer_max_bound
        )
        if hasattr(self, '_prev_u1_noisy'): del self._prev_u1_noisy
        
        if not self.vecnorm_frozen:
            # Warmup
            zero_action = np.zeros(4)
            for _ in range(5):
                self.model.step(np.concatenate([self.u1_random, zero_action]))
                _ = self._get_observation()
            
            # Reset Model State
            self.model.count = 0
            self.model.x = np.array([
                self.config['X_o_0'], self.config['Y_o_0'], self.config['Psi_o_0'],
                self.config['Psi_1_0'], self.config['Psi_2_0'],
                self.config['X_dot_o_0'],   # 修复：使用配置的 VX
                self.config['Y_dot_o_0'],   # 修复：使用配置的 VY
                self.config['Psi_dot_o_0'], # 修复：使用配置的 角速度
                self.config['Psi_dot_1_0'], 
                self.config['Psi_dot_2_0'], 
            ], dtype=np.float64)
            self.model.x_arch[0, :] = self.model.x
            self.model.u_arch.fill(0) 
            self.model.Fh_arch.fill(0)
            if self.enable_visualization: self._reset_visualization()

        options = options or {}
        final_clear_frames = options.get("clear_frames", clear_frames if clear_frames is not None else False)
        
        if self.enable_visualization:
            if final_clear_frames and not self.is_sim_finished:
                self.render_frames = []
            self.trajectories = {
                'cargo': [], 'car1': [], 'car2': [], 'hinge1': [], 'hinge2': []
            }
            self._reset_visualization() 
        self.is_sim_finished = False
        observation = self._get_observation()
        self._record_trajectories()
        return observation, {}

    def freeze_vecnorm(self):
        """冻结 VecNorm 统计量"""
        self.vecnorm_frozen = True
        print("观测归一化统计量已冻结，进入评测模式")

    def unfreeze_vecnorm(self):
        """解冻 VecNorm 统计量"""
        self.vecnorm_frozen = False
        print("观测归一化统计量已解冻，进入训练模式")

    def get_vecnorm_state(self):
        return {
            "vecnorm_mean": self.vecnorm_mean,
            "vecnorm_var": self.vecnorm_var,
            "vecnorm_count": self.vecnorm_count,
            "vecnorm_decay": self.vecnorm_decay,
            "vecnorm_eps": self.vecnorm_eps,
            "vecnorm_frozen": self.vecnorm_frozen
        }

    def set_vecnorm_state(self, vecnorm_state):
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
        i_sim = self.model.count
        x = self.model.x_arch[i_sim, :]
        self.trajectories['cargo'].append((x[0], x[1]))
        self.trajectories['car1'].append(self.model.getXYi(x, 0))
        self.trajectories['car2'].append(self.model.getXYi(x, 1))
        self.trajectories['hinge1'].append(self.model.getXYhi(x, 0))
        self.trajectories['hinge2'].append(self.model.getXYhi(x, 1))

    def _reset_visualization(self):
        if self.fig is not None: plt.close(self.fig)
        self.fig, self.ax = plt.subplots(figsize=(8, 8), dpi=60)
        self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        self.ax.set_facecolor('#f8f8f8')
        self.ax.set_xlabel('X (m)', fontsize=20)
        self.ax.set_ylabel('Y (m)', fontsize=20)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("两车运载超大件系统仿真可视化", fontsize=16)

        self.plot_handles = {
            'tire': None,
            'Fh': [],
            'hinge': [],
            'cargo': None,
            'car': [],
            'cargo_traj': self.ax.plot([], [], 'k--', alpha=0.3, linewidth=1)[0],
            'car1_traj': self.ax.plot([], [], '#3498db', linestyle='--', alpha=0.4, linewidth=1)[0],
            'car2_traj': self.ax.plot([], [], '#e74c3c', linestyle='--', alpha=0.4, linewidth=1)[0],
            # ================== 【新增】参考路径 Handle ==================
            # 用红色点划线表示前车计划要走的路径
            'ref_path': self.ax.plot([], [], 'r-.', alpha=0.5, linewidth=1.5, label='Target Path')[0],
            # ===========================================================
            'hinge1_traj': self.ax.plot([], [], ':', color='blue', alpha=0.2, linewidth=0.8)[0],
            'hinge2_traj': self.ax.plot([], [], ':', color='orange', alpha=0.2, linewidth=0.8)[0]
        }
        self.first_render = True

    def _render_frame(self):
        if self.fig is None or self.ax is None:
            self._reset_visualization()

        i_sim = self.model.count
        tire_segments = self.model.getTireVis(i_sim)
        fh_arrows, hinge_markers = self.model.getHingeVis(i_sim)
        cargo_polygon = self.model.getOversizedCargoVis(i_sim)
        car_polygons = self.model.getCarrierVis(i_sim)
        fh_color = self.model.config.get('c_Fh', 'green')
        fh_width = self.model.config.get('width_Fh', 0.01)

        if self.first_render:
            self.plot_handles['tire'] = LineCollection(
                tire_segments,
                colors=self.model.config['c_tire'],
                linewidths=self.model.config['lw_tire'],
                zorder=2.4
            )
            self.ax.add_collection(self.plot_handles['tire'])

            for arrow_data in fh_arrows:
                h = self.ax.arrow(
                    arrow_data[0], arrow_data[1], arrow_data[2], arrow_data[3],
                    width=fh_width,
                    color=fh_color,
                    zorder=2.4,
                    alpha=0.7
                )
                self.plot_handles['Fh'].append(h)

            for marker_poly in hinge_markers:
                h = Polygon(marker_poly, zorder=2.6, alpha=1.0, fc='black', ec='white')
                self.ax.add_patch(h)
                self.plot_handles['hinge'].append(h)

            if cargo_polygon:
                self.plot_handles['cargo'] = Polygon(
                    cargo_polygon, zorder=2.5, alpha=self.model.config['alpha_o'],
                    fc=self.model.config['fc_o'], ec='black', linewidth=1.5
                )
                self.ax.add_patch(self.plot_handles['cargo'])

            for i, poly in enumerate(car_polygons):
                h = Polygon(
                    poly, zorder=2.5, alpha=self.model.config['alpha_c'],
                    fc=self.model.config['fc_c'][i], ec='black', linewidth=1
                )
                self.ax.add_patch(h)
                self.plot_handles['car'].append(h)

            self.first_render = False

        else:
            self.plot_handles['tire'].set_segments(tire_segments)

            for h in self.plot_handles['Fh']:
                h.remove()
            self.plot_handles['Fh'].clear()
            for arrow_data in fh_arrows:
                h = self.ax.arrow(
                    arrow_data[0], arrow_data[1], arrow_data[2], arrow_data[3],
                    width=fh_width,
                    color=fh_color,
                    zorder=2.4,
                    alpha=0.7
                )
                self.plot_handles['Fh'].append(h)

            for h, marker_poly in zip(self.plot_handles['hinge'], hinge_markers):
                h.set_xy(marker_poly)

            if cargo_polygon and self.plot_handles['cargo']:
                self.plot_handles['cargo'].set_xy(cargo_polygon)
            elif self.plot_handles['cargo']:
                self.plot_handles['cargo'].set_xy([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])

            for h, poly in zip(self.plot_handles['car'], car_polygons):
                h.set_xy(poly)

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
            if hasattr(self, 'path_x') and hasattr(self, 'path_y'):
                self.plot_handles['ref_path'].set_data(self.path_x, self.path_y)

        X_o = self.model.x_arch[i_sim, 0]
        Y_o = self.model.x_arch[i_sim, 1]
        vis_range = self.model.config['range']
        self.ax.set_xlim([X_o - vis_range, X_o + vis_range])
        self.ax.set_ylim([Y_o - vis_range, Y_o + vis_range])

        self.fig.canvas.draw_idle()

        if self.render_mode == "rgb_array":
            frame = None
            buf = None
            img = None
            try:
                buf = BytesIO()
                self.fig.savefig(
                    buf, format='png', bbox_inches='tight', dpi=96,
                    facecolor=self.fig.get_facecolor()
                )
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
                frame = np.array(img, dtype=np.uint8)

                if len(self.render_frames) > 0:
                    ref_shape = self.render_frames[0].shape
                    if frame.shape != ref_shape:
                        frame = cv2.resize(
                            frame, (ref_shape[1], ref_shape[0]),
                            interpolation=cv2.INTER_AREA
                        )

                max_cache_frames = 1001
                if len(self.render_frames) >= max_cache_frames:
                    self.render_frames.pop(0)
                self.render_frames.append(frame)
                return frame
            except Exception as e:
                print(f"帧保存失败，错误：{type(e).__name__}: {e}")
            finally:
                if buf is not None: buf.close()
                if img is not None: del img
                if buf is not None: del buf
            return frame

    def save_eval_video(self, eval_round=None, video_save_dir=None):
        if not self.enable_visualization or self.render_mode != "rgb_array" or len(self.render_frames) == 0:
            print("警告：不满足视频保存条件")
            return None
        
        out = None
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if video_save_dir is None:
                default_ckpt_dir = os.path.join(current_dir, "checkpoints")
                video_save_dir = default_ckpt_dir if os.path.exists(default_ckpt_dir) else os.path.join(current_dir, "output")
            os.makedirs(video_save_dir, exist_ok=True)
            
            time_str = datetime.datetime.now().strftime('%y%m%d%H%M%S')
            file_prefix = f"{self.config_name}_eval_round_{eval_round}" if eval_round is not None else f"{self.config_name}_vis"
            file_name = f"{file_prefix}_{time_str}.mp4"
            video_path = os.path.join(video_save_dir, file_name)
            
            fps = self.metadata['render_fps']
            height, width, _ = self.render_frames[0].shape
            video_writer_opened = False
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            if out.isOpened():
                video_writer_opened = True
            
            if not video_writer_opened:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_path = video_path.replace(".mp4", ".avi")
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                if out.isOpened():
                    video_writer_opened = True
                    print(f"mp4格式不支持，切换为avi格式，保存路径：{video_path}")
                else:
                    raise RuntimeError("无法初始化VideoWriter")
            
            batch_size = 60
            total_frames = len(self.render_frames)
            for i in range(0, total_frames, batch_size):
                batch_frames = self.render_frames[i:i+batch_size]
                for frame in batch_frames:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.uint8)
                    out.write(bgr_frame)
                del batch_frames
            
            out.release()
            out = None
            print(f"单轮评测视频已成功保存至: {video_path}")
            return video_path
        
        except Exception as e:
            if out is not None and out.isOpened():
                out.release()
            out = None
            print(f"生成视频失败：{e}")
            return None
        finally:
            if out is not None and out.isOpened():
                out.release()
    
    def clear_render_frames(self):
        if hasattr(self, 'render_frames'):
            self.render_frames = []
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)

        if self.enable_visualization and self.render_mode == "rgb_array" and len(self.render_frames) > 0:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(current_dir, "output")
                os.makedirs(output_dir, exist_ok=True)
                
                time_str = datetime.datetime.now().strftime(r'%y%m%d%H%M%S')
                file_name = f"{self.config_name}_vis_{time_str}.mp4"
                video_path = os.path.join(output_dir, file_name)

                fps = self.metadata['render_fps']
                height, width, _ = self.render_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_path = video_path.replace(".mp4", ".avi")
                    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                for idx, frame in enumerate(self.render_frames):
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)

                out.release()
                print(f"可视化视频已成功保存至: {video_path}")

            except Exception as e:
                print(f"生成视频失败：{e}")


# 注册环境
gym.register(
    id="TwoCarrierEnv-v1",
    entry_point="occt_2d2c:TwoCarrierEnv",
    max_episode_steps=1024,
    kwargs={}
)

if __name__ == "__main__":
    # 1. 初始化环境，开启可视化模式 (enable_visualization=True)
    # render_mode="rgb_array" 用于后台生成视频，不弹窗
    env = gym.make("TwoCarrierEnv-v1", render_mode="rgb_array", enable_visualization=True)
    
    print("\n=== 🚀 开始样条曲线跟踪测试 ===")
    
    # 2. 重置环境 (Seed固定以便复现)
    obs, info = env.reset(seed=42)
    
    # 获取原始环境句柄，用于访问内部变量
    raw_env = env.unwrapped
    print(f"✅ 随机路径已生成，路径长度: {len(raw_env.path_x)} 点")
    print(f"✅ 前车初始位置: ({raw_env.model.x[0]:.2f}, {raw_env.model.x[1]:.2f})")
    
    # 3. 运行仿真循环
    # 我们运行 300 步，足够观察前车过弯
    steps = 1000
    print(f"⏳ 正在运行 {steps} 步仿真...")
    
    for i in range(steps):
        # 后车（Agent）给一个静止或简单的动作，我们主要观察前车（环境控制）
        # 动作全是 0 (归一化后)，意味着后车处于中间状态
        action = np.zeros(4) 
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 50 == 0:
            # 打印前车与路径的偏差 (Lat Error)
            # 我们需要手动计算一下当前的偏差来打印日志
            x_front, y_front = raw_env.model.getXYi(raw_env.model.x, 0)
            target_y = raw_env.spline_func(x_front)
            error = y_front - target_y
            print(f"Step {i:03d} | 前车X: {x_front:.2f} | 目标Y: {target_y:.2f} | 实际Y: {y_front:.2f} | 偏差: {error:.4f}")

        if terminated or truncated:
            print("⚠️ 环境提前终止 (可能是触发了物理熔断)")
            break
            
    # 4. 保存视频
    # 视频将保存在当前目录下的 output_test 文件夹中
    print("\n💾 正在保存测试视频...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "output_test")
    
    video_path = raw_env.save_eval_video(eval_round="spline_verify", video_save_dir=save_dir)
    
    if video_path:
        print(f"🎉 视频保存成功！请打开查看效果: {video_path}")
        print("👀 观察重点：视频中应该有一条红色的虚线（目标路径），前车（蓝色）应该沿着这条线行驶。")
    else:
        print("❌ 视频保存失败，请检查环境配置。")
        
    env.close()