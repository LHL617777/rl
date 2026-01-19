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

    def __init__(self, render_mode=None, config_path=None, enable_visualization=False, 
                 vecnorm_frozen: bool = False, vecnorm_mean=None, vecnorm_var=None):
        super().__init__()
        
        # 加载2d2c.yaml配置文件
        self.config_name = '2d2c'
        self.config = self._load_config(config_path)

        self.rng = np.random.default_rng()
        self.model = Model2D2C(self.config)
        
        # =========================================================================================
        # ================== 1. 状态空间 (12维 - Rear/Follower View) ==================
        # 移除了 Human_Steer，增加了路径跟踪和力变化率信息
        # [0]    Hitch_Angle       : 铰接角 (前车 - 后车) -> 安全核心
        # [1]    Omega_Rear        : 后车角速度
        # [2]    Omega_Diff        : 角速度差 (前 - 后) -> 折叠趋势
        # [3]    V_Rear_Body_X     : 后车纵向速度
        # [4]    V_Rear_Body_Y     : 后车侧滑速度
        # [5]    Fh_Rear_Long      : 纵向牵引力
        # [6]    Fh_Rear_Lat       : 横向撕裂力 (意图感知的核心替代品)
        # [7]    Risk_Predict      : 预测未来1s的折叠风险
        # [8]    Local_Cargo_X     : 货物相对后车的 X (后车也要看路)
        # [9]    Local_Cargo_Y     : 货物相对后车的 Y (横向偏差)
        # [10]   Heading_Err_Cargo : 后车与货物路径的夹角
        # [11]   dFh_Lat           : 横向力变化率 (Jerk Sensing) -> 感知前车是否急打方向
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

        # 缓存上一帧的力，用于计算变化率
        self.last_Fh_lat = 0.0

        # 人类驾驶参数
        self.human_noise_std = 0.03   # 正常驾驶的抖动
        self.human_panic_prob = 0.02  # 出现“惊慌失措”急转弯的概率
        
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
    
    def _get_human_action(self, obs_state):
        """
        [Adversarial Human Driver]
        模拟一个有噪声的人类驾驶员。
        博弈点：他并不总是完美的，有时会犯错（噪声），Agent必须容忍这些错误。
        """
        x = obs_state
        target_x, target_y = x[0], x[1] # 货物中心作为目标
        
        self_x, self_y = self.model.getXYi(x, 0) # 前车位置
        psi_front = x[3]
        
        # 1. 基础驾驶逻辑 (Pure Pursuit)
        dx = target_x - self_x
        dy = target_y - self_y
        target_angle = np.arctan2(dy, dx)
        steer_error = self._normalize_angle(target_angle - psi_front)
        
        kp = 0.8
        base_steer = np.clip(kp * steer_error, -np.pi/6, np.pi/6)
        
        # 2. [博弈对抗] 注入人类的不确定性
        # 正常抖动
        noise = self.rng.normal(0, self.human_noise_std)
        
        # "惊慌时刻" (Panic Moment) - 模拟突然避让或操作失误
        # 这是对后车Robustness的最大考验
        if self.rng.random() < self.human_panic_prob:
            panic_steer = self.rng.choice([-0.3, 0.3]) # 突然大幅度打方向
            noise += panic_steer
            
        final_steer = np.clip(base_steer + noise, -np.pi/6, np.pi/6)
        
        # 简单的定速巡航
        throttle = 300.0 
        
        return np.array([final_steer, 0, throttle, 0])
    
    def _get_observation(self):
        """
        [Follower View - 12 Dim]
        无法获取前车转向，必须通过物理量的变化来感知。
        """
        x = self.model.x
        i_sim = self.model.count
        
        # 状态提取
        X_cargo, Y_cargo = x[0], x[1]
        Psi_cargo = x[2]
        Psi_1 = x[3] # Front
        Psi_2 = x[4] # Rear (Self)
        Psi_dot_1 = x[8]
        Psi_dot_2 = x[9]
        
        X_2, Y_2 = self.model.getXYi(x, 1) # Rear Pos
        X_dot_2, Y_dot_2 = self.model.getXYdoti(x, 1)
        
        # --- 1. 铰接状态 (安全核心) ---
        hitch_angle = self._normalize_angle(Psi_1 - Psi_2)
        omega_diff = Psi_dot_1 - Psi_dot_2
        
        # --- 2. 自身动力学 ---
        vx_body = X_dot_2 * np.cos(Psi_2) + Y_dot_2 * np.sin(Psi_2)
        vy_body = -X_dot_2 * np.sin(Psi_2) + Y_dot_2 * np.cos(Psi_2)
        
        # --- 3. 意图感知 (Force Sensing) ---
        Fh2_x = self.model.Fh_arch[i_sim, 2] 
        Fh2_y = self.model.Fh_arch[i_sim, 3]
        # 投影到后车坐标系
        fh_long = Fh2_x * np.cos(Psi_2) + Fh2_y * np.sin(Psi_2)
        fh_lat  = -Fh2_x * np.sin(Psi_2) + Fh2_y * np.cos(Psi_2)
        
        # 计算横向力变化率 (Jerk) - 这是感知前车急转向的“替代传感器”
        dfh_lat = fh_lat - self.last_Fh_lat
        self.last_Fh_lat = fh_lat # 更新缓存
        
        # --- 4. 风险预测 ---
        risk_pred = hitch_angle + omega_diff * 1.0
        
        # --- 5. 路径跟踪 (后车也要看路) ---
        # 即使是被拉着走，后车也应该知道自己在路的哪一边
        loc_x_cargo, loc_y_cargo = self._transform_to_local(X_cargo, Y_cargo, X_2, Y_2, Psi_2)
        heading_err_cargo = self._normalize_angle(Psi_2 - Psi_cargo)

        # 组装 12维 向量
        raw_obs = np.array([
            hitch_angle,        # [0] 铰接角
            Psi_dot_2,          # [1] 自身角速度
            omega_diff,         # [2] 相对角速度
            vx_body,            # [3] 自身Vx
            vy_body,            # [4] 自身Vy (侧滑)
            fh_long,            # [5] 牵引力
            fh_lat,             # [6] 侧向力 (意图感知)
            risk_pred,          # [7] 风险预测
            loc_x_cargo,        # [8] 相对路 X
            loc_y_cargo,        # [9] 相对路 Y (偏差)
            heading_err_cargo,  # [10] 相对路朝向
            dfh_lat             # [11] 侧力变化率 (Action Inference)
        ], dtype=np.float64)
        
        self._update_vecnorm_stats(raw_obs)
        return self._normalize_observation(raw_obs)
    
    def _calculate_reward(self):
        """Physics-Feedback Reward with Progress Projection"""
        i_sim = self.model.count
        Fh2_x = self.model.Fh_arch[i_sim, 2]
        Fh2_y = self.model.Fh_arch[i_sim, 3]
        F_force_mag = np.hypot(Fh2_x, Fh2_y)
        
        x = self.model.x
        Psi_cargo = x[2]      # 货物航向
        Psi_front = x[3]      # 前车航向 (Tractor)
        Psi_rear = x[4]       # 后车航向 (Carrier)

        # 1. 核心博弈目标：防折叠 (Stability)
        # 不管前车怎么乱开，后车必须把角度控制在安全范围内
        hitch_angle = self._normalize_angle(Psi_front - Psi_rear)
        r_stability = -2.0 * np.square(hitch_angle)
        
        # 2. 顺从性 (Compliance) - 最小化内力对抗
        # 如果后车转对了方向，铰接处的横向剪切力应该很小
        # 这意味着后车“顺”着前车的意图在走
        i_sim = self.model.count
        Fh2_x = self.model.Fh_arch[i_sim, 2]
        Fh2_y = self.model.Fh_arch[i_sim, 3]
        fh_lat  = -Fh2_x * np.sin(Psi_rear) + Fh2_y * np.cos(Psi_rear)
        r_compliance = -0.0005 * np.square(fh_lat) 
        
        # 3. 辅助跟踪 (Auxiliary Tracking)
        # 如果可能，后车也尽量别偏离路中心
        target_x, target_y = x[0], x[1]
        self_x, self_y = self.model.getXYi(x, 1)
        # 这里用一种软约束，不要因为为了对齐路而跟前车较劲
        dist_to_cargo = np.hypot(target_x - self_x, target_y - self_y)
        # 我们只惩罚横向偏差，不惩罚纵向距离(因为必须跟着前车)
        # 这是一个近似的横向偏差惩罚
        r_track = -0.1 * dist_to_cargo
        
        F_safe = self.config.get('force_safe', 2000.0) 

        # R_force
        r_force = -1.0 * np.tanh(F_force_mag / F_safe)

        # R_align_rear: 后车与货物夹角协同 (原有)
        delta_psi_rear = self._normalize_angle(Psi_rear - Psi_cargo)
        r_align_rear = -1.0 * np.square(delta_psi_rear)

        # [新增] R_align_front: 前车与货物夹角协同
        # 鼓励前车也尽量与货物保持一致，减少“折叠”风险
        delta_psi_front = self._normalize_angle(Psi_front - Psi_cargo)
        r_align_front = -1.0 * np.square(delta_psi_front)


        # R_smooth
        if i_sim > 0:
            u_curr = self.model.u_arch[i_sim, 4:8]
            u_prev = self.model.u_arch[i_sim - 1, 4:8]
            steer_diff = np.sum(np.abs(u_curr[:2] - u_prev[:2]))
            thrust_diff = np.sum(np.abs(u_curr[2:] - u_prev[2:])) / 1000.0
            r_smooth = -0.1 * (5.0 * steer_diff + 0.5 * thrust_diff)
        else:
            r_smooth = 0.0

        # R_progress: 向量投影 (Vector Projection)
        # 用货物中心
        target_x, target_y = x[0], x[1]
        self_x, self_y = self.model.getXYi(x, 1)
        
        vec_x = target_x - self_x
        vec_y = target_y - self_y
        dist = np.hypot(vec_x, vec_y)
        
        if dist > 1e-3:
            dir_x, dir_y = vec_x / dist, vec_y / dist
        else:
            dir_x, dir_y = 0, 0
            
        X_dot_2, Y_dot_2 = self.model.getXYdoti(x, 1)
        v_effective = X_dot_2 * dir_x + Y_dot_2 * dir_y
        
        if F_force_mag < F_safe:
            r_progress = 0.2 * v_effective 
        else:
            r_progress = 0.0
        
        # # R_stability
        # Psi_dot_rear = self.model.x[9]
        # r_stability = -5.0 * np.square(Psi_dot_rear)

        total_reward = r_stability + r_compliance + r_track + r_smooth
        
        self.reward_info = {
            "r_stability": r_stability * 1.0,
            "r_compliance": r_compliance * 1.0,
            "r_track": r_track * 1.0,
            "r_smooth": r_smooth * 1.0,
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
        # 1. 人类驾驶前车 (Stochastic Leader)
        u1_human = self._get_human_action(self.model.x)
        
        # 2. Agent 辅助后车 (Defender Follower)
        u2_agent = self.denormalize_action(action)
        
        # 3. 物理步进
        u_total = np.concatenate([u1_human, u2_agent])
        self.model.step(u_total)

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
            "reward_r_compliance": np.array(self.reward_info.get("r_compliance", 0.0), dtype=np.float32),
            "reward_r_track": np.array(self.reward_info.get("r_track", 0.0), dtype=np.float32),
            "reward_r_smooth": np.array(self.reward_info.get("r_smooth", 0.0), dtype=np.float32),
            "reward_r_stability": np.array(self.reward_info.get("r_stability", 0.0), dtype=np.float32),
            "reward_val_force": np.array(self.reward_info.get("val_force", 0.0), dtype=np.float32),
            "reward_val_delta_psi_rear": np.array(self.reward_info.get("val_delta_psi_rear", 0.0), dtype=np.float32),
            "reward_val_delta_psi_front": np.array(self.reward_info.get("val_delta_psi_front", 0.0), dtype=np.float32),
            'Fh2': (self.model.Fh_arch[self.model.count, 2], 
                    self.model.Fh_arch[self.model.count, 3]),
            'pos_error': np.hypot(X2 - X1, Y2 - Y1),
            'u1': u1_human,
            'u2_normalized': action,
            'u2_original': self.denormalize_action(action),
            'x': np.array([X1, Y1, X2, Y2]),
            "hinge_force_penalty": self.hinge_force_penalty,
            "control_smooth_penalty": self.control_smooth_penalty
        }
        
        # 物理熔断
        Fh2_x = self.model.Fh_arch[self.model.count, 2]
        Fh2_y = self.model.Fh_arch[self.model.count, 3]
        current_force = np.hypot(Fh2_x, Fh2_y)
        FORCE_TERMINATE_THRESHOLD = 10000.0 
        
        if current_force > FORCE_TERMINATE_THRESHOLD:
            terminated = True
            reward -= 2000.0 
            info['termination_reason'] = 'force_limit'
        else:
            info['termination_reason'] = 'time_limit'
        
        if np.abs(observation[0]) > np.pi/2.5: 
            terminated = True
            reward -= 2000.0 
            info['termination_reason'] = 'excessive_hitch_angle'
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, clear_frames=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        # ================== 1. 随机化初始状态 ==================
        # 模拟不同的入弯速度和初始偏差
        v_init = self.rng.uniform(0.5, 1.2)
        psi_init = self.rng.uniform(-0.3, 0.3)
        
        # 更新 Config (确保重新初始化的 Model 使用新参数)
        self.config['X_dot_o_0'] = v_init * np.cos(psi_init)
        self.config['Y_dot_o_0'] = v_init * np.sin(psi_init)
        self.config['Psi_o_0'] = psi_init
        self.config['Psi_1_0'] = psi_init 
        self.config['Psi_2_0'] = 0.0

        # ================== 2. 重新初始化物理模型 ==================
        self.model = Model2D2C(self.config)
        # 双重保险：强制覆盖状态向量
        self.model.x[3] = psi_init
        self.model.x[5] = v_init * np.cos(psi_init)
        self.model.x[6] = v_init * np.sin(psi_init)
        
        # 重置力缓存 (用于计算 dfh_lat)
        self.last_Fh_lat = 0.0 
        
        # ================== 3. 物理热身 (Physical Warmup) ==================
        # 【关键修改】无论是否训练，都必须跑几步！
        # 原因：我们需要填充 self.last_Fh_lat，否则第一帧的力变化率会是巨大的跳变。
        warmup_steps = 5
        u2_passive = np.zeros(4) # 后车在热身阶段不动作
        
        for _ in range(warmup_steps):
            # 获取人类驾驶动作 (前车)
            u1_human = self._get_human_action(self.model.x)
            
            # 物理步进
            self.model.step(np.concatenate([u1_human, u2_passive]))
            
            # 调用观测函数：
            # 1. 它会计算当前的力 Fh_lat
            # 2. 它会更新 self.last_Fh_lat = Fh_lat
            # 3. 如果没冻结，它还会更新 VecNorm 统计量
            _ = self._get_observation()
            
        # 【注意】热身结束后，不要再重置 self.model.x！
        # 我们直接从第 5 步开始 Episode，这样力反馈才是连续的，符合物理规律。

        # ================== 4. 可视化与清理 ==================
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
        
        # 获取正式的第一帧观测
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
    env = gym.make("TwoCarrierEnv-v1", render_mode=None)
    
    print("\n--- 【速度验证】 ---")
    obs, info = env.reset(seed=42)
    
    # 1. 获取物理引擎的真实状态 (Raw Physics State)
    raw_x_dot = env.unwrapped.model.x[5] # 全局 Vx
    raw_y_dot = env.unwrapped.model.x[6] # 全局 Vy
    raw_v_mag = np.hypot(raw_x_dot, raw_y_dot)
    
    print(f"【物理引擎真值】 绝对速度: {raw_v_mag:.4f} m/s (应 > 0.2)")
    print(f"【物理引擎真值】 全局 Vx : {raw_x_dot:.4f} m/s")
    
    # 2. 对比观测值 (Normalized Observation)
    print(f"【归一化观测值】 车身 Vx : {obs[2]:.4f} (接近0是正常的，因为均值也是{raw_v_mag:.2f})")
    
    env.close()