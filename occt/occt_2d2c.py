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
        
        # 加载2d2c.yaml配置文件（优先使用传入路径，否则加载同一目录下的2d2c.yaml）
        self.config_name = '2d2c'
        self.config = self._load_config(config_path)

        # ================== [新增] 初始速度热启动 ==================
        # 设定一个初始速度大小，例如 0.5 m/s
        self.rng = np.random.default_rng()  # 随机数生成器（保证可复现性）
        v_init = self.rng.uniform(0.2, 1.0) 
        
        # 获取当前的初始朝向 (在域随机化之后获取)
        psi_init = self.config['Psi_o_0'] 
        
        # 分解速度到 X 和 Y 轴
        # 假设初始时刻是直线行驶，所有部件速度一致
        vx_init = v_init * np.cos(psi_init)
        vy_init = v_init * np.sin(psi_init)
        
        # 将速度写入 Config (用于 Model 初始化)
        self.config['X_dot_o_0'] = vx_init
        self.config['Y_dot_o_0'] = vy_init
        self.config['Psi_dot_o_0'] = 0.0 # 初始角速度为0
        # ========================================================
        
        # 初始化动力学模型（使用加载的config）
        self.model = Model2D2C(self.config)
        
        # =========================================================================================
        # 修改：状态空间定义（以第二辆车/后车为基准的坐标系）
        # 维度映射 (12维):
        # [0]  X_2          : 后车全局X坐标
        # [1]  Y_2          : 后车全局Y坐标
        # [2]  Psi_2        : 后车全局航向角
        # [3]  Psi_o_2      : 超大件相对后车夹角 (Psi_o - Psi_2)
        # [4]  Psi_1_o      : 前车相对超大件夹角 (Psi_1 - Psi_o)
        # [5]  X_dot_2      : 后车全局X速度
        # [6]  Y_dot_2      : 后车全局Y速度
        # [7]  Psi_dot_2    : 后车角速度
        # [8]  Psi_dot_o_2  : 超大件相对后车角速度差 (Psi_dot_o - Psi_dot_2)
        # [9]  Psi_dot_1_o  : 前车相对超大件角速度差 (Psi_dot_1 - Psi_dot_o)
        # [10] Fh2_x        : 后车铰接力X
        # [11] Fh2_y        : 后车铰接力Y
        # =========================================================================================
        
        # 定义物理边界（主要用于参考，VecNorm会处理实际数值范围）
        obs_low = np.array([
            -np.inf, -np.inf, -np.pi,   # X_2, Y_2, Psi_2
            -np.pi, -np.pi,             # Psi_o_2, Psi_1_o (相对角)
            -20, -20, -5,               # X_dot_2, Y_dot_2, Psi_dot_2
            -10, -10,                   # Psi_dot_o_2, Psi_dot_1_o
            -1e5, -1e5                  # Fh
        ])
        obs_high = np.array([
            np.inf, np.inf, np.pi,
            np.pi, np.pi,
            20, 20, 5,
            10, 10,
            1e5, 1e5
        ])
        
        # 归一化后的观测空间（合理范围，覆盖归一化后的所有可能值）
        obs_norm_low = np.full(12, -1000.0, dtype=np.float64)  # 12维观测，下限-1000
        obs_norm_high = np.full(12, 1000.0, dtype=np.float64)   # 12维观测，上限1000
        self.observation_space = spaces.Box(
            low=obs_norm_low, high=obs_norm_high, dtype=np.float64
        )
        
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
        # 1. 前轮转角：轮级大偏移 + 步级小扰动（核心：提升轮间差异）
        self.steer_episode_base_std = np.pi/15  # 轮级基础偏移标准差
        self.steer_step_dynamic_std = 0.008     # 步级动态小扰动标准差
        self.steer_max_bound = np.pi/6          # 最大转角约束
        self.steer_min_bound = -np.pi/6
        # 新增：存储轮级固定基础偏移（每轮重置时更新）
        self.steer_episode_offset = 0.0
        # 2. 推力噪声
        self.thrust_noise_rel_std = 0.02 
        self.thrust_noise_abs_min = 0  
        self.thrust_noise_abs_max = 1e3 
        
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
        self.vecnorm_count = 0        # 统计更新次数（用于初始阶段的无偏估计）
        # ==================================================================================================================
        # 如果用户传入了固定的 mean 和 var，则直接加载并强制冻结
        if vecnorm_mean is not None and vecnorm_var is not None:
            self.vecnorm_mean = np.array(vecnorm_mean, dtype=np.float64)
            self.vecnorm_var = np.array(vecnorm_var, dtype=np.float64)
            
            # 完整性检查：确保维度匹配 (这里假设是12维)
            assert self.vecnorm_mean.shape == (12,), f"Expected mean shape (12,), got {self.vecnorm_mean.shape}"
            assert self.vecnorm_var.shape == (12,), f"Expected var shape (12,), got {self.vecnorm_var.shape}"
            
            # 强制执行最小方差限制，防止传入的方差中有0
            self.vecnorm_var = np.maximum(self.vecnorm_var, self.vecnorm_min_var)
            
            # 既然是传入固定值，通常意味着用于测试/部署，因此强制冻结
            self.vecnorm_frozen = True
            print(f"【TwoCarrierEnv】已加载固定归一化统计量，VecNorm 状态已冻结。")
        else:
            # 否则使用原本的在线统计逻辑
            self.vecnorm_frozen = vecnorm_frozen
            self.vecnorm_mean = np.zeros(12, dtype=np.float64) 
            self.vecnorm_var = np.ones(12, dtype=np.float64) * self.vecnorm_min_var
        
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
        """将原始动作（原始空间）归一化到[-1, 1]区间"""
        orig_range = self.original_action_high - self.original_action_low
        orig_range = np.where(orig_range == 0, 1e-8, orig_range)
        norm_action = 2 * (original_action - self.original_action_low) / orig_range - 1
        norm_action = np.clip(norm_action, -1, 1)
        return norm_action.astype(np.float64)

    def denormalize_action(self, normalized_action):
        """将归一化动作（[-1, 1]）反归一化到原始动作空间"""
        orig_range = self.original_action_high - self.original_action_low
        orig_action = self.original_action_low + (normalized_action + 1) * orig_range / 2
        orig_action = np.clip(
            orig_action, 
            self.original_action_low, 
            self.original_action_high
        )
        return orig_action.astype(np.float64)

    def _update_vecnorm_stats(self, obs):
        """在线更新 VecNorm 滑动统计量"""
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
        """执行观测归一化"""
        if self.vecnorm_mean is None:
            return obs
        obs_np = np.asarray(obs, dtype=np.float64)
        std_np = np.sqrt(self.vecnorm_var) + self.vecnorm_eps
        normalized_obs_np = (obs_np - self.vecnorm_mean) / std_np
        return normalized_obs_np.astype(obs.dtype) if hasattr(obs, 'dtype') else normalized_obs_np

    def _get_observation(self):
        """
        从模型提取观测值（新版：后车坐标系/相对状态）
        不修改 model.py，利用其提供的 getXYi, getXYdoti 等方法计算
        """
        x = self.model.x  # 当前全局状态向量 (10维)
        i_sim = self.model.count
        
        # --- 1. 获取基础物理量 ---
        # 索引定义 (基于 model.py):
        # x[0:2] Cargo Pos, x[2] Cargo Psi
        # x[3] Front Psi, x[4] Rear Psi
        # x[5:7] Cargo Vel, x[7] Cargo Omega
        # x[8] Front Omega, x[9] Rear Omega
        
        idx_rear = 1  # 后车索引
        idx_front = 0 # 前车索引
        
        # --- 2. 计算后车 (Agent) 的全局状态 ---
        # 调用 model 方法计算后车质心全局坐标 (X_2, Y_2)
        X_2, Y_2 = self.model.getXYi(x, idx_rear)
        Psi_2 = x[4]
        
        # 调用 model 方法计算后车全局速度 (X_dot_2, Y_dot_2)
        X_dot_2, Y_dot_2 = self.model.getXYdoti(x, idx_rear)
        Psi_dot_2 = x[9]
        
        # --- 3. 计算相对状态 ---
        Psi_o = x[2]      # 超大件航向
        Psi_dot_o = x[7]
        
        Psi_1 = x[3]      # 前车航向
        Psi_dot_1 = x[8]

        # 计算相对角度并归一化到 [-pi, pi]
        # Psi_o_2: 超大件相对后车
        Psi_o_2 = self._normalize_angle(Psi_o - Psi_2)
        # Psi_1_o: 前车相对超大件
        Psi_1_o = self._normalize_angle(Psi_1 - Psi_o)
        
        # 计算相对角速度 (无需归一化，直接作差)
        Psi_dot_o_2 = Psi_dot_o - Psi_dot_2
        Psi_dot_1_o = Psi_dot_1 - Psi_dot_o
        
        # --- 4. 获取铰接力 ---
        # Fh_arch 存储格式: [Fh1_x, Fh1_y, Fh2_x, Fh2_y]
        Fh2_x = self.model.Fh_arch[i_sim, 2]
        Fh2_y = self.model.Fh_arch[i_sim, 3]

        # --- 5. 组装观测向量 (12维) ---
        raw_obs = np.array([
            X_2, Y_2, Psi_2,            # [0-2] 后车全局位姿
            Psi_o_2, Psi_1_o,           # [3-4] 相对角度 (Articulations)
            X_dot_2, Y_dot_2, Psi_dot_2,# [5-7] 后车全局速度
            Psi_dot_o_2, Psi_dot_1_o,   # [8-9] 相对角速度
            Fh2_x, Fh2_y                # [10-11] 铰接力
        ], dtype=np.float64)
        
        # 步骤1：更新VecNorm统计量（仅训练阶段未冻结时生效）
        self._update_vecnorm_stats(raw_obs)
        
        # 步骤2：执行观测归一化（返回归一化后的观测）
        normalized_obs = self._normalize_observation(raw_obs)
        
        return normalized_obs

    def _calculate_reward(self):
        """
        基于物理反馈的领航跟随奖励函数 (Physics-Feedback Reward)
        无需参考轨迹，仅依赖动力学状态。
        """
        # ================= 1. 从 Model 获取实时物理量 =================
        i_sim = self.model.count
        Fh2_x = self.model.Fh_arch[i_sim, 2]
        Fh2_y = self.model.Fh_arch[i_sim, 3]
        F_force_mag = np.hypot(Fh2_x, Fh2_y)
        
        x = self.model.x
        Psi_cargo = x[2]      # 货物航向
        Psi_rear = x[4]       # 后车(Agent)航向
        
        # 货物速度
        V_cargo_mag = np.hypot(x[5], x[6])

        # ================= 2. 读取 Config 参数 =================
        F_safe = self.config.get('force_safe', 2000.0) 

        # ================= 3. 计算分项奖励 =================
        # R_force: 铰接力惩罚
        r_force = -1.0 * np.tanh(F_force_mag / F_safe)

        # R_align: 姿态协同惩罚 (后车与货物夹角)
        delta_psi = self._normalize_angle(Psi_rear - Psi_cargo)
        r_align = -2.0 * np.square(delta_psi)

        # R_smooth: 动作平滑
        if i_sim > 0:
            u_curr = self.model.u_arch[i_sim, 4:8]
            u_prev = self.model.u_arch[i_sim - 1, 4:8]
            steer_diff = np.sum(np.abs(u_curr[:2] - u_prev[:2]))
            thrust_diff = np.sum(np.abs(u_curr[2:] - u_prev[2:])) / 1000.0
            r_smooth = -0.1 * (5.0 * steer_diff + 0.5 * thrust_diff)
        else:
            r_smooth = 0.0

        # R_progress: 前进激励
        if F_force_mag < F_safe:
            r_progress = 0.1 * V_cargo_mag
        else:
            r_progress = 0.0
        
        # R_stability: 稳定性
        Psi_dot_rear = self.model.x[9]
        r_stability = -5.0 * np.square(Psi_dot_rear)

        # ================= 4. 总奖励合成 =================
        total_reward = (1.0 * r_force) + \
                       (2.0 * r_align) + \
                       (2.0 * r_smooth) + \
                       (2.0 * r_progress) + \
                       r_stability
        
        self.reward_info = {
            "r_force": r_force * 1.0,
            "r_align": r_align * 2.0,
            "r_smooth": r_smooth * 2.0,
            "r_progress": r_progress * 2.0,
            "r_stability": r_stability,
            "val_force": F_force_mag,
            "val_delta_psi": delta_psi
        }

        return total_reward

    def _normalize_angle(self, angle):
        """将角度标准化到 [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def _get_noisy_u1(self):
        """生成带有真实车辆特性的第一辆车控制量"""
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
        """环境一步交互"""
        original_action = self.denormalize_action(action)
        
        u1 = self._get_noisy_u1()
        u = np.concatenate([u1, original_action])
        
        self.model.step(u)
        observation = self._get_observation()  # 获取新版观测
        reward = self._calculate_reward()
        self._record_trajectories()

        if self.enable_visualization:
            self._record_trajectories()
            self._render_frame()  
            if self.render_mode == "human":
                plt.pause(0.001)  
        
        terminated = self.model.is_finish
        truncated = False
        X1, Y1 = self.model.getXYi(self.model.x, 0)
        X2, Y2 = self.model.getXYi(self.model.x, 1)  
        info = {
            "reward_r_force": np.array(self.reward_info.get("r_force", 0.0), dtype=np.float32),
            "reward_r_align": np.array(self.reward_info.get("r_align", 0.0), dtype=np.float32),
            "reward_r_smooth": np.array(self.reward_info.get("r_smooth", 0.0), dtype=np.float32),
            "reward_r_progress": np.array(self.reward_info.get("r_progress", 0.0), dtype=np.float32),
            "reward_r_stability": np.array(self.reward_info.get("r_stability", 0.0), dtype=np.float32),
            "reward_val_force": np.array(self.reward_info.get("val_force", 0.0), dtype=np.float32),
            "reward_val_delta_psi": np.array(self.reward_info.get("val_delta_psi", 0.0), dtype=np.float32),
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
        # ==================== [核心修改] 添加物理熔断机制 ====================
        # 1. 获取当前物理指标
        Fh2_x = self.model.Fh_arch[self.model.count, 2]
        Fh2_y = self.model.Fh_arch[self.model.count, 3]
        current_force = np.hypot(Fh2_x, Fh2_y)
        
        # 2. 定义阈值 (根据你的 yaml 配置，safe 是 2000，这里给个宽容度，比如 3倍或5倍)
        # 如果力超过 10,000N，说明已经没救了
        FORCE_TERMINATE_THRESHOLD = 10000.0 
        
        # 3. 判断是否提前结束
        terminated = self.model.is_finish # 原有的时间结束条件
        
        if current_force > FORCE_TERMINATE_THRESHOLD:
            terminated = True
            # 给予额外的“死亡惩罚”，让模型刻骨铭心
            reward -= 200.0 
            # 在 info 里记录一下是因为受力过大挂掉的
            info['termination_reason'] = 'force_limit'
        else:
            info['termination_reason'] = 'time_limit'

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, clear_frames=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
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
        if hasattr(self, '_prev_u1_noisy'):
            del self._prev_u1_noisy
        
        # Warmup for VecNorm
        if not self.vecnorm_frozen:
            warmup_steps = 5
            zero_action = np.zeros(4) 
            for _ in range(warmup_steps):
                u_zero = np.concatenate([self.u1_random, zero_action]) 
                self.model.step(u_zero)
                _ = self._get_observation()
            
            self.model.count = 0
            self.model.x = np.array([
                self.config['X_o_0'], self.config['Y_o_0'], self.config['Psi_o_0'],
                self.config['Psi_1_0'], self.config['Psi_2_0'],
                0, 0, 0, 0, 0 
            ], dtype=np.float64)
            self.model.x_arch[0, :] = self.model.x
            self.model.u_arch.fill(0) 
            self.model.Fh_arch.fill(0)
            if self.enable_visualization:
                self._reset_visualization()

        options = options or {}
        final_clear_frames = options.get(
            "clear_frames", 
            clear_frames if clear_frames is not None else False
        )
        
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
        cargo_pos = (x[0], x[1])
        self.trajectories['cargo'].append(cargo_pos)
        car1_pos = self.model.getXYi(x, 0)
        car2_pos = self.model.getXYi(x, 1)
        self.trajectories['car1'].append(car1_pos)
        self.trajectories['car2'].append(car2_pos)
        hinge1_pos = self.model.getXYhi(x, 0)
        hinge2_pos = self.model.getXYhi(x, 1)
        self.trajectories['hinge1'].append(hinge1_pos)
        self.trajectories['hinge2'].append(hinge2_pos)

    def _reset_visualization(self):
        if self.fig is not None:
            plt.close(self.fig)
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
    # 测试代码
    RENDER_MODE = "rgb_array"
    ENABLE_VISUALIZATION = True
    SEED = 42
    
    env = gym.make(
        "TwoCarrierEnv-v1",
        render_mode=RENDER_MODE,
        config_path=None,
        enable_visualization=ENABLE_VISUALIZATION
    )
    raw_env = env.unwrapped
    
    print("\n--- 【新版观测】后车坐标系/相对状态验证 ---")
    obs, info = env.reset(seed=SEED)
    print(f"观测维度：{obs.shape} (应为12维)")
    
    # 打印前几维的含义
    print(f"后车 X: {obs[0]:.2f} (VecNorm后)")
    print(f"后车 Y: {obs[1]:.2f}")
    print(f"后车 Psi: {obs[2]:.2f}")
    print(f"相对角度 Psi_o_2: {obs[3]:.2f}")
    print(f"相对角度 Psi_1_o: {obs[4]:.2f}")
    
    # 模拟一步
    normalized_action = np.array([0, 0, 0, 0])
    obs, reward, term, trunc, info = env.step(normalized_action)
    print("Step 1 完成")
    
    env.close()