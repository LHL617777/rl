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

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

class TwoCarrierEnv(gym.Env):
    """两辆车运载超大件系统的自定义强化学习环境"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, config_path=None, enable_visualization=False):
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
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float64
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
        # print("动作空间已归一化至[-1, 1]区间")
        # print(f"原始动作范围：\n  转向角：±{np.pi/6:.2f}rad（±30°），推力：[0, {1e3}]")
        # print(f"归一化后动作范围：[-1, 1]（4维）")
        
        # # 第一辆车的固定控制量（简化问题，仅训练第二辆车）
        # self.u1_fixed = np.array([np.pi/6, 0, 1e3, 1e3])  # 示例：固定前轮转角和推力
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
        # print(f"初始化渲染模式：{self.render_mode}，是否为rgb_array：{self.render_mode == 'rgb_array'}")
        # print(f"可视化功能：{'启用' if enable_visualization else '关闭'}")

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
            # print(f"成功加载YAML配置文件：{config_path}")
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

    def _get_observation(self):
        """从模型提取观测值"""
        x = self.model.x  # 当前状态向量
        # 提取第二辆车的铰接力（Fh_arch存储格式：[Fh1_x, Fh1_y, Fh2_x, Fh2_y]）
        Fh2_x = self.model.Fh_arch[self.model.count, 2]
        Fh2_y = self.model.Fh_arch[self.model.count, 3]
        return np.concatenate([
            x[:5],   # 位置与姿态：[Xo, Yo, Psio, Psi1, Psi2]
            x[5:10], # 速度：[Xo_dot, Yo_dot, Psio_dot, Psi1_dot, Psi2_dot]
            [Fh2_x, Fh2_y]  # 第二辆车铰接力
        ])

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
            control_smooth_penalty = 0.1 * np.sum((u2_current - u2_prev)**2)
        else:
            control_smooth_penalty = 0
        
        # 总奖励 = 负惩罚（最小化目标）
        return - (hinge_force_penalty + 0 * tracking_penalty + 0 * control_smooth_penalty)
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
        # 2.4 裁剪约束（不超出车辆最大转向角，符合真实物理特性）
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
        observation = self._get_observation()
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
        info = {
            'Fh2': (self.model.Fh_arch[self.model.count, 2], 
                    self.model.Fh_arch[self.model.count, 3]),
            'pos_error': np.hypot(
                self.model.getXYi(self.model.x, 1)[0] - self.model.getXYi(self.model.x, 0)[0],
                self.model.getXYi(self.model.x, 1)[1] - self.model.getXYi(self.model.x, 0)[1]
            ),
            'u1': u1,  # 新增：保存第一辆车控制量，用于可视化车轮摆角
            'u2_normalized': action,  # 归一化后的动作
            'u2_original': original_action  # 原始动作（便于调试）
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, clear_frames=True):
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
            del self._prev_u1_noisy # 重置第一辆车扰动历史
        if self.enable_visualization:
            if clear_frames and not self.is_sim_finished:
                self.render_frames = []
            self.trajectories = {   # 重置轨迹
                'cargo': [],
                'car1': [],
                'car2': [],
                'hinge1': [],
                'hinge2': []
            }
            self._reset_visualization()  # 仅在启用时初始化可视化
        observation = self._get_observation()
        return observation, {}

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
        self.fig, self.ax = plt.subplots(figsize=(15, 15), dpi=80)
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
        """重构帧渲染，完全遵循参考代码的坐标变换与可视化逻辑"""
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
                    zorder=2.3,
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

        # 7. rgb_array模式下保存帧（保持原有逻辑）
        if self.render_mode == "rgb_array":
            try:
                buf = BytesIO()
                self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=self.fig.get_facecolor())
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
                frame = np.array(img)
                # 校验帧尺寸一致性
                if len(self.render_frames) > 0:
                    ref_shape = self.render_frames[0].shape
                    if frame.shape != ref_shape:
                        frame = cv2.resize(frame, (ref_shape[1], ref_shape[0]))
                self.render_frames.append(frame)
                buf.close()
                return frame
            except Exception as e:
                print(f"帧保存失败，错误：{type(e).__name__}: {e}")

    def close(self):
        """关闭环境并生成视频"""
        # 新增：打印关键状态，确认帧列表和渲染模式
        # print(f"===== 进入close()方法 ======")
        # print(f"当前render_mode：{self.render_mode}")
        # print(f"可视化功能状态：{'启用' if self.enable_visualization else '关闭'}")
        
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
                time_str = datetime.datetime.now().strftime(r'%y%m%d%H%M%S')  # 修正时间格式，避免重复
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
    entry_point=TwoCarrierEnv,
    max_episode_steps=1000,
    kwargs={"enable_visualization": False}  # 添加默认参数
)

# 测试代码
if __name__ == "__main__":
    # ===================== 配置调整（便于测试扰动功能） =====================
    RENDER_MODE = "rgb_array"  # 可选"human"（可视化）或"rgb_array"（保存视频）
    ENABLE_VISUALIZATION = False  # 测试扰动时建议先关闭，避免卡顿；验证可视化时再开启
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

    # ===================== 步骤2：新增【前轮转角多样性扰动专属测试】（核心修改） =====================
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
    print(f"  取值范围：[{steer_records.min():.4f}, {steer_records.max():.4f}]rad")
    print(f"  对应角度：[{np.rad2deg(steer_records.min()):.1f}, {np.rad2deg(steer_records.max()):.1f}]°")
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

    # ===================== 步骤3：（可选）多轮仿真验证重置有效性 =====================
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

    # ===================== 步骤4：保留原有环境关闭+视频生成逻辑 =====================
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