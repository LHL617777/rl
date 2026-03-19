import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from model import Model2D2C

# ==========================================
# 动画辅助几何函数
# ==========================================
def get_box_corners(cx, cy, yaw, length, width):
    """根据中心点、偏航角和长宽计算矩形的四个绝对坐标顶点"""
    dx = length / 2
    dy = width / 2
    corners = np.array([
        [dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]
    ])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    return (R @ corners.T).T + np.array([cx, cy])

# ==========================================
# 高颜值动画播放引擎
# ==========================================
def play_animation(time_arr, x_arch, dt, scenario_name):
    print(f"🎬 正在渲染 [{scenario_name}] 的动力学动画...")
    
    # 几何与尺寸参数 (可根据实际模型微调视觉效果)
    L_car, W_car = 2.42, 1.2
    L_cargo, W_cargo = 6.0, 2.5
    x__o_1, x__o_2 = 3.0, -3.0
    x__1_1, x__2_2 = 0, 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 设置兼容中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    ax.set_xlabel('全局 X 坐标 (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('全局 Y 坐标 (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'超大件多车协同动力学姿态 - {scenario_name}', fontsize=14, fontweight='bold')

    # 初始化图形实体
    cargo_patch = Polygon(np.zeros((4, 2)), closed=True, fc='gray', ec='black', alpha=0.7, zorder=2)
    car1_patch = Polygon(np.zeros((4, 2)), closed=True, fc='#1f77b4', ec='black', alpha=0.9, zorder=3)
    car2_patch = Polygon(np.zeros((4, 2)), closed=True, fc='#d62728', ec='black', alpha=0.9, zorder=3)
    hinge_lines, = ax.plot([], [], 'ko-', lw=2.5, markersize=6, zorder=4)
    traj_line, = ax.plot([], [], 'k--', lw=1.5, alpha=0.5, zorder=1)

    ax.add_patch(cargo_patch)
    ax.add_patch(car1_patch)
    ax.add_patch(car2_patch)

    # 降采样播放设置（每隔 5 帧画一次，保证播放速度接近真实物理时间）
    play_speed = 5 
    frames_idx = np.arange(0, len(time_arr), play_speed)

    def update(frame):
        # 提取当前步的物理状态
        x_o, y_o, psi_o = x_arch[frame, 0], x_arch[frame, 1], x_arch[frame, 2]
        psi_1, psi_2 = x_arch[frame, 3], x_arch[frame, 4]
        
        # 1. 更新货物姿态
        cargo_patch.set_xy(get_box_corners(x_o, y_o, psi_o, L_cargo, W_cargo))
        
        # 2. 计算铰接点绝对坐标
        h1_x = x_o + np.cos(psi_o) * x__o_1
        h1_y = y_o + np.sin(psi_o) * x__o_1
        h2_x = x_o + np.cos(psi_o) * x__o_2
        h2_y = y_o + np.sin(psi_o) * x__o_2
        
        # 3. 推算前车与后车中心坐标
        c1_x = h1_x - np.cos(psi_1) * x__1_1
        c1_y = h1_y - np.sin(psi_1) * x__1_1
        c2_x = h2_x - np.cos(psi_2) * x__2_2
        c2_y = h2_y - np.sin(psi_2) * x__2_2
        
        # 4. 更新车辆姿态与连杆
        car1_patch.set_xy(get_box_corners(c1_x, c1_y, psi_1, L_car, W_car))
        car2_patch.set_xy(get_box_corners(c2_x, c2_y, psi_2, L_car, W_car))
        hinge_lines.set_data([c1_x, h1_x, x_o, h2_x, c2_x], [c1_y, h1_y, y_o, h2_y, c2_y])
        
        # 5. 更新历史轨迹 (货物中心)
        traj_line.set_data(x_arch[:frame, 0], x_arch[:frame, 1])
        
        # 6. 动态锁定视角 (无人机跟随相机)
        ax.set_xlim(x_o - 15, x_o + 25)
        ax.set_ylim(y_o - 20, y_o + 20)
        
        return cargo_patch, car1_patch, car2_patch, hinge_lines, traj_line

    ani = animation.FuncAnimation(fig, update, frames=frames_idx, interval=dt*1000*play_speed, blit=False)
    plt.show()

# ==========================================
# 核心仿真引擎
# ==========================================
def run_scenario(scenario_name, T_sim, u_func, animate=False):
    # 1. 动态获取配置路径并加载
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "2d2c.yaml")
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config['T'] = T_sim  
    config['integrator'] = 'RK4'
    
    # 2. 初始化自建动力学模型
    model = Model2D2C(config)
    steps = int(T_sim / config['dt'])
    time_arr = np.linspace(0, T_sim, steps + 1)
    
    print(f"\n🚀 正在运行 [{scenario_name}] 动力学解算...")
    for t in time_arr[:-1]:
        u = u_func(t)
        model.step(u)
        
    # 3. 提取内力并坐标转换
    Fh1_x_global = model.Fh_arch[:, 0]
    Fh1_y_global = model.Fh_arch[:, 1]
    Psi_1 = model.x_arch[:, 3] 
    # Fh1_x_global = model.Fh_arch[:, 2]
    # Fh1_y_global = model.Fh_arch[:, 3]
    # Psi_1 = model.x_arch[:, 4] 


    Fh1_x_local = Fh1_x_global * np.cos(Psi_1) + Fh1_y_global * np.sin(Psi_1)
    Fh1_y_local = -Fh1_x_global * np.sin(Psi_1) + Fh1_y_global * np.cos(Psi_1)
    
    x_history = np.array(model.x_arch[:len(time_arr)])

    # 4. 保存 CSV 对比数据
    df = pd.DataFrame({
        'Time': time_arr,
        'Fh1_x': Fh1_x_local,
        'Fh1_y': Fh1_y_local,
        'alpha1_f': model.alpha1_f_arch[:len(time_arr)], 
        'alpha1_r': model.alpha1_r_arch[:len(time_arr)],
        'Fy1_f': model.Fy1_f_arch[:len(time_arr)],
        'Fy1_r': model.Fy1_r_arch[:len(time_arr)],
        'alpha2_f': model.alpha2_f_arch[:len(time_arr)], 
        'alpha2_r': model.alpha2_r_arch[:len(time_arr)],
        'Fy2_f': model.Fy2_f_arch[:len(time_arr)],
        'Fy2_r': model.Fy2_r_arch[:len(time_arr)],
        
        # === 新增：各车状态量提取 ===
        'X_o': x_history[:, 0],       # 货物全局X坐标
        'Y_o': x_history[:, 1],      # 货物全局Y坐标
        'Psi_o': x_history[:, 2],     # 货物航向角 (rad)
        'Psi_1': x_history[:, 3],     # 前车航向角 (rad)
        'Psi_2': x_history[:, 4],      # 后车航向角 (rad)
        'X_dot_o': x_history[:, 5],       # 货物全局X方向速度 (m/s)
        'Y_dot_o': x_history[:, 6],       # 货物全局Y方向速度 (m/s)
        'omega_o': x_history[:, 7],   # 货物横摆角速度 (rad/s)
        'omega_1': x_history[:, 8],   # 前车横摆角速度 (rad/s)
        'omega_2': x_history[:, 9]   # 后车横摆角速度 (rad/s)
    })

    # === 新增：自动计算并导出相对折叠角 (可选，写论文画图极其方便) ===
    # 折叠角 1 = 前车航向角 - 货物航向角
    # 折叠角 2 = 后车航向角 - 货物航向角
    df['gamma_1'] = df['Psi_1'] - df['Psi_o']
    df['gamma_2'] = df['Psi_2'] - df['Psi_o']

    filename = os.path.join(current_dir, f"python_{scenario_name}.csv")
    df.to_csv(filename, index=False)
    print(f"✅ 数据已成功保存至: {filename}")
    # 5. 触发动画
    if animate:
        play_animation(time_arr, model.x_arch, config['dt'], scenario_name)

# ==========================================
# 仿真主入口
# ==========================================
if __name__ == "__main__":
    
    # # --- 工况 A：纯纵向阶跃 ---
    # def u_longitudinal(t):
    #     return [0.0, 0.0, 2000.0, 2000.0, 0.0, 0.0, 0.0, 0.0]
    
    # # 工况 A 默认只保存数据，不弹动画 (animate=False)
    # run_scenario("longitudinal", 10.0, u_longitudinal, animate=False)

    # # --- 工况 B：加速后滑行转向 ---
    # def u_lateral(t):
    #     if t < 5.0:
    #         delta, thrust = 0.0, 2000.0
    #     else:
    #         delta, thrust = 0.1, 0.0
        
    #     return [delta, 0.0, thrust, thrust, 0.0, 0.0, 0.0, 0.0]
    
    # # 工况 B 跑完数据后，立即播放酷炫动画 (animate=True)
    
    # run_scenario("lateral", 15.0, u_lateral, animate=True)

    def get_u(t, scenario='sine'):
        u = np.zeros(8)  # [delta_f1, delta_r1, T_f1, T_r1, delta_f2, delta_r2, T_f2, T_r2]
        
        # 基础纵向推力 (假设不用 PID，硬给推力)
        base_thrust = 500.0
        u[2] = base_thrust / 2; u[3] = base_thrust / 2
        u[6] = base_thrust / 2; u[7] = base_thrust / 2
        
        if scenario == 'sine':
            # 工况一：正弦转向 (5秒后开始，周期4秒)
            if t > 4.0:
                u[0] = 0.2 * np.sin(2 * np.pi * (t - 4.0) / 6.0)
                
        elif scenario == 'pivot':
            # 工况二：协同枢轴转向
            if t > 4.0:
                u[0] = 0.15    # 前车左打
                u[4] = -0.15  # 后车右打
                
        elif scenario == 'trail_braking':
            # 工况三：极限循迹刹车
            if t < 5.0:
                u[2] = 3000; u[3] = 3000  # 猛烈加速
            else:
                u[0] = 0.3                # 猛打方向
                u[2] = -6000; u[3] = -6000 # 猛烈刹车 (抱死边缘)
                
        elif scenario == 'u_turn':
            # 工况四：持续大角度转向掉头
            if t > 3.0:
                u[0] = 0.4  # 前车前轮持续大角度
                
        return u.tolist()
    run_scenario("pivot", 10.0, lambda t: get_u(t, 'pivot'), animate=True)


