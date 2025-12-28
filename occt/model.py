import os
import time
import glob
import subprocess
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection


class ModelBase:
    """Base class for all models.
    It stores general visualization methods and state conversion methods.
    """

    def __init__(self) -> None:
        """Initialize.
        """
        self.N_c = 0  # 载体数量（2/3，子类赋值）
        self.N_q = 0  # 广义坐标数（子类赋值）
        self.N_x = 0  # 状态向量维度（N_x=2*N_q，位置+速度）
        self.N_u = 0  # 控制输入维度
        self.config = dict()  # 仿真配置（帧率、范围、颜色等）
        # 几何参数：载体/铰链相对于主载体的局部坐标
        self.x__o_i, self.y__o_i = list(), list()  # 铰链相对于主载体的局部坐标
        self.x__i_i, self.y__i_i = list(), list()  # 载体质心相对于自身铰链的局部坐标
        self.l_f, self.l_r = 0.0, 0.0  # 载体前后轮到质心的距离
        self.dt = 0.0  # 仿真步长
        # 轨迹存储：状态x、控制u、铰链力Fh
        self.N = 0  # 总仿真步数
        self.x_arch = np.array([0.0])  # 状态轨迹 (N+1, N_x)
        self.u_arch = np.array([0.0])  # 控制输入轨迹 (N+1, N_u)
        self.Fh_arch = np.array([0.0])  # 铰链力轨迹 (N+1, 2*N_c)

    def generateVideo(
            self,
            dir_output: str,
            name_video: str,
    ) -> None:
        """Generate a video using the simulation results.

        Parameters
        ----------
        dir_output: str
            the directory to output.
        name_video: str
            the file name of the video.
        """
        start_time = time.time()
        fig, ax = plt.subplots(figsize=(15, 15), dpi=80)
        ax.set_xlabel('X (m)', fontsize=20)
        ax.set_ylabel('Y (m)', fontsize=20)
        ax.set_aspect('equal', adjustable='box')

        # Clear directory
        dir_current = os.getcwd()
        os.chdir(dir_output)
        for f in glob.glob("*.png"):
            os.remove(f)

        # Iterative plot
        dt_frame = 1 / self.config['framerate'] # 计算每帧视频对应的时间间隔
        total_frames = round(self.T / dt_frame) + 1  # 新增：总帧数（用于日志）
        for i_frame in range(total_frames):
            # Data for visualization
            i = round(i_frame * dt_frame / self.dt)
            # 1 轮胎可视化数据
            segments_tire = self.getTireVis(i)
            # 2 获取铰链力和货物可视化数据。
            # 铰链力箭头数据和货物多边形数据
            arrows_Fh, hinge_markers = self.getHingeVis(i)
            polygon_o = self.getOversizedCargoVis(i) # 超大货物多边形数据
            # 3 载体多边形数据
            polygons_c = self.getCarrierVis(i)

            # Draw and save 
            if i == 0:  # 初始帧需创建绘图元素，后续帧仅更新数据
                # 1 handle_tire：轮胎绘图句柄
                handle_tire = LineCollection(segments_tire,
                                            colors=self.config['c_tire'], linewidths=self.config['lw_tire'])
                ax.add_collection(handle_tire)
                # 2 handle_Fh and handle_hinge：铰链力箭头和铰链位置绘图句柄
                handle_Fh = []
                handle_hinge = []
                for arrow in arrows_Fh:
                    handle_Fh.append(ax.arrow(*arrow, zorder=2.4,
                                            width=0, color=self.config['c_Fh']))
                for marker_poly in hinge_markers:
                    h = Polygon(marker_poly, zorder=2.6, alpha=1.0, fc='black', ec='white')
                    ax.add_patch(h)
                    handle_hinge.append(h)
                # 3 handle_o: 货物绘图句柄
                if polygon_o:
                    handle_o = Polygon(polygon_o, zorder=2.5,
                                    alpha=self.config['alpha_o'], fc=self.config['fc_o'])
                    ax.add_patch(handle_o)
                # 4 handle_c：载体绘图句柄
                handle_c = []
                for polygon in polygons_c:
                    handle_c.append(Polygon(polygon, zorder=2.3,
                                            alpha=self.config['alpha_c'], fc=self.config['fc_c']))
                    ax.add_patch(handle_c[-1])
            else:
                # 1
                handle_tire.set_segments(segments_tire)
                # 2
                for h_Fh, arrow, h_hinge in zip(handle_Fh, arrows_Fh, handle_hinge):
                    h_Fh.set_data(x=arrow[0], y=arrow[1], dx=arrow[2], dy=arrow[3],
                                width=self.config['width_Fh'])
                for h_hinge, marker_poly in zip(handle_hinge, hinge_markers):
                    h_hinge.set_xy(marker_poly)
                # 3
                if polygon_o:
                    handle_o.set_xy(polygon_o)
                else:
                    handle_o.set_xy([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])  # 移出视野
                # 4
                for h_c, polygon_c in zip(handle_c, polygons_c):
                    h_c.set_xy(polygon_c)
            X_o = self.x_arch[i, 0]
            Y_o = self.x_arch[i, 1]
            ax.set_xlim([X_o - self.config['range'], X_o + self.config['range']])
            ax.set_ylim([Y_o - self.config['range'], Y_o + self.config['range']])
            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.savefig(os.path.join(dir_output, 'vis%04d.png' % i_frame))

        # Generate video（核心修改：添加异常处理+自动删图）
        try:
            # 调用ffmpeg生成视频，新增-y参数自动覆盖已有视频
            subprocess.call([
                'ffmpeg', '-framerate', str(self.config['framerate']),
                '-i', 'vis%04d.png', '-pix_fmt', 'yuv420p', '-y',  # 新增：-y 参数
                name_video
            ])  # ffmpeg -framerate 10 -i vis%04d.png -pix_fmt yuv420p vis.mp4
            
            # ========== 新增：生成视频后删除所有临时PNG图片 ==========
            print(f"[INFO] 开始清理临时PNG图片（目录：{dir_output}）")
            png_files = glob.glob("vis*.png")  # 精准匹配生成的图片，避免误删其他PNG
            for f in png_files:
                try:
                    os.remove(f)
                    print(f"[INFO] 已删除临时文件：{f}")
                except Exception as e:
                    print(f"[WARNING] 删除文件 {f} 失败：{str(e)}")
            # =======================================================
            
        except Exception as e:
            print(f"[ERROR] 生成视频失败：{str(e)}")
        finally:
            # 无论视频生成是否成功，都切回原目录
            os.chdir(dir_current)
            # 新增：关闭绘图窗口，释放内存/文件句柄（避免图片被占用删不掉）
            plt.close(fig)

        print('[INFO] video takes: {:.2f} s'.format(time.time() - start_time))
    def getXYhi(
            self,
            x: npt.NDArray,
            i: int,
    ) -> tuple[float, float]:
        """Get X and Y coordinates of the i-th hinge.

        Parameters
        ----------
        x: npt.NDArray
            state variables.
        i: int
            index of the hinge.

        Returns
        ----------
        Xh_i and Yh_i.
        """
        X_o = x[0]
        Y_o = x[1]
        Psi_o = x[2]

        Xh_i = X_o \
               + (np.cos(Psi_o) * self.x__o_i[i] - np.sin(Psi_o) * self.y__o_i[i])
        Yh_i = Y_o \
               + (np.sin(Psi_o) * self.x__o_i[i] + np.cos(Psi_o) * self.y__o_i[i])
        return Xh_i, Yh_i

    def getXYi(
            self,
            x: npt.NDArray,
            i: int,
    ) -> tuple[float, float]:
        """Get X and Y coordinates of the i-th carrier.

        Parameters
        ----------
        x: npt.NDArray
            state variables.
        i: int
            index of the carrier.

        Returns
        ----------
        X_i and Y_i.
        """
        X_o = x[0]
        Y_o = x[1]
        Psi_o = x[2]
        Psi_i = x[3 + i]

        X_i = X_o \
              + (np.cos(Psi_o) * self.x__o_i[i] - np.sin(Psi_o) * self.y__o_i[i]) \
              - (np.cos(Psi_i) * self.x__i_i[i] - np.sin(Psi_i) * self.y__i_i[i])
        Y_i = Y_o \
              + (np.sin(Psi_o) * self.x__o_i[i] + np.cos(Psi_o) * self.y__o_i[i]) \
              - (np.sin(Psi_i) * self.x__i_i[i] + np.cos(Psi_i) * self.y__i_i[i])
        return X_i, Y_i

    def getXYdoti(
            self,
            x: npt.NDArray,
            i: int,
    ) -> tuple[float, float]:
        """Get X and Y velocity of the i-th carrier.
        计算第i个载体质心的全局 X/Y 速度
        Parameters
        ----------
        x: npt.NDArray
            state variables.
        i: int
            index of the carrier.

        Returns
        ----------
        X_dot_i and Y_dot_i.
        """
        Psi_o = x[2]
        Psi_i = x[3 + i]
        X_dot_o = x[self.N_q + 0]
        Y_dot_o = x[self.N_q + 1]
        Psi_dot_o = x[self.N_q + 2]
        Psi_dot_i = x[self.N_q + 3 + i]

        X_dot_i = X_dot_o \
                  + Psi_dot_o * (- np.sin(Psi_o) * self.x__o_i[i] - np.cos(Psi_o) * self.y__o_i[i]) \
                  - Psi_dot_i * (- np.sin(Psi_i) * self.x__i_i[i] - np.cos(Psi_i) * self.y__i_i[i])
        Y_dot_i = Y_dot_o \
                  + Psi_dot_o * (np.cos(Psi_o) * self.x__o_i[i] - np.sin(Psi_o) * self.y__o_i[i]) \
                  - Psi_dot_i * (np.cos(Psi_i) * self.x__i_i[i] - np.sin(Psi_i) * self.y__i_i[i])
        return X_dot_i, Y_dot_i

    def getXYddoti(
            self,
            x: npt.NDArray,
            x_dot: npt.NDArray,
            i: int,
    ) -> tuple[float, float]:
        """Get X and Y acceleration of the i-th carrier.

        Parameters
        ----------
        x: npt.NDArray
            state variables.
        x_dot: npt.NDArray
            derivative of the state vector.
        i: int
            index of the carrier.

        Returns
        ----------
        X_ddot_i and Y_ddot_i.
        """
        Psi_o = x[2]
        Psi_i = x[3 + i]
        Psi_dot_o = x[self.N_q + 2]
        Psi_dot_i = x[self.N_q + 3 + i]
        X_ddot_o = x_dot[self.N_q + 0]
        Y_ddot_o = x_dot[self.N_q + 1]
        Psi_ddot_o = x_dot[self.N_q + 2]
        Psi_ddot_i = x_dot[self.N_q + 3 + i]

        X_ddot_i = X_ddot_o \
                   + Psi_ddot_o * (- np.sin(Psi_o) * self.x__o_i[i] - np.cos(Psi_o) * self.y__o_i[i]) \
                   + Psi_dot_o ** 2 * (- np.cos(Psi_o) * self.x__o_i[i] + np.sin(Psi_o) * self.y__o_i[i]) \
                   - Psi_ddot_i * (- np.sin(Psi_i) * self.x__i_i[i] - np.cos(Psi_i) * self.y__i_i[i]) \
                   - Psi_dot_i ** 2 * (- np.cos(Psi_i) * self.x__i_i[i] + np.sin(Psi_i) * self.y__i_i[i])
        Y_ddot_i = Y_ddot_o \
                   + Psi_ddot_o * (np.cos(Psi_o) * self.x__o_i[i] - np.sin(Psi_o) * self.y__o_i[i]) \
                   + Psi_dot_o ** 2 * (- np.sin(Psi_o) * self.x__o_i[i] - np.cos(Psi_o) * self.y__o_i[i]) \
                   - Psi_ddot_i * (np.cos(Psi_i) * self.x__i_i[i] - np.sin(Psi_i) * self.y__i_i[i]) \
                   - Psi_dot_i ** 2 * (- np.sin(Psi_i) * self.x__i_i[i] - np.cos(Psi_i) * self.y__i_i[i])
        return X_ddot_i, Y_ddot_i

    def getAtf2vi(
            self,
            u: npt.NDArray,
            i: int,
    ) -> npt.NDArray:
        """Get the homogeneous transformation matrix to transform 
        a vector in front tire coordinate system to vehicle coordinate system.
        前轮坐标系→载体坐标系变换
        Parameters
        ----------
        u: npt.NDArray
            input variables.
        i: int
            index of the carrier.

        Returns
        ----------
        The homogeneous transformation matrix.
        """
        delta_f_i = u[0 + 4 * i]
        return np.array([
            [np.cos(delta_f_i), -np.sin(delta_f_i), self.l_f],
            [np.sin(delta_f_i), np.cos(delta_f_i), 0],
            [0, 0, 1],
        ])

    def getAtr2vi(
            self,
            u: npt.NDArray,
            i: int,
    ) -> npt.NDArray:
        """Get the homogeneous transformation matrix to transform 
        a vector in rear tire coordinate system to vehicle coordinate system.
        后轮坐标系→载体坐标系变换
        Parameters
        ----------
        u: npt.NDArray
            input variables.
        i: int
            index of the carrier.

        Returns
        ----------
        The homogeneous transformation matrix.
        """
        delta_r_i = u[1 + 4 * i]
        return np.array([
            [np.cos(delta_r_i), -np.sin(delta_r_i), -self.l_r],
            [np.sin(delta_r_i), np.cos(delta_r_i), 0],
            [0, 0, 1],
        ])

    def getAv2gi(
            self,
            x: npt.NDArray,
            i: int,
    ) -> npt.NDArray:
        """Get the homogeneous transformation matrix to transform 
        a vector in vehicle coordinate system to global coordinate system.
        载体坐标系→全局坐标系变换
        Parameters
        ----------
        x: npt.NDArray
            state variables.
        i: int
            index of the carrier.

        Returns
        ----------
        The homogeneous transformation matrix.
        """
        Psi_i = x[3 + i]
        X_i, Y_i = self.getXYi(x, i)
        return np.array([
            [np.cos(Psi_i), -np.sin(Psi_i), X_i],
            [np.sin(Psi_i), np.cos(Psi_i), Y_i],
            [0, 0, 1],
        ])

    def getTireVis(
            self,
            i_sim: int,
    ) -> list[list]:
        """Get the tire visualization data at step i_sim.

        Parameters
        ----------
        i_sim: int
            index of simulation step.

        Returns
        ----------
        Line segments formatted as [[[x1, y1], [x2, y2]], ...].
        """
        x = self.x_arch[i_sim, :]
        u = self.u_arch[i_sim, :]
        line = np.array([
            [self.config['R'], 0, 1],
            [-self.config['R'], 0, 1],
        ]).T

        segments = []
        for i in range(self.N_c):
            Av2gi = self.getAv2gi(x, i)
            line_f = Av2gi @ self.getAtf2vi(u, i) @ line
            line_r = Av2gi @ self.getAtr2vi(u, i) @ line

            segments.append(line_f[0:2, :].T.tolist())
            segments.append(line_r[0:2, :].T.tolist())

        return segments

    def getHingeVis(
            self,
            i_sim: int,
    ) -> tuple[list[list], list[list]]:
        """仅生成铰接力箭头和铰接点标记的可视化数据（职责单一）。

        Parameters
        ----------
        i_sim: int
            仿真步索引。

        Returns
        -------
        arrows: list[list]
            铰接力箭头数据，每个元素为 [x, y, dx, dy]（起点+方向长度）。
        hinge_markers: list[list]
            铰接点标记数据，每个铰接点对应一个小正方形多边形。
        """
        x = self.x_arch[i_sim, :]

        arrows = []
        hinge_markers = []


        # 读取配置参数（带默认值，避免报错）
        scale_Fh = self.config.get('scale_Fh', 0.0003)       # 力缩放因子
        hinge_size = self.config.get('hinge_size', 0.2)     # 铰接点标记大小（默认0.2m）

        # 遍历所有铰链，生成箭头和标记
        for i in range(self.N_c):
            # 1. 获取铰接点全局坐标
            Xh_i, Yh_i = self.getXYhi(x, i)

            # 2. 生成铰接力箭头数据
            Fh_x = self.Fh_arch[i_sim, 2 * i]
            Fh_y = self.Fh_arch[i_sim, 2 * i + 1]
            arrows.append([
                Xh_i,
                Yh_i,
                Fh_x * scale_Fh,
                Fh_y * scale_Fh,
            ])

            # 3. 生成铰接点标记（小正方形，中心在铰接点）
            marker_poly = [
                [Xh_i - hinge_size/2, Yh_i - hinge_size/2],  # 左下
                [Xh_i + hinge_size/2, Yh_i - hinge_size/2],  # 右下
                [Xh_i + hinge_size/2, Yh_i + hinge_size/2],  # 右上
                [Xh_i - hinge_size/2, Yh_i + hinge_size/2],  # 左上
            ]
            hinge_markers.append(marker_poly)

        return arrows, hinge_markers

    def getOversizedCargoVis(
            self,
            i_sim: int,
    ) -> list[list]:
        """专门生成超大货物的可视化多边形（仅双载体生效）。

        核心逻辑：
        - 仅当 self.N_c == 2 时生成货物
        - 货物长度 = 两个铰链点间距 + config['oversized_cargo_bias']
        - 货物宽度 = config['oversized_cargo_width']
        - 货物中心 = 两个铰链中点，方向 = 铰链连线方向

        Parameters
        ----------
        i_sim: int
            仿真步索引。

        Returns
        -------
        cargo_polygon: list[list]
            货物4顶点的全局坐标列表([[x1,y1], [x2,y2], ...]),
            非双载体时返回空列表。
        """
        # 仅双载体生成货物，否则返回空
        if self.N_c != 2:
            # 仅警告一次，避免干扰输出
            if not hasattr(self, '_cargo_vis_warned'):
                print(f"[WARNING] 超大货物仅支持双载体(当前N_c={self.N_c})，未生成货物")
                self._cargo_vis_warned = True
            return []

        # 1. 获取当前仿真步的状态和两个铰链坐标
        x = self.x_arch[i_sim, :]
        Xh0, Yh0 = self.getXYhi(x, 0)  # 第一个铰链
        Xh1, Yh1 = self.getXYhi(x, 1)  # 第二个铰链

        # 2. 从config读取货物参数（带默认值，避免KeyError）
        cargo_bias = self.config.get('oversized_cargo_bias', 1.0)
        cargo_width = self.config.get('oversized_cargo_width', 2.0)

        # 3. 计算货物关键参数（长度、中心、方向）
        hinge_dist = np.hypot(Xh1 - Xh0, Yh1 - Yh0)  # 铰链间距
        cargo_length = hinge_dist + cargo_bias        # 货物长度
        cargo_center_x = (Xh0 + Xh1) / 2.0           # 货物中心X
        cargo_center_y = (Yh0 + Yh1) / 2.0           # 货物中心Y
        cargo_angle = np.arctan2(Yh1 - Yh0, Xh1 - Xh0)  # 货物方向角

        # 4. 生成货物局部坐标→全局坐标（旋转+平移）
        cargo_local = np.array([
            [cargo_length/2,  cargo_width/2],  # 右上
            [cargo_length/2, -cargo_width/2],  # 右下
            [-cargo_length/2, -cargo_width/2], # 左下
            [-cargo_length/2,  cargo_width/2], # 左上
        ])
        # 旋转矩阵
        cos_ang = np.cos(cargo_angle)
        sin_ang = np.sin(cargo_angle)
        rotation_mat = np.array([[cos_ang, -sin_ang], [sin_ang, cos_ang]])
        # 旋转+平移到全局坐标
        cargo_global = np.dot(cargo_local, rotation_mat.T)
        cargo_global[:, 0] += cargo_center_x
        cargo_global[:, 1] += cargo_center_y

        return cargo_global.tolist()  # 转换为绘图所需格式
    def getCarrierVis(
            self,
            i_sim: int,
    ) -> list[list]:
        """Get the carrier polygon visualization data at step i_sim.

        Parameters
        ----------
        i_sim: int
            index of simulation step.

        Returns
        ----------
        Polygon formatted as [[x1, y1], ...].
        """
        x = self.x_arch[i_sim, :]
        polygon_raw = np.array([
            [self.config['fe_c'], self.config['w_c'] / 2, 1],
            [self.config['fe_c'], -self.config['w_c'] / 2, 1],
            [-self.config['re_c'], -self.config['w_c'] / 2, 1],
            [-self.config['re_c'], self.config['w_c'] / 2, 1],
        ]).T

        polygons = []
        for i in range(self.N_c):
            polygon = self.getAv2gi(x, i) @ polygon_raw
            polygons.append(polygon[0:2, :].T.tolist())

        return polygons

class Model2D2C(ModelBase):
    """Planar (two-dimensional) two-carrier dynamics model.
    """

    def __init__(
            self,
            config: dict,
    ) -> None:
        """Initialize with configurations.

        Parameters
        ----------
        config: dict
            dictionary containing configurations.
        """
        self.config = config

        # Numbers 维度参数
        self.N_c = config['N_c']
        self.N_q = config['N_q']
        self.N_x = config['N_x']
        self.N_u = config['N_u']

        # Parameters 模型参数
        self.M_o = config['M_o']
        self.I_o = config['I_o']
        self.M_1 = config['M_1']
        self.M_2 = config['M_2']
        self.M_i = [self.M_1, self.M_2]
        self.I_1 = config['I_1']
        self.I_2 = config['I_2']
        self.x__o_1 = config['x__o_1']
        self.x__o_2 = config['x__o_2']
        self.y__o_1 = config['y__o_1']
        self.y__o_2 = config['y__o_2']
        self.x__1_1 = config['x__1_1']
        self.x__2_2 = config['x__2_2']
        self.y__1_1 = config['y__1_1']
        self.y__2_2 = config['y__2_2']
        self.x__o_i = [self.x__o_1, self.x__o_2]
        self.y__o_i = [self.y__o_1, self.y__o_2]
        self.x__i_i = [self.x__1_1, self.x__2_2]
        self.y__i_i = [self.y__1_1, self.y__2_2]
        self.C_f = config['C_f']
        self.C_r = config['C_r']
        self.l_f = config['l_f']
        self.l_r = config['l_r']

        # Initial values 初始值
        self.x = np.array([
            config['X_o_0'],
            config['Y_o_0'],
            config['Psi_o_0'],
            config['Psi_1_0'],
            config['Psi_2_0'],
            config['X_dot_o_0'],
            config['Y_dot_o_0'],
            config['Psi_dot_o_0'],
            config['Psi_dot_1_0'],
            config['Psi_dot_2_0'],
        ], dtype=np.float64)

        # Simulation settings 仿真设置
        self.T = config['T']  # overall time
        self.dt = config['dt']
        if config['integrator'] == 'EE':
            self.integrator = self.EE
        elif config['integrator'] == 'RK4':
            self.integrator = self.RK4
        

        # Processing variables 处理变量
        self.N = round(self.T / self.dt)
        self.count = 0
        self.is_finish = False
        self.x_arch = np.zeros((self.N + 1, self.N_x))
        self.x_arch[0, :] = self.x
        self.u_arch = np.zeros((self.N + 1, self.N_u))
        self.Fx_i = [0] * self.N_c
        self.Fy_i = [0] * self.N_c
        self.Fh_arch = np.zeros((self.N + 1, 2 * self.N_c))  # hinge forces

    def step(
            self,
            u: list,
    ) -> None:
        """One step forward in the simulation.

        Parameters
        ----------
        u: list
            list containing control inputs.
        """
        if self.count == 0: self.sim_start_time = time.time()
        self.count += 1
        self.t = self.count * self.dt
        # if self.t % 1 == 0: print('[INFO] sim time:', self.t, 's')

        self.integrator(u)

        self.x_arch[self.count, :] = self.x
        self.u_arch[self.count, :] = np.array(u)

        if self.count == self.N:
            self.is_finish = True
            print('[INFO] sim takes: {:.2f} s'.format(time.time() - self.sim_start_time))

    def EE(
            self,
            u: list,
    ) -> None:
        """Explicit Euler method.

        Parameters
        ----------
        u: list
            list containing control inputs.
        """
        x_dot = self.f(u)
        self.storeHingeForces(x_dot)
        self.x += x_dot * self.dt

    def RK4(
            self,
            u: list,
    ) -> None:
        """4-th order Runge-Kutta method.

        Parameters
        ----------
        u: list
            list containing control inputs.
        """
        x = self.x.copy()

        k1 = self.f(u)
        self.storeHingeForces(k1)
        self.x = x + self.dt * k1 / 2
        k2 = self.f(u)
        self.x = x + self.dt * k2 / 2
        k3 = self.f(u)
        self.x = x + self.dt * k3
        k4 = self.f(u)

        self.x = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def f(
            self,
            u: list,
    ) -> npt.NDArray:
        """System equations of motion.
        状态导数计算:求解拉格朗日方程B*q_ddot + n = xi,得到q_ddot = B⁻¹*(xi - n)
        Parameters
        ----------
        u: list
            list containing control inputs.

        Returns
        ----------
        The derivative of the state vector.
        """
        q_ddot = np.linalg.solve(self.getB(), self.getxi(u) - self.getn())
        q_dot = self.x[self.N_q:]
        return np.append(q_dot, q_ddot)

    def storeHingeForces(
            self,
            x_dot: npt.NDArray,
    ) -> None:
        """Calculate and store the hinge forces in self.Fh_arch.
        The hinge forces are acted upon the oversized cargo.
        计算每个铰链的力,并存储到Fh_arch中
        Parameters
        ----------
        x_dot: npt.NDArray
            derivative of the state vector.
        """
        for i in range(self.N_c):
            X_ddot_i, Y_ddot_i = self.getXYddoti(self.x, x_dot, i)

            self.Fh_arch[self.count, 2 * i] = - (self.M_i[i] * X_ddot_i - self.Fx_i[i])
            self.Fh_arch[self.count, 2 * i + 1] = - (self.M_i[i] * Y_ddot_i - self.Fy_i[i])

    def getxi(
            self,
            u: list,
    ) -> npt.NDArray:
        """Calculate the generalized force vector.
        计算拉格朗日方程中的广义力向量xi
        Parameters
        ----------
        u: list
            list containing control inputs.

        Returns
        ----------
        The xi vector.
        """
        xi = np.zeros(self.N_q)

        for i in range(self.N_c):
            Psi_i = self.x[3 + i]
            # Psi_dot_i = self.x[self.N_q + 3 + i]
            Psi_dot_i = self.x[self.N_q + 3 + i]
            # Steering angle
            delta_f_i = u[0 + 4 * i]
            delta_r_i = u[1 + 4 * i]
            # longitudinal thrust force in wheel coordinate frame
            T_f_i = u[2 + 4 * i]
            T_r_i = u[3 + 4 * i]

            X_dot_i, Y_dot_i = self.getXYdoti(self.x, i)

            vx_i = np.cos(Psi_i) * X_dot_i + np.sin(Psi_i) * Y_dot_i
            vy_i = - np.sin(Psi_i) * X_dot_i + np.cos(Psi_i) * Y_dot_i
            vy_f_i = vy_i + self.l_f * Psi_dot_i
            vy_r_i = vy_i - self.l_r * Psi_dot_i
            alpha_f_i = np.atan2(vy_f_i, np.fabs(vx_i)) - delta_f_i
            alpha_r_i = np.atan2(vy_r_i, np.fabs(vx_i)) - delta_r_i

            # Forces on front and rear axles in vehicle coordinate frame
            Fx_f_i = np.cos(delta_f_i) * T_f_i - np.sin(delta_f_i) * (self.C_f * alpha_f_i)
            Fy_f_i = np.sin(delta_f_i) * T_f_i + np.cos(delta_f_i) * (self.C_f * alpha_f_i)
            Fx_r_i = np.cos(delta_r_i) * T_r_i - np.sin(delta_r_i) * (self.C_r * alpha_r_i)
            Fy_r_i = np.sin(delta_r_i) * T_r_i + np.cos(delta_r_i) * (self.C_r * alpha_r_i)

            # Forces and moment on COG in vehicle coordinate frame
            Fx_v_i = Fx_f_i + Fx_r_i
            Fy_v_i = Fy_f_i + Fy_r_i
            Mz_i = Fy_f_i * self.l_f - Fy_r_i * self.l_r

            # Forces on COG in global coordinate frame
            self.Fx_i[i] = np.cos(Psi_i) * Fx_v_i - np.sin(Psi_i) * Fy_v_i
            self.Fy_i[i] = np.sin(Psi_i) * Fx_v_i + np.cos(Psi_i) * Fy_v_i

            # combine i-th generalized force
            xi += self.getJ(i).T @ np.array([self.Fx_i[i], self.Fy_i[i], Mz_i])

        return xi

    def getB(
            self,
    ) -> npt.NDArray:
        """Calculate the B matrix in Lagrangian dynamics.
        计算系统的惯性矩阵B
        The file is originally generated by MATLAB symbolic and
        later transformed to python by hand.

        Returns
        ----------
        The B matrix.
        """
        Psi_1 = self.x[4 - 1]
        Psi_2 = self.x[5 - 1]
        Psi_o = self.x[3 - 1]
        t2 = np.cos(Psi_1)
        t3 = np.cos(Psi_2)
        t4 = np.cos(Psi_o)
        t5 = np.sin(Psi_1)
        t6 = np.sin(Psi_2)
        t7 = np.sin(Psi_o)
        t8 = self.M_1 + self.M_2 + self.M_o
        t9 = t2 * self.x__1_1
        t10 = t3 * self.x__2_2
        t11 = t4 * self.x__o_1
        t12 = t4 * self.x__o_2
        t13 = t2 * self.y__1_1
        t14 = t3 * self.y__2_2
        t15 = t4 * self.y__o_1
        t16 = t4 * self.y__o_2
        t17 = t5 * self.x__1_1
        t18 = t6 * self.x__2_2
        t19 = t7 * self.x__o_1
        t20 = t7 * self.x__o_2
        t21 = t5 * self.y__1_1
        t22 = t6 * self.y__2_2
        t23 = t7 * self.y__o_1
        t24 = t7 * self.y__o_2
        t25 = t9 * 2.0
        t26 = t10 * 2.0
        t27 = t11 * 2.0
        t28 = t12 * 2.0
        t29 = t13 * 2.0
        t30 = t14 * 2.0
        t31 = t15 * 2.0
        t32 = t16 * 2.0
        t33 = t17 * 2.0
        t34 = t18 * 2.0
        t35 = t19 * 2.0
        t36 = t20 * 2.0
        t37 = t21 * 2.0
        t38 = t22 * 2.0
        t39 = t23 * 2.0
        t40 = t24 * 2.0
        t41 = -t21
        t43 = -t22
        t45 = -t23
        t47 = -t24
        t49 = t13 + t17
        t50 = t14 + t18
        t51 = t15 + t19
        t52 = t16 + t20
        t42 = -t37
        t44 = -t38
        t46 = -t39
        t48 = -t40
        t53 = t9 + t41
        t54 = t29 + t33
        t55 = t10 + t43
        t56 = t30 + t34
        t57 = t11 + t45
        t58 = t12 + t47
        t59 = t31 + t35
        t60 = t32 + t36
        t77 = t49 * t51 * 2.0
        t78 = t50 * t52 * 2.0
        t61 = t25 + t42
        t62 = t26 + t44
        t63 = t27 + t46
        t64 = t28 + t48
        t65 = (self.M_1 * t54) / 2.0
        t66 = (self.M_2 * t56) / 2.0
        t67 = (self.M_1 * t59) / 2.0
        t68 = (self.M_2 * t60) / 2.0
        t79 = t53 * t57 * 2.0
        t80 = t55 * t58 * 2.0
        t69 = (self.M_1 * t61) / 2.0
        t70 = (self.M_2 * t62) / 2.0
        t71 = (self.M_1 * t63) / 2.0
        t72 = -t67
        t73 = (self.M_2 * t64) / 2.0
        t74 = -t68
        t83 = t77 + t79
        t84 = t78 + t80
        t75 = -t69
        t76 = -t70
        t81 = t71 + t73
        t82 = t72 + t74
        t85 = (self.M_1 * t83) / 2.0
        t86 = (self.M_2 * t84) / 2.0
        t87 = -t85
        t88 = -t86
        return np.array([
            t8, 0.0, t82, t65, t66, 0.0, t8, t81, t75, t76, t82, t81, self.I_o
                                                                      + (self.M_1 * (
                        t51 ** 2 * 2.0 + t57 ** 2 * 2.0)) / 2.0 + (self.M_2 * (t52 ** 2 * 2.0 + t58 ** 2 * 2.0)) / 2.0,
            t87, t88, t65, t75, t87,
                                                                      self.I_1 + (self.M_1 * (
                                                                                  t49 ** 2 * 2.0 + t53 ** 2 * 2.0)) / 2.0,
            0.0, t66, t76, t88, 0.0,
                                                                      self.I_2 + (self.M_2 * (
                                                                                  t50 ** 2 * 2.0 + t55 ** 2 * 2.0)) / 2.0]).reshape(
            (5, 5), order='F')

    def getn(
            self,
    ) -> npt.NDArray:
        """Calculate the n vector in Lagrangian dynamics.
        计算非惯性力向量n
        The file is originally generated by MATLAB symbolic and
        later transformed to python by hand.

        Returns
        ----------
        The n matrix.
        """
        Psi_o = self.x[2]
        Psi_1 = self.x[3]
        Psi_2 = self.x[4]
        Psi_dot_o = self.x[7]
        Psi_dot_1 = self.x[8]
        Psi_dot_2 = self.x[9]

        t2 = np.cos(Psi_1)
        t3 = np.cos(Psi_2)
        t4 = np.cos(Psi_o)
        t5 = np.sin(Psi_1)
        t6 = np.sin(Psi_2)
        t7 = np.sin(Psi_o)
        t8 = Psi_dot_1 ** 2
        t9 = Psi_dot_2 ** 2
        t10 = Psi_dot_o ** 2
        t11 = -Psi_o
        t12 = Psi_1 + t11
        t13 = Psi_2 + t11
        t14 = np.cos(t12)
        t15 = np.cos(t13)
        t16 = np.sin(t12)
        t17 = np.sin(t13)
        mt1 = np.array([-Psi_dot_o * (
                    (self.M_1 * (Psi_dot_o * t4 * self.x__o_1 * 2.0 - Psi_dot_o * t7 * self.y__o_1 * 2.0)) / 2.0 +
                    (self.M_2 * (Psi_dot_o * t4 * self.x__o_2 * 2.0 - Psi_dot_o * t7 * self.y__o_2 * 2.0)) / 2.0) +
                        (self.M_1 * Psi_dot_1 * (
                                    Psi_dot_1 * t2 * self.x__1_1 * 2.0 - Psi_dot_1 * t5 * self.y__1_1 * 2.0)) / 2.0 +
                        (self.M_2 * Psi_dot_2 * (
                                    Psi_dot_2 * t3 * self.x__2_2 * 2.0 - Psi_dot_2 * t6 * self.y__2_2 * 2.0)) / 2.0,
                        -Psi_dot_o * ((self.M_1 * (
                                    Psi_dot_o * t7 * self.x__o_1 * 2.0 + Psi_dot_o * t4 * self.y__o_1 * 2.0)) / 2.0
                                      + (self.M_2 * (
                                            Psi_dot_o * t7 * self.x__o_2 * 2.0 + Psi_dot_o * t4 * self.y__o_2 * 2.0)) / 2.0)
                        + (self.M_1 * Psi_dot_1 * (
                                    Psi_dot_1 * t5 * self.x__1_1 * 2.0 + Psi_dot_1 * t2 * self.y__1_1 * 2.0)) / 2.0
                        + (self.M_2 * Psi_dot_2 * (
                                    Psi_dot_2 * t6 * self.x__2_2 * 2.0 + Psi_dot_2 * t3 * self.y__2_2 * 2.0)) / 2.0])
        mt2 = np.array([
                           self.M_1 * t8 * t16 * self.x__1_1 * self.x__o_1 + self.M_2 * t9 * t17 * self.x__2_2 * self.x__o_2 - self.M_1 * t8 * t14 * self.x__1_1 * self.y__o_1
                           + self.M_1 * t8 * t14 * self.x__o_1 * self.y__1_1 - self.M_2 * t9 * t15 * self.x__2_2 * self.y__o_2 + self.M_2 * t9 * t15 * self.x__o_2 * self.y__2_2
                           + self.M_1 * t8 * t16 * self.y__1_1 * self.y__o_1 + self.M_2 * t9 * t17 * self.y__2_2 * self.y__o_2,
                           -self.M_1 * t10 * (
                                       t16 * self.x__1_1 * self.x__o_1 - t14 * self.x__1_1 * self.y__o_1 + t14 * self.x__o_1 * self.y__1_1 + t16 * self.y__1_1 * self.y__o_1),
                           -self.M_2 * t10 * (
                                       t17 * self.x__2_2 * self.x__o_2 - t15 * self.x__2_2 * self.y__o_2 + t15 * self.x__o_2 * self.y__2_2 + t17 * self.y__2_2 * self.y__o_2)])
        return np.concatenate((mt1, mt2))  # 5*1

    def getJ(
            self,
            i: int,
    ) -> npt.NDArray:
        """Calculate the i-th geometric Jacobian in Lagrangian dynamics.

        The file is originally generated by MATLAB symbolic and
        later transformed to python by hand.

        Parameters
        ----------
        i: int
            index of the end-effector.

        Returns
        ----------
        The i-th geometric Jacobian.
        """
        if i == 0:
            Psi_1 = self.x[3]
            Psi_o = self.x[2]
            t2 = np.cos(Psi_1)
            t3 = np.cos(Psi_o)
            t4 = np.sin(Psi_1)
            t5 = np.sin(Psi_o)
            return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                             -t5 * self.x__o_1 - t3 * self.y__o_1, t3 * self.x__o_1 - t5 * self.y__o_1, 0.0,
                             t4 * self.x__1_1
                             + t2 * self.y__1_1, -t2 * self.x__1_1 + t4 * self.y__1_1, 1.0, 0.0, 0.0, 0.0]).reshape(
                (3, 5), order='F')
        elif i == 1:
            Psi_2 = self.x[4]
            Psi_o = self.x[2]
            t2 = np.cos(Psi_2)
            t3 = np.cos(Psi_o)
            t4 = np.sin(Psi_2)
            t5 = np.sin(Psi_o)
            return np.array([1.0,0.0,0.0,0.0,1.0,0.0,
                             -t5*self.x__o_2-t3*self.y__o_2,t3*self.x__o_2-t5*self.y__o_2,
                             0.0,0.0,0.0,0.0,t4*self.x__2_2+t2*self.y__2_2,-t2*self.x__2_2+t4*self.y__2_2,1.0]).reshape(
                (3, 5), order='F')
