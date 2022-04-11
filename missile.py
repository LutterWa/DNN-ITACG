import math
import random
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt  # clf()清图  # cla()清坐标轴  # close()关窗口

pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度
g = 9.81

Ma2 = np.array([[0.4, 39.056, 0.4604, 39.072],
                [0.6, 39.468, 0.4635, 39.242],
                [0.8, 40.801, 0.4682, 40.351],
                [0.9, 41.372, 0.4776, 41.735],
                [1.0, 41.878, 0.4804, 43.014],
                [1.1, 42.468, 0.4797, 42.801],
                [1.2, 41.531, 0.4784, 42.656],
                [1.3, 41.224, 0.4771, 42.593],
                [1.4, 40.732, 0.4768, 42.442],
                [1.5, 40.321, 0.4707, 42.218]])


def load_atm(path):
    file = open(path)
    atm_str = file.read().split()
    atm = []
    for _ in range(0, len(atm_str), 3):
        atm.append([float(atm_str[_]), float(atm_str[_ + 1]), float(atm_str[_ + 2])])
    return np.array(atm)


class MISSILE:
    def __init__(self, missile=None, target=None):
        self.target = target  # 目标位置
        self.ad = -90. / RAD  # 目标落角
        self.td = 100.  # 目标飞行时间

        if missile is None:
            missile = [0., 200., 0. / RAD, -10000., 10000, 100]

        self.Y = np.array(missile)  # 导弹初始状态: 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量

        self.v = self.Y[1]  # 速度
        self.theta = self.Y[2]  # 弹道倾角
        self.x = self.Y[3]  # 横向位置
        self.y = self.Y[4]  # 纵向位置
        self.m = self.Y[5]  # 导弹重量l

        self.X = 0.  # 阻力drag force
        self.L = 0.  # 升力lift force

        self.S = 0.05  # 参考面积

        self.R, self.q, self.Rdot, self.qdot = 0., 0., 0., 0.  # 弹目相对关系
        self.am = 0.  # 制导指令
        self.alpha = 0.  # 平衡攻角

        # 创建大气数据插值函数
        atm = load_atm('atm2.txt')  # 大气参数
        self.f_rho = interpolate.interp1d(atm[:, 0], atm[:, 1], 'linear')
        self.f_ma = interpolate.interp1d(atm[:, 0], atm[:, 2], 'linear')
        self.f_clalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 1], 'cubic')  # 升力系数
        self.f_cd0 = interpolate.interp1d(Ma2[:, 0], Ma2[:, 2], 'cubic')  # 零升阻力
        self.f_cdalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 3], 'cubic')  # 攻角阻力

        # 全弹道历史信息
        self.reR, self.reY = [], []

    def modify(self, missile=None):  # 修改导弹初始状态
        self.terminate()
        if missile is None:
            missile = [0.,
                       random.uniform(200, 300),
                       random.uniform(-15, 30) / RAD,
                       random.uniform(-10000, -5000),
                       random.uniform(5000, 10000),
                       100]  # 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量
            # if missile[3] > 0:
            #     missile[2] = pi - missile[2]
        # 随机缩放气动参数
        # k = random.uniform(0.8, 1.2)
        # self.f_clalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 1] * k, 'cubic')
        # self.f_cd0 = interpolate.interp1d(Ma2[:, 0], Ma2[:, 2] * k, 'cubic')
        # self.f_cdalpha = interpolate.interp1d(Ma2[:, 0], Ma2[:, 3] * k, 'cubic')

        self.ad = random.uniform(-90, -30) / RAD

        self.Y = np.array(missile)
        self.seeker(self.target)
        return missile

    def guidance(self):
        # ac = 2*(tsg_n+2)*v*qdot+(tsg_n+1)*(tsg_n+2)*v*(q-ad)/R/v
        tgo = self.R / self.v  # 剩余飞行时间
        ac_zem = 4 * self.v * self.qdot  # 零控脱靶量
        ac_angle = 2 * self.v * (self.q - self.ad) / tgo  # 落角约束
        ac_gravity = math.cos(self.theta) * g  # 重力补偿
        ac = ac_zem + ac_angle + ac_gravity
        return ac

    def step(self, h=0.001, ab=0.):
        if self.Y[0] < 400:
            RHO = self.get_rho(self.y)  # 大气密度
            ma = self.get_ma(self.y, self.v)  # 马赫数
            Q = 0.5 * RHO * self.v ** 2  # 动压

            R, Rdot, q, qdot = self.seeker(self.target)  # 导引头给出弹目信息

            self.reR.append(self.R)
            self.reY.append(self.Y)

            if self.y < 0:
                return True

            cl_alpha = self.get_clalpha(ma)
            m_max = 5 * g
            self.am = np.clip(self.guidance() + ab, -m_max, m_max)

            self.alpha = alpha = (self.m * self.am) / (Q * self.S * cl_alpha)  # 平衡攻角，使用了sin(x)=x的近似，在10°以内满足这一关系
            alpha_bound = 15 / RAD
            if alpha > alpha_bound:
                self.alpha = alpha = alpha_bound
            elif alpha < -alpha_bound:
                self.alpha = alpha = -alpha_bound

            cd = self.get_cd0(ma) + self.get_cdalpha(ma) * alpha ** 2  # 阻力系数
            cl = cl_alpha * alpha  # 升力系数

            self.X = cd * Q * self.S  # 阻力
            self.L = cl * Q * self.S  # 升力

            def rk4(func, Y):
                k1 = h * func(Y)
                k2 = h * func(Y + 0.5 * k1)
                k3 = h * func(Y + 0.5 * k2)
                k4 = h * func(Y + k3)
                output = Y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                return output

            self.Y = rk4(self.dery, self.Y)
            if self.Y[2] > pi:
                self.Y[2] = self.Y[2] - 2 * pi
            if self.Y[2] < -pi:
                self.Y[2] = self.Y[2] + 2 * pi

            return False
        else:
            self.reR.append(self.R)
            self.reY.append(self.Y)
            print("超时！未击中目标！")
            return True

    def plot_data(self):
        R = np.array(self.reR)
        Y = np.array(self.reY)

        fig = plt.figure(0)

        plt.ion()
        plt.clf()
        # 弹道曲线
        plt.subplots_adjust(hspace=0.6)
        ax1 = fig.add_subplot(221)
        ax1.plot(Y[:, 3] / 1000, Y[:, 4] / 1000, 'k-')
        ax1.set_xlabel('$X (km)$')
        ax1.set_ylabel('$Y (km)$')
        ax1.grid(True)

        # 速度曲线
        ax2 = fig.add_subplot(222)
        ax2.plot(Y[:, 0], Y[:, 1], 'k-')
        ax2.set_xlabel('$t (s)$')
        ax2.set_ylabel('$v (m/s)$')
        ax2.grid(True)

        # 弹目距离
        ax3 = fig.add_subplot(223)
        ax3.plot(Y[:, 0], R / 1000, 'k-')
        ax3.set_xlabel('$t (s)$')
        ax3.set_ylabel(r'$R (km)$')
        ax3.grid(True)

        # 弹道倾角
        ax4 = fig.add_subplot(224)
        ax4.plot(Y[:, 0], Y[:, 2] * RAD, 'k-')
        ax4.set_xlabel('$t (s)$')
        ax4.set_ylabel(r'$a_b(m/s^2)$')
        ax4.grid(True)

        plt.pause(0.1)

        return [ax1, ax2, ax3, ax4]

    def seeker(self, item):  # 计算弹目相对信息
        self.x = x = self.Y[3]  # 横向位置
        self.y = y = self.Y[4]  # 纵向位置
        self.v = v = self.Y[1]  # 速度
        self.theta = theta = self.Y[2]  # 弹道倾角

        Rx = item.x - x
        Ry = item.y - y
        vx = item.v * math.cos(item.theta) - v * math.cos(theta)  # x向速度
        vy = item.v * math.sin(item.theta) - v * math.sin(theta)  # y向速度

        self.R = R = np.linalg.norm([Rx, Ry], ord=2)  # 弹目距离
        self.Rdot = Rdot = (Rx * vx + Ry * vy) / R

        self.q = q = math.atan2(Ry, Rx)  # 弹目视线角
        self.qdot = qdot = (Rx * vy - Ry * vx) / R ** 2

        return R, Rdot, q, qdot

    def get_ma(self, y, v):  # 计算马赫数
        y = max(0, y)
        sonic = self.f_ma(y)
        return v / sonic

    def get_rho(self, y):  # 计算空气密度
        y = max(0, y)
        return self.f_rho(y)

    def get_clalpha(self, ma):
        return self.f_clalpha(max(min(ma, 1.5), 0.4))

    def get_cd0(self, ma):
        return self.f_cd0(max(min(ma, 1.5), 0.4))

    def get_cdalpha(self, ma):
        return self.f_cdalpha(max(min(ma, 1.5), 0.4))

    def dery(self, Y):  # 右端子函数
        v = Y[1]
        theta = Y[2]
        m = Y[5]
        dy = np.array(Y)
        dy[0] = 1
        dy[1] = (-self.X) / m - g * math.sin(theta)
        dy[2] = (self.L - m * g * math.cos(theta)) / (v * m)
        dy[3] = v * math.cos(theta)
        dy[4] = v * math.sin(theta)
        dy[5] = 0.
        return dy

    def collect(self):
        t = self.Y[0]  # 时间
        v = self.Y[1]  # 速度
        theta = self.Y[2]  # 弹道倾角
        r = self.R  # 弹目距离
        q = self.q  # 弹目视线角
        x = self.Y[3]  # 弹横向位置
        y = self.Y[4]  # 弹纵向位置
        return v, theta, r, q, x, y, t

    def terminate(self):
        self.X = 0.  # 阻力drag force
        self.L = 0.  # 升力lift force

        self.R, self.q, self.Rdot, self.qdot = 0., 0., 0., 0.  # 弹目相对关系
        self.am = 0.  # 制导指令
        self.alpha = 0.  # 平衡攻角

        self.reR, self.reY = [], []


if __name__ == '__main__':
    from scipy.io import savemat
    from regression import regression

    resnet = regression()
    miss = MISSILE(target=MISSILE([0, 0, 0, 0, 0, 0]))
    monte = {"td": [], "tf": [], "te": [], "ad": [], "af": [], "ae": []}
    for i in range(1000):
        miss.modify()  # [0, 300, 0, -10000, 10000, 100]
        x = np.array([miss.v * 1e-2, miss.theta, miss.R * 1e-4, miss.ad, miss.x * 1e-4, miss.y * 1e-4])[np.newaxis, :]
        miss.td = miss.R / miss.v + float(resnet.predict(x, use_multiprocessing=True)) + 20

        list_tgo = []

        done = False
        while done is False:
            x = np.array([miss.v * 1e-2,
                          miss.theta,
                          miss.R * 1e-4,
                          miss.ad,
                          miss.x * 1e-4,
                          miss.y * 1e-4])[np.newaxis, :]
            P = miss.R / miss.v + float(resnet.predict(x, use_multiprocessing=True))
            D = miss.td - miss.Y[0]
            E = D - P
            ab = 3 * miss.v ** 2 / miss.R / (4 * (miss.theta - miss.q) + miss.q - miss.ad) * max(E, 0)
            done = miss.step(0.01, ab=ab)  # 单步运行

            list_tgo.append([D, P])

        # miss.plot_data()
        #
        # fig = plt.figure(1)
        # Y = np.array(miss.reY)
        # plt.clf()
        # plt.plot(Y[:, 0], np.array(list_tgo)[:, 0], label='desired tgo')
        # plt.plot(Y[:, 0], np.array(list_tgo)[:, 1], label='actual tgo')
        # plt.xlabel('Time (s)')
        # plt.ylabel('t_go(s)')
        # plt.legend()
        # plt.grid()
        # plt.pause(0.1)

        monte["td"].append(miss.td)
        monte["tf"].append(miss.Y[0])
        monte["te"].append(miss.td - miss.Y[0])
        monte["ad"].append(miss.ad * RAD)
        monte["af"].append(miss.theta * RAD)
        monte["ae"].append((miss.ad - miss.theta) * RAD)

        print("落角误差={:.4f} 时间误差={:.4f}".format((miss.ad - miss.theta) * RAD, miss.td - miss.Y[0]))

    savemat('./monte_test.mat', monte)
