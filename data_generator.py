""" Code for loading data. """
import os
import numpy as np
from scipy.io import loadmat, savemat
from missile import MISSILE

pi = 3.141592653589793
RAD = 180 / pi  # 弧度转角度
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 关闭对显卡的调用


def generate(attacker):
    # 任务生成思路是以确定的导弹模型和初始条件出发，完成一条弹道，从弹道中随机选取固定数量的自变量，生成样本
    h = 0.05  # 仿真步长
    while True:
        step = []  # 记录飞行轨迹数据
        runtime = np.array([])
        f_time = np.array([])
        attacker.modify()  # 随机初始化攻击者
        done = attacker.step(h)  # 单步运行
        while done is False:
            step.append([attacker.v * 1e-2,
                         attacker.theta,
                         attacker.R * 1e-4,
                         attacker.ad,
                         attacker.x * 1e-4,
                         attacker.y * 1e-4])
            runtime = np.append(runtime, attacker.Y[0])
            f_time = np.append(f_time, attacker.R / attacker.v)
            done = attacker.step(h)  # 单步运行

        print("脱靶量={}, 落角误差={}".format(attacker.R, (attacker.ad - attacker.theta) * RAD))
        if attacker.R < 20:
            tf = attacker.Y[0]
            inputs = np.array(step)
            outputs = np.array([tf * np.ones([runtime.shape[0]]) - runtime, f_time]).T
            break
    return inputs, outputs


def collect_data(iterations):
    try:
        data_raw = loadmat('./anti_flight_data.mat')
        print("data load done!")
        inputa = data_raw["inputa"]
        labela = data_raw["labela"]
    except FileNotFoundError:
        print('Data initializing.')
        attacker = MISSILE(target=MISSILE([0, 0, 0, 0, 0, 0]))  # 为攻击者绑定目标
        inputa, labela = generate(attacker)  # 生成随机正弦样本
        for itr in range(1, iterations):
            print("==========迭代步数：{}==========".format(itr))
            batch_x, batch_y = generate(attacker)  # 生成随机正弦样本
            inputa = np.concatenate([inputa, batch_x])
            labela = np.concatenate([labela, batch_y])
        flight_data = {"inputa": inputa, "labela": labela}
        savemat('./anti_flight_data.mat', flight_data)
    return inputa, labela


if __name__ == "__main__":
    collect_data(1000)
