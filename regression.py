import os
import tensorflow.python.keras as keras
from data_generator import collect_data  # 加载导弹模块

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def init_network(learn_rate=0.001):
    x = keras.Input(shape=[6])
    l1 = keras.layers.Dense(units=100, activation='relu')(x)
    for i in range(10):
        l1 = keras.layers.add([l1, keras.layers.Dense(units=100, activation='relu')(l1)])  # 残差神经网络
    y = keras.layers.Dense(1, name='dense_output_angle')(l1)
    model = keras.Model(inputs=x, outputs=y)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learn_rate))
    return model


def regression():
    try:
        model = keras.models.load_model("./predictor_module/tgo_model.h5")
    except OSError:
        inputa, labela = collect_data(10000)
        x_raw = inputa
        tgo = labela[:, 0]  # 实际剩余飞行时间
        tgo_hat = labela[:, 1]  # 预测剩余飞行时间
        y_raw = tgo - tgo_hat  # 剩余飞行时间预测误差
        # model = init_network(learn_rate=0.001)
        model = keras.models.load_model("./predictor_module/tgo_model.h5")
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0000001))
        model.fit(x_raw, y_raw, batch_size=500, epochs=5, verbose=1, validation_split=0.02, use_multiprocessing=True)
        model.save("./predictor_module/tgo_model.h5")
    return model


if __name__ == "__main__":
    regression()
