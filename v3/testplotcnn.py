import os
import numpy as np
import tables
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# 从 cnn.py 中导入 CNN 网络结构，确保推断时的网络结构与训练时一致
from cnn import build_network
import phase_parameters.params as params
import tf_functions

def build_inference_graph():
    """
    构建与训练时完全一致的推断图：
      - x_ph: [None, 29498]（即 301×98 的 flatten 结果）
      - is_training: bool 占位符
      - 输出 pred_params: 9 维向量（前5维为 xuv_coefs，后4维为 ir_params）
    """
    x_ph = tf.compat.v1.placeholder(tf.float32, [None, 29498], name="x_input")
    is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
    pred_params = build_network(x_ph, is_training)
    return x_ph, is_training, pred_params

def plot_prediction(index=0,
                    test_hdf5=r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\test3.hdf5",
                    model_ckpt_path=r"models\cnn_experiment\cnn_model.ckpt-40"):
    """
    1) 从 test3.hdf5 中加载测试数据；
    2) 用训练好的模型对第 index 个样本进行预测；
    3) 分别绘制三个图：
         - 图1：原始 streaking trace（二维图）；
         - 图2：预测的 XUV 光谱（强度和相位）；
         - 图3：真实的 XUV 光谱（强度和相位）；
    4) 同时输出该测试样本的均方误差（MSE）。
    """
    # ------------------- 读取测试数据 -------------------
    with tables.open_file(test_hdf5, 'r') as h5:
        trace_data = h5.root.noise_trace[:]
        xuv_coefs_data = h5.root.xuv_coefs[:]
        ir_params_data = h5.root.ir_params[:]

    # 取出第 index 个样本
    trace_1d = trace_data[index]           # shape: (29498,)
    # 真实的 xuv_coefs 和 ir_params
    xuv_coefs_true = xuv_coefs_data[index]   # shape: (5,)
    ir_params_true = ir_params_data[index]   # shape: (4,)

    # ------------------- 构建推断图并恢复训练好的模型 -------------------
    tf.compat.v1.reset_default_graph()
    x_ph, is_training, pred_params = build_inference_graph()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_ckpt_path)
        feed = {x_ph: trace_1d.reshape(1, -1), is_training: False}
        pred = sess.run(pred_params, feed_dict=feed)[0]  # 得到 9 维向量
        xuv_coefs_pred = pred[:5]
        ir_params_pred = pred[5:]

        # ------------------- 输出选中样本的均方误差 -------------------
        # 构造真实标签向量（xuv_coefs 和 ir_params 拼接）
        y_true = np.concatenate([xuv_coefs_true, ir_params_true], axis=0)
        mse = np.mean((pred - y_true) ** 2)
        print("Test sample index {}: MSE = {:.6f}".format(index, mse))

        # ------------------- 图1：Streaking Trace -------------------
        # 假设 params.K 长度为 301，params.delay_values 长度为 98
        Kvals = params.K
        tauvals = params.delay_values
        trace_2d = trace_1d.reshape(len(Kvals), len(tauvals))
        fig1 = plt.figure(figsize=(6,5))
        ax1 = fig1.add_subplot(1, 1, 1)
        c = ax1.pcolormesh(tauvals, Kvals, trace_2d, cmap='jet')
        ax1.set_title(f"Streaking Trace (index={index})")
        ax1.set_xlabel("Time Delay [fs]")
        ax1.set_ylabel("Energy [eV]")
        plt.colorbar(c, ax=ax1)
        plt.tight_layout()

        # ------------------- 计算 XUV 复数光谱 -------------------
        # 利用预测的 xuv_coefs_pred 生成预测的 XUV 复数光谱
        xuv_coefs_pred_tensor = tf.constant(xuv_coefs_pred.reshape(1, 5), dtype=tf.float32)
        xuv_E_prop_pred = tf_functions.xuv_taylor_to_E(xuv_coefs_pred_tensor)
        xuv_f_pred = sess.run(xuv_E_prop_pred["f_cropped"])[0]  # 预测复数光谱

        # 利用真实的 xuv_coefs_true 生成真实的 XUV 复数光谱
        xuv_coefs_true_tensor = tf.constant(xuv_coefs_true.reshape(1, 5), dtype=tf.float32)
        xuv_E_prop_actual = tf_functions.xuv_taylor_to_E(xuv_coefs_true_tensor)
        xuv_f_actual = sess.run(xuv_E_prop_actual["f_cropped"])[0]  # 真实复数光谱

        # 根据 xuv_f 的长度生成光子能量轴（假设范围为 50～350 eV）
        photon_energy = np.linspace(50, 350, len(xuv_f_pred))

        # 计算预测的幅值和相位
        amplitude_pred = np.abs(xuv_f_pred)
        phase_pred = np.unwrap(np.angle(xuv_f_pred))
        # 计算真实的幅值和相位
        amplitude_actual = np.abs(xuv_f_actual)
        phase_actual = np.unwrap(np.angle(xuv_f_actual))

        # ------------------- 图2：预测的 XUV 光谱 -------------------
        fig2 = plt.figure(figsize=(6,5))
        ax2 = fig2.add_subplot(1,1,1)
        ax2.plot(photon_energy, amplitude_pred, color='black', label='Predicted Intensity')
        ax2.set_xlabel("Photon Energy [eV]")
        ax2.set_ylabel("Intensity (a.u.)", color='black')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(photon_energy, phase_pred, color='green', label='Predicted Phase')
        ax2_twin.set_ylabel("Phase [rad]", color='green')
        ax2.set_title("Predicted XUV Spectrum")
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')
        plt.tight_layout()

        # ------------------- 图3：真实的 XUV 光谱 -------------------
        fig3 = plt.figure(figsize=(6,5))
        ax3 = fig3.add_subplot(1,1,1)
        ax3.plot(photon_energy, amplitude_actual, color='black', label='Actual Intensity')
        ax3.set_xlabel("Photon Energy [eV]")
        ax3.set_ylabel("Intensity (a.u.)", color='black')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(photon_energy, phase_actual, color='green', label='Actual Phase')
        ax3_twin.set_ylabel("Phase [rad]", color='green')
        ax3.set_title("Actual XUV Spectrum")
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc='best')
        plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    # 例如，对测试集第 2 个样本进行推断，并绘制 streaking trace、预测的 XUV 光谱和真实的 XUV 光谱，同时输出该样本的 MSE
    plot_prediction(index=95,
                    test_hdf5=r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\test3.hdf5",
                    model_ckpt_path=r"models\cnn_experiment\cnn_model.ckpt-70")
