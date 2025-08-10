import os
import numpy as np
import tables
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

# 禁用 TF1.x eager 模式
tf.compat.v1.disable_eager_execution()

# 从 cnn.py 中导入网络结构
from cnn import build_network
import phase_parameters.params as params
import tf_functions

def build_inference_graph():
    """
    构建与训练时完全一致的推断图：
      - x_ph: [None, 29498]
      - is_training: bool 占位符
      - 输出 pred_params: 9 维向量
    """
    x_ph = tf.compat.v1.placeholder(tf.float32, [None, 29498], name="x_input")
    is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
    pred_params = build_network(x_ph, is_training)
    return x_ph, is_training, pred_params

def evaluate_sample(index,
                    test_hdf5=r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\test3.hdf5",
                    model_ckpt=r"models\cnn_experiment\cnn_model.ckpt-70"):
    """
    对单个样本计算并可视化：
      - MSE
      - PSNR, SSIM (XUV 幅值)
      - 相位误差分布
    """
    # 构建推断图
    tf.compat.v1.reset_default_graph()
    x_ph, is_training, pred_params = build_inference_graph()
    saver = tf.compat.v1.train.Saver()

    # 读取数据
    with tables.open_file(test_hdf5, 'r') as h5:
        trace_data      = h5.root.noise_trace[:]
        xuv_coefs_data  = h5.root.xuv_coefs[:]
        ir_params_data  = h5.root.ir_params[:]

    trace = trace_data[index]
    xuv_true = xuv_coefs_data[index]
    ir_true  = ir_params_data[index]

    # 恢复模型并预测
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_ckpt)
        pred = sess.run(pred_params,
                        feed_dict={x_ph: trace.reshape(1,-1), is_training: False})[0]
    xuv_pred = pred[:5]
    ir_pred  = pred[5:]

    # 计算 MSE
    y_true = np.concatenate([xuv_true, ir_true])
    mse = np.mean((pred - y_true)**2)
    print(f"Sample {index} MSE: {mse:.6f}")

    # 计算 XUV 复数光谱（f_cropped）
    with tf.compat.v1.Session() as sess:
        # xuv_taylor_to_E 接受 [1,5] 的 coeffs
        f_pred   = sess.run(tf_functions.xuv_taylor_to_E(
                        tf.constant(xuv_pred.reshape(1,5), tf.float32)
                    )["f_cropped"])[0]
        f_actual = sess.run(tf_functions.xuv_taylor_to_E(
                        tf.constant(xuv_true.reshape(1,5), tf.float32)
                    )["f_cropped"])[0]

    amp_pred   = np.abs(f_pred)
    amp_actual = np.abs(f_actual)
    phase_pred   = np.unwrap(np.angle(f_pred))
    phase_actual = np.unwrap(np.angle(f_actual))

    # PSNR
    mse_amp = np.mean((amp_pred - amp_actual)**2)
    psnr_val = 10 * np.log10(np.max(amp_actual)**2 / (mse_amp + 1e-12))
    # SSIM
    ssim_val = ssim(amp_actual, amp_pred,
                    data_range=amp_actual.max() - amp_actual.min())
    # 相位误差
    phase_err = phase_pred - phase_actual

    # 绘图
    Kvals, tauvals = params.K, params.delay_values
    trace_2d = trace.reshape(len(Kvals), len(tauvals))
    photon_E = np.linspace(50, 350, amp_pred.size)

    fig, axs = plt.subplots(2, 2, figsize=(12,10))

    # (1) Streaking Trace
    im = axs[0,0].pcolormesh(tauvals, Kvals, trace_2d, cmap='jet')
    axs[0,0].set_title(f"Streaking Trace (idx={index})")
    axs[0,0].set_xlabel("Delay [fs]"); axs[0,0].set_ylabel("Energy [eV]")
    fig.colorbar(im, ax=axs[0,0])

    # (2) XUV 幅值对比
    axs[0,1].plot(photon_E, amp_pred,   label="Pred")
    axs[0,1].plot(photon_E, amp_actual, '--', label="Actual")
    axs[0,1].set_title(f"XUV Amp | PSNR={psnr_val:.2f}dB SSIM={ssim_val:.4f}")
    axs[0,1].set_xlabel("Photon Energy [eV]"); axs[0,1].legend()

    # (3) XUV 相位对比
    axs[1,0].plot(photon_E, phase_pred,   label="Pred")
    axs[1,0].plot(photon_E, phase_actual, '--', label="Actual")
    axs[1,0].set_title("XUV Phase")
    axs[1,0].set_xlabel("Photon Energy [eV]"); axs[1,0].legend()

   # (4) Phase Error Distribution — 带填充色的直方图
    axs[1,1].hist(
        phase_err,
        bins=50,
        color='white',    # 柱子的填充色
        edgecolor='black'     # 柱子的边缘色
    )
    axs[1,1].set_title("Phase Error Distribution")
    axs[1,1].set_xlabel("Error (rad)")
    axs[1,1].set_ylabel("Count")


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 修改 index、路径 后直接运行
    evaluate_sample(index=69)
    

