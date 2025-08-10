import os
import numpy as np
import tables
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim  # pip install scikit-image

tf.compat.v1.disable_eager_execution()

# 从 vit.py 中导入 ViT 网络结构（确保网络结构与训练时一致）
from vit import build_vit
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
    pred_params = build_vit(x_ph, is_training)
    return x_ph, is_training, pred_params

def plot_metrics_for_sample(index=0,
                            test_hdf5=r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\test3.hdf5",
                            model_ckpt_path=r"vit_models/vit_experiment/vit_model.ckpt-25"):
    """
    可视化并计算指标：
      1) Streaking Trace
      2) XUV 幅值对比（Pred vs Actual）— 并显示 PSNR & SSIM
      3) XUV 相位对比
      4) Phase Error Distribution（带填充色和黑色边框）
      同时在控制台打印 MSE。
    """
    # 1. 构建推断图、恢复模型
    tf.compat.v1.reset_default_graph()
    x_ph, is_training, pred_params = build_inference_graph()
    saver = tf.compat.v1.train.Saver()

    # 2. 读取测试数据
    with tables.open_file(test_hdf5, 'r') as h5:
        traces = h5.root.noise_trace[:]
        xuv_true_data = h5.root.xuv_coefs[:]
        ir_true_data  = h5.root.ir_params[:]

    trace_1d      = traces[index]
    xuv_coefs_true = xuv_true_data[index]
    ir_params_true = ir_true_data[index]

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_ckpt_path)
        pred = sess.run(pred_params,
                        feed_dict={x_ph: trace_1d.reshape(1, -1),
                                   is_training: False})[0]

    # 拆分预测结果
    xuv_coefs_pred = pred[:5]
    ir_params_pred = pred[5:]

    # 3. 打印 MSE
    y_true = np.concatenate([xuv_coefs_true, ir_params_true])
    mse = np.mean((pred - y_true)**2)
    print(f"Sample {index} MSE: {mse:.6f}")

    # 4. 计算复数光谱
    with tf.compat.v1.Session() as sess:
        f_pred = sess.run(
            tf_functions.xuv_taylor_to_E(
                tf.constant(xuv_coefs_pred.reshape(1,5), tf.float32)
            )["f_cropped"]
        )[0]
        f_actual = sess.run(
            tf_functions.xuv_taylor_to_E(
                tf.constant(xuv_coefs_true.reshape(1,5), tf.float32)
            )["f_cropped"]
        )[0]

    amp_pred    = np.abs(f_pred)
    amp_actual  = np.abs(f_actual)
    phase_pred  = np.unwrap(np.angle(f_pred))
    phase_actual= np.unwrap(np.angle(f_actual))

    # 5. PSNR 与 SSIM
    mse_amp = np.mean((amp_pred - amp_actual)**2)
    psnr_val = 10 * np.log10(np.max(amp_actual)**2 / (mse_amp + 1e-12))
    ssim_val = ssim(amp_actual, amp_pred,
                    data_range=amp_actual.max() - amp_actual.min())

    phase_error = phase_pred - phase_actual

    # 6. 绘图
    Kvals   = params.K
    tauvals = params.delay_values
    trace_2d = trace_1d.reshape(len(Kvals), len(tauvals))
    photon_E = np.linspace(50, 350, amp_pred.size)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # (1) Streaking Trace
    im = axs[0,0].pcolormesh(tauvals, Kvals, trace_2d, cmap='jet')
    axs[0,0].set_title(f"Streaking Trace (idx={index})")
    axs[0,0].set_xlabel("Delay [fs]")
    axs[0,0].set_ylabel("Energy [eV]")
    fig.colorbar(im, ax=axs[0,0])

    # (2) XUV 幅值对比
    axs[0,1].plot(photon_E, amp_pred,   label="Predicted")
    axs[0,1].plot(photon_E, amp_actual, '--', label="Actual")
    axs[0,1].set_title(f"XUV Amplitude\nPSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}")
    axs[0,1].set_xlabel("Photon Energy [eV]")
    axs[0,1].set_ylabel("Intensity (a.u.)")
    axs[0,1].legend()

    # (3) XUV 相位对比
    axs[1,0].plot(photon_E, phase_pred,   label="Predicted")
    axs[1,0].plot(photon_E, phase_actual, '--', label="Actual")
    axs[1,0].set_title("XUV Phase")
    axs[1,0].set_xlabel("Photon Energy [eV]")
    axs[1,0].set_ylabel("Phase [rad]")
    axs[1,0].legend()

    # (4) Phase Error Distribution
    axs[1,1].hist(phase_error,
                  bins=50,
                  color='white',
                  edgecolor='black')
    axs[1,1].set_title("Phase Error Distribution")
    axs[1,1].set_xlabel("Error [rad]")
    axs[1,1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 调用示例：修改 index 或路径后直接运行
    plot_metrics_for_sample(index=30)
