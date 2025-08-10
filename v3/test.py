import tables
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 导入你工程中定义的模块
import tf_functions
import phase_parameters.params
import xuv_spectrum.spectrum
import ir_spectrum.ir_spectrum

# 禁用 TensorFlow eager execution（TF1.x 兼容模式）
tf.compat.v1.disable_eager_execution()

def plot_opened_file(xuv_coefs, ir_params, trace, sess, tf_graphs):
    """
    通过 TensorFlow 计算 XUV 和 IR 的时域信号，
    并将 XUV、IR 信号以及重构的 streaking trace 图像进行可视化。
    """
    # 利用 TensorFlow 计算 XUV 与 IR 的时域信号
    xuv_t = sess.run(tf_graphs["xuv_E_prop"]["t"],
                     feed_dict={tf_graphs["xuv_coefs_in"]: xuv_coefs.reshape(1, -1)})
    ir_t = sess.run(tf_graphs["ir_E_prop"]["t"],
                    feed_dict={tf_graphs["ir_values_in"]: ir_params.reshape(1, -1)})

    # 创建画布并设置 gridspec 布局
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)

    # 绘制 XUV 时域信号
    ax1 = fig.add_subplot(gs[0, 0])
    try:
        x_axis = xuv_spectrum.spectrum.tmat
    except Exception:
        x_axis = np.arange(len(xuv_t[0]))
    ax1.plot(x_axis, np.real(xuv_t[0]), color='black')
    ax1.set_title("XUV")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")

    # 绘制 IR 时域信号
    ax2 = fig.add_subplot(gs[0, 1])
    try:
        ir_axis = ir_spectrum.ir_spectrum.tmat
    except Exception:
        ir_axis = np.arange(len(ir_t[0]))
    ax2.plot(ir_axis, np.real(ir_t[0]), color='black')
    ax2.set_title("IR")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Amplitude")

    # ------------------- 图3：Streaking Trace（仅绘制 K=50–250） -------------------
    try:
        import phase_parameters.params as params
        tauvals = params.delay_values
        kvals  = params.K
    except Exception:
        tauvals = np.arange(98)
        kvals  = np.arange(301)

    # 将 trace 重塑为 [K, Delay]
    trace_2d = trace.reshape(len(kvals), len(tauvals))

    # 只保留 K 值在 50 到 250 范围内的行
    mask = (kvals >= 70) & (kvals <= 200)
    kvals_sub = kvals[mask]
    trace_sub = trace_2d[mask, :]

    ax3 = fig.add_subplot(gs[1, :])
    c = ax3.pcolormesh(tauvals, kvals_sub, trace_sub, cmap='jet')
    fig.colorbar(c, ax=ax3)
    ax3.set_title("Streaking Trace")
    ax3.set_xlabel("Delay")
    ax3.set_ylabel("K")
    # 同时设置 y 轴显示范围
    ax3.set_ylim(70, 200)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # 构建 TensorFlow 图
    # 初始化 XUV 生成器
    xuv_phase_coeffs = phase_parameters.params.xuv_phase_coefs
    xuv_coefs_in = tf.compat.v1.placeholder(tf.float32, shape=[None, phase_parameters.params.xuv_phase_coefs])
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_in)

    # 初始化 IR 生成器
    ir_values_in = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
    ir_E_prop = tf_functions.ir_from_params(ir_values_in)["E_prop"]

    # 构造 streaking image
    image = tf_functions.streaking_trace(xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
                                         ir_cropped_f_in=ir_E_prop["f_cropped"][0])
    # 用于计算 proof_trace 的占位符（本例中未显示 proof_trace，可按需要扩展）
    image_noisy_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[301, 98])
    proof_trace = tf_functions.proof_trace(image_noisy_placeholder)["proof"]

    # 将相关张量与占位符存入字典，方便后续调用
    tf_graphs = {
        "xuv_coefs_in": xuv_coefs_in,
        "ir_values_in": ir_values_in,
        "xuv_E_prop": xuv_E_prop,
        "ir_E_prop": ir_E_prop,
        "image": image,
        "image_noisy_placeholder": image_noisy_placeholder,
        "proof_trace": proof_trace
    }

    # 指定 HDF5 文件的路径（使用原始字符串防止转义问题）
    hdf5_filename = r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\test3.hdf5"
    sample_index = 69  # 可根据需要更改样本索引

    # 从 HDF5 文件中读取一个样本数据
    with tables.open_file(hdf5_filename, mode='r') as hd5file:
        xuv_coefs = hd5file.root.xuv_coefs[sample_index, :]
        ir_params = hd5file.root.ir_params[sample_index, :]
        trace = hd5file.root.noise_trace[sample_index, :]


    # 启动 TensorFlow Session 并进行可视化
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        plot_opened_file(xuv_coefs=xuv_coefs, ir_params=ir_params,
                         trace=trace, sess=sess, tf_graphs=tf_graphs)
