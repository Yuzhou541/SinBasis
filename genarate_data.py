# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import csv
import pickle
import scipy.constants as sc
import scipy.interpolate
from scipy.special import factorial
import os



# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 定义必要的参数类
class Params:
    # XUV 相位系数的数量
    xuv_phase_coefs = 5
    # 相位幅度
    amplitude = 20.0

    # 红外参数
    ir_param_amplitudes = {
        "phase_range": (0, 2 * np.pi),
        "clambda_range": (1.678, 1.678),
        "pulseduration_range": (11.0, 16.0),
        "I_range": (0.02, 0.12)
    }

    # 电离势
    Ip_eV = 24.587  # eV
    Ip = Ip_eV * sc.electron_volt  # joules
    Ip = Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.

    # 延迟值和动能 K
    delay_values = np.linspace(-5.47e-15, 5.44e-15, 98) / sc.physical_constants['atomic unit of time'][0]  # a.u.
    K = np.linspace(50, 350, 301)  # eV

    # 阈值参数
    threshold_scaler = 0.03
    threshold_min_index = 100
    threshold_max_index = (2 * 1024) - 100

params = Params()

# 定义数据生成函数，包括 xuv_spectrum 和 ir_spectrum 的定义
def my_interp(electronvolts_in, intensity_in, plotting=False):
    # 将 eV 转换为焦耳
    joules = np.array(electronvolts_in) * sc.electron_volt  # joules
    hertz = np.array(joules / sc.h)
    Intensity = np.array(intensity_in)
    # 定义 tmat 和 fmat
    N = int(2 * 1024)
    tmax = 1600e-18
    dt = 2 * tmax / N
    tmat = dt * np.arange(-N / 2, N / 2, 1)
    df = 1 / (N * dt)
    fmat = df * np.arange(-N / 2, N / 2, 1)

    # 用零填充向量
    hertz = np.insert(hertz, 0, hertz[0])
    Intensity = np.insert(Intensity, 0, 0)
    hertz = np.append(hertz, hertz[-1])
    Intensity = np.append(Intensity, 0)
    Intensity[Intensity < 0] = 0

    # 为后续插值再次填充零
    hertz = np.insert(hertz, 0, -6e18)
    Intensity = np.insert(Intensity, 0, 0)
    hertz = np.append(hertz, 6e18)
    Intensity = np.append(Intensity, 0)
    Intensity[Intensity < 0] = 0

    # 获取载波频率
    f0 = hertz[np.argmax(Intensity)]
    # 对强度开方以获取电场幅度
    Ef = np.sqrt(Intensity)
    # 将光谱线性映射到 N 个点
    interpolator = scipy.interpolate.interp1d(hertz, Ef, kind='linear', fill_value="extrapolate")
    Ef_interp = interpolator(fmat)
    # 计算时间域信号
    linear_E_t = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef_interp)))
    # 设置截取输入的索引
    indexmin = np.argmin(np.abs(fmat - 1.26e16))
    indexmax = np.argmin(np.abs(fmat - 9.34e16))

    return hertz, linear_E_t, tmat, fmat, Ef_interp, indexmin, indexmax, f0, N, dt

def retrieve_spectrum3(plotting=False):
    # 读取 spec.p 文件，更新文件路径
    with open(r"C:\Users\Always\Desktop\ai4science\spec.p", "rb") as file:
        spec_data = pickle.load(file)

    # 将电离势添加到电子 eV 轴
    spec_data["electron"]["eV"] = np.array(spec_data["electron"]["eV"]) + params.Ip_eV

    electronvolts = spec_data["electron"]["eV"]
    Intensity = spec_data["electron"]["I"]

    hertz, linear_E_t, tmat, fmat, Ef_interp, indexmin, indexmax, f0, N, dt = my_interp(
        electronvolts_in=electronvolts, intensity_in=Intensity, plotting=plotting
    )

    electronvolts = spec_data["photon"]["eV"]
    Intensity = spec_data["photon"]["I"]

    _, _, _, _, Ef_interp_photon, _, _, _, _, _ = my_interp(
        electronvolts_in=electronvolts, intensity_in=Intensity, plotting=plotting
    )

    # 将 xuv 参数转换为原子单位
    params_xuv = {}
    params_xuv['tmat'] = tmat / sc.physical_constants['atomic unit of time'][0]
    params_xuv['fmat'] = fmat * sc.physical_constants['atomic unit of time'][0]
    params_xuv['Ef'] = Ef_interp
    params_xuv['Ef_photon'] = Ef_interp_photon
    params_xuv['indexmin'] = indexmin
    params_xuv['indexmax'] = indexmax
    params_xuv['f0'] = f0 * sc.physical_constants['atomic unit of time'][0] + 0.2
    params_xuv['N'] = N
    params_xuv['dt'] = dt / sc.physical_constants['atomic unit of time'][0]

    return params_xuv

# 定义 xuv_spectrum.spectrum
class XUVSpectrum:
    params_xuv = retrieve_spectrum3(plotting=False)
    tmat = params_xuv['tmat']
    fmat = params_xuv['fmat']
    Ef = params_xuv['Ef']
    Ef_photon = params_xuv['Ef_photon']
    indexmin = params_xuv['indexmin']
    indexmax = params_xuv['indexmax']
    f0 = params_xuv['f0']
    N = params_xuv['N']
    dt = params_xuv['dt']

xuv_spectrum = type('xuv_spectrum', (), {'spectrum': XUVSpectrum})

# 定义 ir_spectrum.ir_spectrum
class IRSpectrum:
    # SI units for defining parameters
    W = 1
    cm = 1e-2
    um = 1e-6
    fs = 1e-15
    atts = 1e-18

    # 脉冲参数
    N = 128
    tmax = 50e-15
    start_index = 64
    end_index = 84

    # 离散化时间矩阵
    dt = tmax / N
    tmat = dt * np.arange(-N / 2, N / 2, 1)
    tmat_indexes = np.arange(int(-N / 2), int(N / 2), 1)

    # 离散化频谱矩阵
    df = 1 / (dt * N)
    fmat = df * np.arange(-N / 2, N / 2, 1)

    # 将单位转换为原子单位（AU）
    df = df * sc.physical_constants['atomic unit of time'][0]
    dt = dt / sc.physical_constants['atomic unit of time'][0]
    tmat = tmat / sc.physical_constants['atomic unit of time'][0]
    fmat = fmat * sc.physical_constants['atomic unit of time'][0]

    fmat_cropped = fmat[start_index: end_index]

ir_spectrum = type('ir_spectrum', (), {'ir_spectrum': IRSpectrum})

# 定义 tf_ifft 和 tf_fft 函数
def tf_ifft(Ef_prop, shift, axis=0):
    Ef_prop_shifted = tf.signal.ifftshift(Ef_prop, axes=axis)
    Et_prop = tf.signal.ifft(Ef_prop_shifted)
    Et_prop = tf.signal.fftshift(Et_prop, axes=axis)
    return Et_prop

def tf_fft(Et_prop, shift, axis=0):
    Et_prop_shifted = tf.signal.ifftshift(Et_prop, axes=axis)
    Ef_prop = tf.signal.fft(Et_prop_shifted)
    Ef_prop = tf.signal.fftshift(Ef_prop, axes=axis)
    return Ef_prop

# 定义 xuv_taylor_to_E 函数
def xuv_taylor_to_E(coefficients_in):
    assert int(coefficients_in.shape[1]) == params.xuv_phase_coefs

    amplitude = params.amplitude

    Ef = tf.constant(xuv_spectrum.spectrum.Ef, dtype=tf.complex64)
    Ef = tf.reshape(Ef, [1, -1])
    Ef_photon = tf.constant(xuv_spectrum.spectrum.Ef_photon, dtype=tf.complex64)
    Ef_photon = tf.reshape(Ef_photon, [1, -1])

    fmat_taylor = tf.constant(xuv_spectrum.spectrum.fmat - xuv_spectrum.spectrum.f0, dtype=tf.float32)

    # 创建阶乘
    factorials = tf.constant(factorial(np.array(range(coefficients_in.shape[1])) + 1), dtype=tf.float32)
    factorials = tf.reshape(factorials, [1, -1, 1])

    # 创建指数
    exponents = tf.constant(np.array(range(coefficients_in.shape[1])) + 1, dtype=tf.float32)

    # 重新调整 fmat_taylor
    fmat_taylor = tf.reshape(fmat_taylor, [1, 1, -1])

    # 重新调整指数矩阵
    exp_mat = tf.reshape(exponents, [1, -1, 1])

    # 将 fmat 提升到指数幂
    exp_mat_fmat = tf.pow(fmat_taylor, exp_mat)

    # 缩放系数
    amplitude_mat = tf.constant(amplitude, dtype=tf.float32)
    amplitude_mat = tf.reshape(amplitude_mat, [1, -1, 1])

    # 按指数缩放幅度
    amplitude_scaler = tf.pow(amplitude_mat, exp_mat)

    # 额外的缩放器
    scaler_2 = tf.constant(np.array([0.0, 1.3, 0.15, 0.03, 0.01]).reshape(1, -1, 1), dtype=tf.float32)

    # 重新调整系数并缩放
    coef_values = tf.reshape(coefficients_in, [tf.shape(coefficients_in)[0], -1, 1]) * amplitude_scaler * scaler_2

    # 除以阶乘
    coef_div_fact = tf.divide(coef_values, factorials)

    # 乘以 fmat
    taylor_coefs_mat = coef_div_fact * exp_mat_fmat

    # 相位角，总和
    phasecurve = tf.reduce_sum(taylor_coefs_mat, axis=1)

    # 将相位角应用于 Ef
    Ef_prop = Ef * tf.exp(tf.complex(real=tf.zeros_like(phasecurve), imag=phasecurve))
    Ef_photon_prop = Ef_photon * tf.exp(tf.complex(real=tf.zeros_like(phasecurve), imag=phasecurve))

    # 进行傅里叶变换，得到时间域信号
    Et_prop = tf_ifft(Ef_prop, shift=int(xuv_spectrum.spectrum.N / 2), axis=1)
    Et_photon_prop = tf_ifft(Ef_photon_prop, shift=int(xuv_spectrum.spectrum.N / 2), axis=1)

    # 截取 Ef_prop
    Ef_prop_cropped = Ef_prop[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]
    Ef_photon_prop_cropped = Ef_photon_prop[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]

    # 截取相位曲线
    phasecurve_cropped = phasecurve[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]

    E_prop = {
        "f": Ef_prop,
        "f_cropped": Ef_prop_cropped,
        "f_photon_cropped": Ef_photon_prop_cropped,
        "t": Et_prop,
        "t_photon": Et_photon_prop,
        "phasecurve_cropped": phasecurve_cropped
    }

    return E_prop

# 定义 ir_from_params 函数
def ir_from_params(ir_param_values):

    amplitudes = params.ir_param_amplitudes

    # 构建中间值和半范围
    parameters = {}
    for key in ["phase_range", "clambda_range", "pulseduration_range", "I_range"]:
        parameters[key] = {}

        # 获取变量的中间值和半范围
        parameters[key]["avg"] = (amplitudes[key][0] + amplitudes[key][1]) / 2
        parameters[key]["half_range"] = (amplitudes[key][1] - amplitudes[key][0]) / 2

        # 创建 TensorFlow 常量
        parameters[key]["tf_avg"] = tf.constant(parameters[key]["avg"], dtype=tf.float32)
        parameters[key]["tf_half_range"] = tf.constant(parameters[key]["half_range"], dtype=tf.float32)

    # 从标准化输入构建参数值
    scaled_tf_values = {}

    for i, key in enumerate(["phase_range", "clambda_range", "pulseduration_range", "I_range"]):
        scaled_tf_values[key.split("_")[0]] = parameters[key]["tf_avg"] + ir_param_values[:, i] * parameters[key]["tf_half_range"]

    # 转换为 SI 单位
    W = 1
    cm = 1e-2
    um = 1e-6
    fs = 1e-15
    atts = 1e-18

    scaled_tf_values_si = {}
    scaled_tf_values_si["I"] = scaled_tf_values["I"] * 1e13 * W / cm ** 2
    scaled_tf_values_si["f0"] = sc.c / (um * scaled_tf_values["clambda"])
    scaled_tf_values_si["t0"] = scaled_tf_values["pulseduration"] * fs

    # 计算 SI 单位下的平均光子能量（Up）
    Up = (sc.elementary_charge ** 2 * tf.abs(scaled_tf_values_si["I"])) / (
        2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * scaled_tf_values_si["f0"]) ** 2
    )

    # 转换为原子单位（AU）
    values_au = {}
    values_au["Up"] = Up / sc.physical_constants['atomic unit of energy'][0]
    values_au["f0"] = scaled_tf_values_si["f0"] * sc.physical_constants['atomic unit of time'][0]
    values_au["t0"] = scaled_tf_values_si["t0"] / sc.physical_constants['atomic unit of time'][0]

    # 计算 AU 下的驱动振幅
    E0 = tf.sqrt(4 * values_au["Up"] * (2 * np.pi * values_au["f0"]) ** 2)

    # 设置 AU 下的驱动 IR 场振幅
    tf_tmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.tmat, dtype=tf.float32), [1, -1])

    # 缓慢振荡包络
    Et_slow_osc = tf.reshape(E0, [-1, 1]) * tf.exp(
        -2 * np.log(2) * (tf_tmat / tf.reshape(values_au["t0"], [-1, 1])) ** 2
    )

    # 快速振荡包络
    phase = 2 * np.pi * tf.reshape(values_au["f0"], [-1, 1]) * tf_tmat
    Et_fast_osc = tf.exp(tf.complex(real=tf.zeros_like(phase), imag=phase))

    # 应用相位之前的脉冲
    Et = tf.complex(real=Et_slow_osc, imag=tf.zeros_like(Et_slow_osc)) * Et_fast_osc

    # 傅里叶变换
    Ef = tf_fft(Et, shift=int(len(ir_spectrum.ir_spectrum.tmat) / 2), axis=1)

    # 应用相位角
    phase_shift = tf.reshape(scaled_tf_values["phase"], [-1, 1])
    Ef_phase = Ef * tf.exp(tf.complex(real=tf.zeros_like(phase_shift), imag=phase_shift))

    # 逆傅里叶变换
    Et_phase = tf_ifft(Ef_phase, shift=int(len(ir_spectrum.ir_spectrum.tmat) / 2), axis=1)

    # 截取相位
    Ef_phase_cropped = Ef_phase[:, ir_spectrum.ir_spectrum.start_index:ir_spectrum.ir_spectrum.end_index]

    E_prop = {}
    E_prop["f"] = Ef_phase
    E_prop["f_cropped"] = Ef_phase_cropped
    E_prop["t"] = Et_phase

    out = {}
    out["scaled_values"] = scaled_tf_values
    out["E_prop"] = E_prop

    return out

# 定义 streaking_trace 函数
def streaking_trace(xuv_cropped_f_in, ir_cropped_f_in):

    # 定义收集 streaking trace 的角度
    theta_max = np.pi / 2
    N_theta = 10
    angle_in = tf.constant(np.linspace(0, theta_max, N_theta), dtype=tf.float32)
    Beta_in = 1

    # 电离势
    Ip = params.Ip

    #-----------------------------------------------------------------
    # 零填充 xuv 和 ir 频谱，以匹配完整的原始频率矩阵
    #-----------------------------------------------------------------
    paddings_xuv = tf.constant(
        [[xuv_spectrum.spectrum.indexmin, xuv_spectrum.spectrum.N - xuv_spectrum.spectrum.indexmax]], dtype=tf.int32)
    padded_xuv_f = tf.pad(xuv_cropped_f_in, paddings_xuv)
    # 同样适用于 IR
    paddings_ir = tf.constant(
        [[ir_spectrum.ir_spectrum.start_index, ir_spectrum.ir_spectrum.N - ir_spectrum.ir_spectrum.end_index]],
        dtype=tf.int32)
    padded_ir_f = tf.pad(ir_cropped_f_in, paddings_ir)
    # 对 xuv 进行傅里叶逆变换
    xuv_time_domain = tf_ifft(Ef_prop=padded_xuv_f, shift=int(xuv_spectrum.spectrum.N / 2))
    # 对 ir 进行傅里叶逆变换
    ir_time_domain = tf_ifft(Ef_prop=padded_ir_f, shift=int(ir_spectrum.ir_spectrum.N / 2))

    #------------------------------------------------------------------
    #------ 在频率空间中零填充 ir，以匹配 xuv 的时间步长-------
    #------------------------------------------------------------------
    # 计算匹配时间步长所需的 N 值
    N_req = int(1 / (xuv_spectrum.spectrum.dt * ir_spectrum.ir_spectrum.df))
    # 需要在每一侧填充的数量
    pad_2 = int((N_req - ir_spectrum.ir_spectrum.N) / 2)
    # 填充 IR 以匹配 xuv 的 dt
    paddings_ir_2 = tf.constant([[pad_2, pad_2]], dtype=tf.int32)
    padded_ir_2 = tf.pad(padded_ir_f, paddings_ir_2)
    # 计算匹配 dt 的 ir
    ir_t_matched_dt = tf_ifft(Ef_prop=padded_ir_2, shift=int(N_req / 2))
    # 匹配原始的尺度
    scale_factor = tf.constant(N_req / ir_spectrum.ir_spectrum.N, dtype=tf.complex64)
    ir_t_matched_dt_scaled = ir_t_matched_dt * scale_factor

    #------------------------------------------------------------------
    # ---------------------积分 ir 脉冲--------------------------
    #------------------------------------------------------------------
    A_t = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(tf.math.real(ir_t_matched_dt_scaled))

    # 对 A_L(t) 积分
    flipped1 = tf.reverse(A_t, axis=[0])
    flipped_integral = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1, axis=0)
    A_t_integ_t_phase = tf.reverse(flipped_integral, axis=[0])

    # 对 A_L(t)^2 积分
    flipped1_2 = tf.reverse(A_t ** 2, axis=[0])
    flipped_integral_2 = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1_2, axis=0)
    A_t_integ_t_phase_2 = tf.reverse(flipped_integral_2, axis=[0])

    # ------------------------------------------------------------------
    # ---------------------创建 ir t 轴-------------------------------
    # ------------------------------------------------------------------
    ir_taxis = xuv_spectrum.spectrum.dt * np.arange(-N_req / 2, N_req / 2, 1)

    # ------------------------------------------------------------------
    # ---------------------找到 tau 值的索引-------------------
    # ------------------------------------------------------------------
    center_indexes = []
    delay_vals_au = params.delay_values  # 已经是原子单位
    for delay_value in delay_vals_au:
        index = np.argmin(np.abs(delay_value - ir_taxis))
        center_indexes.append(index)
    center_indexes = np.array(center_indexes)
    rangevals = np.array(range(xuv_spectrum.spectrum.N)) - int((xuv_spectrum.spectrum.N / 2))
    delayindexes = center_indexes.reshape(1, -1) + rangevals.reshape(-1, 1)

    # ------------------------------------------------------------------
    # ------------从积分数组中收集值-------------------
    # ------------------------------------------------------------------
    ir_values = tf.gather(A_t_integ_t_phase, delayindexes.astype(np.int32))
    ir_values = tf.expand_dims(tf.expand_dims(ir_values, axis=0), axis=3)
    # 对于平方积分
    ir_values_2 = tf.gather(A_t_integ_t_phase_2, delayindexes.astype(np.int32))
    ir_values_2 = tf.expand_dims(tf.expand_dims(ir_values_2, axis=0), axis=3)

    #------------------------------------------------------------------
    #-------------------构建 streaking trace----------------------
    #------------------------------------------------------------------
    # 将 K 转换为原子单位
    K = params.K * sc.electron_volt  # 焦耳
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    K = K.reshape(-1, 1, 1, 1)
    K_tf = tf.constant(K, dtype=tf.float32)
    Ip_tf = tf.constant(Ip, dtype=tf.float32)
    tmat_tf = tf.constant(xuv_spectrum.spectrum.tmat, dtype=tf.float32)

    p = np.sqrt(2 * K).reshape(-1, 1, 1, 1)

    spec_angle = tf.reshape(tf.cos(angle_in), [1, 1, 1, -1])
    # 转换为 TensorFlow 张量
    p_tf = tf.constant(p, dtype=tf.float32)

    # 计算相位（确保数据类型兼容）
    phase = - (K_tf + Ip_tf) * tf.reshape(tmat_tf, [1, -1, 1, 1])

    # 将相位转换为复数类型，并计算 e_fft
    e_fft = tf.exp(tf.complex(real=tf.zeros_like(phase), imag=phase))

    # 添加用于积分的 xuv
    xuv_time_domain_integrate = tf.reshape(xuv_time_domain, [1, -1, 1, 1])

    # 计算角度分布项
    angular_distribution = 1 + (Beta_in / 2) * (3 * (tf.cos(angle_in)) ** 2 - 1)
    angular_distribution = tf.reshape(angular_distribution, [1, 1, 1, -1])
    angular_distribution = tf.complex(real=angular_distribution, imag=tf.zeros_like(angular_distribution))

    # 计算 ir_phi（确保 p_tf 和 ir_values 的数据类型为 float32）
    p_A_t_integ_t_phase3d = spec_angle * p_tf * ir_values + 0.5 * ir_values_2
    ir_phi = tf.exp(tf.complex(real=tf.zeros_like(p_A_t_integ_t_phase3d), imag=p_A_t_integ_t_phase3d))

    product = angular_distribution * xuv_time_domain_integrate * ir_phi * e_fft
    # 在 xuv 时间上积分
    integration = tf.constant(xuv_spectrum.spectrum.dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)
    # 对矩阵取绝对值平方
    image_not_scaled = tf.square(tf.abs(integration))
    image_not_scaled = image_not_scaled * tf.reshape(tf.sin(angle_in), [1, 1, -1])

    # 在 theta 轴上积分
    dtheta = angle_in[1] - angle_in[0]
    theta_integration = dtheta * tf.reduce_sum(image_not_scaled, axis=2)

    scaled = theta_integration - tf.reduce_min(theta_integration)
    image = scaled / tf.reduce_max(scaled)

    return image

# 定义 generate_xuv_coefs 和 generate_ir_params 函数
def generate_xuv_coefs(batch_size, xuv_coefs_num):
    return np.random.uniform(-1, 1, size=(batch_size, xuv_coefs_num))

def generate_ir_params(batch_size):
    return np.random.uniform(-1, 1, size=(batch_size, 4))

# 定义数据生成函数，生成 X_train 和 y_train
def generate_samples_to_csv(tf_graphs, n_samples, xuv_coefs_num, sess, X_csv_filename, y_csv_filename, batch_size=1000):
    import csv

    # 打开 CSV 文件，准备写入数据
    with open(X_csv_filename, mode='w', newline='') as X_csv_file, \
         open(y_csv_filename, mode='w', newline='') as y_csv_file:

        X_writer = csv.writer(X_csv_file)
        y_writer = csv.writer(y_csv_file)

        # 计算总的批次数
        num_batches = n_samples // batch_size
        remaining_samples = n_samples % batch_size

        for batch_index in range(num_batches):
            print(f"正在生成第 {batch_index+1}/{num_batches} 批数据...")
            # 生成一批数据
            batch_size_current = batch_size
            xuv_coefs_data = generate_xuv_coefs(batch_size_current, xuv_coefs_num)
            ir_params_data = generate_ir_params(batch_size_current)
            images = []
            labels = []

            for i in range(batch_size_current):
                feed_dict = {
                    tf_graphs['xuv_coefs_in']: xuv_coefs_data[i:i+1],
                    tf_graphs['ir_values_in']: ir_params_data[i:i+1]
                }
                image_out, xuv_E_prop_out, ir_E_prop_out = sess.run(
                    [tf_graphs['image'], tf_graphs['xuv_E_prop'], tf_graphs['ir_E_prop']], feed_dict=feed_dict)

                # 调整图像尺寸（截取中间的 58 列）
                image_cropped = image_out[:, 20:78]  # 58 列
                images.append(image_cropped.flatten())

                # 提取 XUV 和 IR 频谱的复数部分
                xuv_f = xuv_E_prop_out['f_cropped'][0]  # 取第一个样本
                ir_f = ir_E_prop_out['f_cropped'][0]

                # 调整 xuv_f 和 ir_f 的长度
                xuv_f = xuv_f[:125]  # 截取前 125 个数据点
                ir_f = ir_f[:20]    # 截取前 20 个数据点

                # 将复数频谱展开为实部和虚部
                xuv_f_real = np.real(xuv_f)
                xuv_f_imag = np.imag(xuv_f)
                ir_f_real = np.real(ir_f)
                ir_f_imag = np.imag(ir_f)

                # 合并为一个向量，长度为 125 + 125 + 20 + 20 = 290
                label = np.concatenate([xuv_f_real, xuv_f_imag, ir_f_real, ir_f_imag])
                labels.append(label)

            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)

            # 将数据写入 CSV 文件
            X_writer.writerows(images)
            y_writer.writerows(labels)

        # 处理剩余的样本
        if remaining_samples > 0:
            print(f"正在生成最后一批数据，样本数量：{remaining_samples}")
            xuv_coefs_data = generate_xuv_coefs(remaining_samples, xuv_coefs_num)
            ir_params_data = generate_ir_params(remaining_samples)
            images = []
            labels = []

            for i in range(remaining_samples):
                feed_dict = {
                    tf_graphs['xuv_coefs_in']: xuv_coefs_data[i:i+1],
                    tf_graphs['ir_values_in']: ir_params_data[i:i+1]
                }
                image_out, xuv_E_prop_out, ir_E_prop_out = sess.run(
                    [tf_graphs['image'], tf_graphs['xuv_E_prop'], tf_graphs['ir_E_prop']], feed_dict=feed_dict)

                # 调整图像尺寸（截取中间的 58 列）
                image_cropped = image_out[:, 20:78]  # 58 列
                images.append(image_cropped.flatten())

                # 提取 XUV 和 IR 频谱的复数部分
                xuv_f = xuv_E_prop_out['f_cropped'][0]  # 取第一个样本
                ir_f = ir_E_prop_out['f_cropped'][0]

                # 调整 xuv_f 和 ir_f 的长度
                xuv_f = xuv_f[:125]  # 截取前 125 个数据点
                ir_f = ir_f[:20]    # 截取前 20 个数据点

                # 将复数频谱展开为实部和虚部
                xuv_f_real = np.real(xuv_f)
                xuv_f_imag = np.imag(xuv_f)
                ir_f_real = np.real(ir_f)
                ir_f_imag = np.imag(ir_f)

                # 合并为一个向量，长度为 125 + 125 + 20 + 20 = 290
                label = np.concatenate([xuv_f_real, xuv_f_imag, ir_f_real, ir_f_imag])
                labels.append(label)

            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)

            # 将数据写入 CSV 文件
            X_writer.writerows(images)
            y_writer.writerows(labels)



# 主程序
if __name__ == "__main__":
    # 处理 TensorFlow 版本兼容性
    tf.compat.v1.disable_eager_execution()

    # 初始化 XUV 生成器
    xuv_coefs_in = tf.compat.v1.placeholder(tf.float32, shape=[None, params.xuv_phase_coefs])
    xuv_E_prop = xuv_taylor_to_E(xuv_coefs_in)

    # 初始化 IR 生成器
    ir_values_in = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
    ir_E_prop = ir_from_params(ir_values_in)["E_prop"]

    # 构建 streaking 图像
    image = streaking_trace(
        xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
        ir_cropped_f_in=ir_E_prop["f_cropped"][0]
    )

    tf_graphs = {
        "xuv_coefs_in": xuv_coefs_in,
        "ir_values_in": ir_values_in,
        "xuv_E_prop": xuv_E_prop,
        "ir_E_prop": ir_E_prop,
        "image": image,
    }

    with tf.compat.v1.Session() as sess:
        total_samples = 80000
        X_csv_filename = r"C:\Users\Always\Desktop\ai4science/X_train.csv"
        y_csv_filename = r"C:\Users\Always\Desktop\ai4science/y_train.csv"
        generate_samples_to_csv(
            tf_graphs=tf_graphs,
            n_samples=total_samples,
            xuv_coefs_num=params.xuv_phase_coefs,
            sess=sess,
            X_csv_filename=X_csv_filename,
            y_csv_filename=y_csv_filename,
            batch_size=100  # 根据您的内存情况调整批次大小
        )

    # 加载生成的数据
    import pandas as pd

    # 加载 X_train
    X_train = pd.read_csv(X_csv_filename, header=None)
    num_samples = X_train.shape[0]
    X_train = X_train.values.reshape(-1, 301, 58, 1).astype(np.float32)

    # 加载 y_train
    y_train = pd.read_csv(y_csv_filename, header=None)
    y_train = y_train.values.astype(np.float32)

    # 检查数据形状
    print("X_train shape:", X_train.shape)  # (样本数量, 301, 58, 1)
    print("y_train shape:", y_train.shape)  # (样本数量, 290)

    # 获取标签维度
    output_dim = y_train.shape[1]



