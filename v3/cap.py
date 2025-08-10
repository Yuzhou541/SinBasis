#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cap.py

基于原论文《Attosecond streaking phase retrieval with neural network》的训练流程，
将 CNN 网络替换为 capsule network，整体框架复杂度保持相似。

训练流程包括：
1. 监督学习阶段：使用电脑生成的噪声数据集 (train3.hdf5 / test3.hdf5)，每个 epoch 遍历所有样本，
   使用 Adam 优化器最小化均方误差；
2. 无监督微调阶段（可选）：对实验测量的 streaking trace 进行微调，使重构迹与输入更吻合。

依赖模块：tensorflow, tf_functions, phase_parameters.params, unsupervised_retrieval, tables, matplotlib, numpy, os, shutil
"""

import os
import sys
import shutil
import numpy as np
import tables
import tensorflow as tf
import matplotlib.pyplot as plt

# 导入工程内部模块（确保这些模块在 PYTHONPATH 中可找到）
import tf_functions
import phase_parameters.params as params
import unsupervised_retrieval
import measured_trace.get_trace as get_measured_trace

# 关闭 eager 执行（TF1.x 风格）
tf.compat.v1.disable_eager_execution()


###############################################################################
# 数据获取类：按批次读取 HDF5 文件（用于训练和测试）
###############################################################################
class GetData:
    def __init__(self, train_file, batch_size):
        self.train_file = train_file
        self.batch_size = batch_size
        h5 = tables.open_file(self.train_file, mode="r")
        self.samples = h5.root.noise_trace.shape[0]
        h5.close()
        self.batch_index = 0

    def next_batch(self):
        h5 = tables.open_file(self.train_file, mode="r")
        x_batch = h5.root.noise_trace[self.batch_index:self.batch_index + self.batch_size, :]
        xuv_coefs = h5.root.xuv_coefs[self.batch_index:self.batch_index + self.batch_size, :]
        ir_params = h5.root.ir_params[self.batch_index:self.batch_index + self.batch_size, :]
        y_batch = np.append(xuv_coefs, ir_params, axis=1)
        h5.close()
        self.batch_index += self.batch_size
        return x_batch.astype(np.float32), y_batch.astype(np.float32)

    def reset(self):
        self.batch_index = 0


class GetTestData:
    def __init__(self, test_file):
        self.test_file = test_file
        h5 = tables.open_file(self.test_file, mode="r")
        self.x_data = h5.root.noise_trace[:]
        xuv_coefs = h5.root.xuv_coefs[:]
        ir_params = h5.root.ir_params[:]
        self.y_data = np.append(xuv_coefs, ir_params, axis=1)
        h5.close()

    def get_all(self):
        return self.x_data.astype(np.float32), self.y_data.astype(np.float32)


###############################################################################
# 辅助函数：卷积层与池化层
###############################################################################
def convolutional_layer(input_x, filter_shape, stride, activation='relu', padding='SAME', name="conv"):
    with tf.compat.v1.variable_scope(name):
        W = tf.compat.v1.get_variable("W", shape=filter_shape,
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        b = tf.compat.v1.get_variable("b", shape=[filter_shape[-1]],
                initializer=tf.compat.v1.constant_initializer(0.1))
        conv = tf.nn.conv2d(input_x, W, strides=[1, stride[0], stride[1], 1], padding=padding) + b
        if activation == 'relu':
            return tf.nn.relu(conv)
        elif activation == 'leaky':
            return tf.nn.leaky_relu(conv, alpha=0.01)
        else:
            return conv

def max_pooling_layer(input_x, pool_size, strides, padding='SAME', name="pool"):
    with tf.compat.v1.variable_scope(name):
        return tf.nn.max_pool2d(input_x, ksize=[1, pool_size[0], pool_size[1], 1],
                           strides=[1, strides[0], strides[1], 1], padding=padding)


###############################################################################
# Capsule Network 辅助函数：squash 非线性函数
###############################################################################
def squash(s, axis=-1, epsilon=1e-7):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
    scale = squared_norm / (1.0 + squared_norm) / tf.sqrt(squared_norm + epsilon)
    return scale * s


###############################################################################
# Capsule 层（全连接 capsule 层，使用动态路由）
###############################################################################
def capsule_layer(input_caps, num_capsule, dim_capsule, num_routing, scope):
    """
    构建 capsule 层
    input_caps: [batch_size, num_caps_input, dim_caps_input]
    输出: [batch_size, num_capsule, dim_capsule]
    """
    with tf.compat.v1.variable_scope(scope):
        batch_size = tf.shape(input_caps)[0]
        num_caps_input = input_caps.get_shape().as_list()[1]
        dim_caps_input = input_caps.get_shape().as_list()[2]
        # 定义变换矩阵，形状 [1, num_caps_input, num_capsule, dim_capsule, dim_caps_input]
        W = tf.compat.v1.get_variable("W", shape=[1, num_caps_input, num_capsule, dim_capsule, dim_caps_input],
                           initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        # 扩展 input_caps 维度：
        # 首先扩展为列向量: [batch, num_caps_input, dim_caps_input, 1]
        input_caps_expanded = tf.expand_dims(input_caps, -1)
        # 再在 axis=2 插入一个维度，得到 [batch, num_caps_input, 1, dim_caps_input, 1]
        input_caps_expanded = tf.expand_dims(input_caps_expanded, 2)
        # 将 W tile 为 batch_size 份
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1])
        # 计算预测向量 u_hat: [batch, num_caps_input, num_capsule, dim_capsule, 1]
        u_hat = tf.matmul(W_tiled, input_caps_expanded)
        u_hat = tf.squeeze(u_hat, axis=-1)  # shape: [batch, num_caps_input, num_capsule, dim_capsule]
        
        # 初始化路由 logits b: [batch, num_caps_input, num_capsule]
        b = tf.zeros([batch_size, num_caps_input, num_capsule], dtype=tf.float32)
        
        # 动态路由
        def routing_body(i, b, u_hat, v):
            c = tf.nn.softmax(b, axis=2)  # [batch, num_caps_input, num_capsule]
            c_expanded = tf.expand_dims(c, -1)  # [batch, num_caps_input, num_capsule, 1]
            s = tf.reduce_sum(c_expanded * u_hat, axis=1)  # [batch, num_capsule, dim_capsule]
            v = squash(s)
            v_expanded = tf.expand_dims(v, 1)  # [batch, 1, num_capsule, dim_capsule]
            b += tf.reduce_sum(u_hat * v_expanded, axis=-1)  # 更新 b
            return i+1, b, u_hat, v

        v_initial = tf.zeros([batch_size, num_capsule, dim_capsule], tf.float32)
        i = 0
        cond = lambda i, b, u_hat, v: tf.less(i, num_routing)
        _, b, u_hat, v = tf.while_loop(cond, routing_body, loop_vars=[i, b, u_hat, v_initial])
    return v  # shape: [batch, num_capsule, dim_capsule]


###############################################################################
# 构建 Capsule Network 网络结构
###############################################################################
def build_network(x, is_training):
    """
    输入：展平的 streaking trace（尺寸 301x98=29498）
    输出：9 维向量 [xuv_coefs (5), ir_params (4)]
    
    网络先将输入 reshape 为 2D 图像，然后利用卷积层提取低级特征，
    接着构造 Primary Capsules（将卷积输出按 capsule 维度重构），
    再通过全连接 capsule 层（动态路由）输出一个 capsule，其 9 个维度作为回归输出。
    """
    # reshape 输入为 2D 图像： [None, 301, 98, 1]
    input_image = tf.reshape(x, [-1, 301, 98, 1])
    
    # 第一层卷积：采用 stride [2,2] 以大幅下采样
    conv1 = convolutional_layer(input_image, filter_shape=[9, 9, 1, 64], stride=[2, 2],
                                activation='relu', padding='SAME', name="conv1")
    pool1 = max_pooling_layer(conv1, pool_size=[2, 2], strides=[2, 2],
                              padding='SAME', name="pool1")
    
    # Primary Capsules 层：利用卷积生成 capsule 特征，采用 stride [2,2] 进一步降采样
    primary_caps = convolutional_layer(pool1, filter_shape=[9, 9, 64, 64], stride=[2, 2],
                                       activation='relu', padding='SAME', name="primary_caps")
    primary_caps_shape = primary_caps.get_shape().as_list()
    if primary_caps_shape[1] is None or primary_caps_shape[2] is None:
        height = tf.shape(primary_caps)[1]
        width = tf.shape(primary_caps)[2]
    else:
        height = primary_caps_shape[1]
        width = primary_caps_shape[2]
    caps_dim = 8
    # 每个 spatial 位置上有 64//8 个 capsule
    num_capsules_per_loc = 64 // caps_dim
    # 重塑为 [batch, height * width * num_capsules_per_loc, caps_dim]
    primary_caps_reshaped = tf.reshape(primary_caps, [-1, height * width * num_capsules_per_loc, caps_dim])
    
    # 全连接 Capsule 层：汇聚所有 Primary Capsule 到一个 capsule，输出维度为 9
    num_caps_out = 1
    dim_caps_out = 9
    final_caps = capsule_layer(primary_caps_reshaped, num_capsule=num_caps_out, dim_capsule=dim_caps_out,
                               num_routing=3, scope="digit_caps")
    # final_caps 的 shape 为 [batch, 1, 9]，reshape 为 [batch, 9]
    output = tf.reshape(final_caps, [-1, dim_caps_out])
    
    # 可选：使用 dropout（训练时启用）
    def apply_dropout():
        return tf.compat.v1.layers.dropout(output, rate=0.5, training=True)
    def no_dropout():
        return output
    output = tf.cond(is_training, apply_dropout, no_dropout)
    return output


###############################################################################
# 重构 streaking trace（用于无监督微调阶段）
###############################################################################
def reconstruct_trace(pred_params):
    # 分离 xuv_coefs 和 ir_params
    xuv_coefs_pred = pred_params[:, :5]
    ir_params_pred = pred_params[:, 5:]
    zeros = tf.zeros([tf.shape(xuv_coefs_pred)[0], 1], dtype=tf.float32)
    xuv_coefs_nolin = tf.concat([zeros, xuv_coefs_pred[:, 1:]], axis=1)
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_nolin)
    ir_E_prop = tf_functions.ir_from_params(ir_params_pred)["E_prop"]
    trace = tf_functions.streaking_trace(
        xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
        ir_cropped_f_in=ir_E_prop["f_cropped"][0]
    )
    trace_flat = tf.reshape(trace, [1, -1])
    return trace_flat


###############################################################################
# 训练流程
###############################################################################
def train_network(train_file, test_file, epochs=40, batch_size=10, lr=1e-4, model_dir="models"):
    x_ph = tf.compat.v1.placeholder(tf.float32, [None, 29498], name="x_input")
    y_ph = tf.compat.v1.placeholder(tf.float32, [None, 9], name="y_target")
    is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
    
    pred_params = build_network(x_ph, is_training)
    sup_loss = tf.reduce_mean(tf.square(pred_params - y_ph))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(sup_loss)
    
    reconstructed = reconstruct_trace(pred_params)
    unsup_loss = tf.reduce_mean(tf.square(reconstructed - x_ph))
    unsup_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    unsup_train_op = unsup_optimizer.minimize(unsup_loss)
    
    train_data = GetData(train_file, batch_size=batch_size)
    test_data = GetTestData(test_file)
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    
    # 使用 GPU 内存按需增长配置
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        num_train_samples = train_data.samples
        steps_per_epoch = num_train_samples // batch_size
        
        print("开始监督学习训练：共 {} 个 epoch，每个 epoch {} 个步骤".format(epochs, steps_per_epoch))
        for epoch in range(epochs):
            train_loss_epoch = 0.0
            train_data.reset()
            for step in range(steps_per_epoch):
                x_batch, y_batch = train_data.next_batch()
                feed = {x_ph: x_batch, y_ph: y_batch, is_training: True}
                _, loss_val = sess.run([train_op, sup_loss], feed_dict=feed)
                train_loss_epoch += loss_val
                if (step+1) % 200 == 0:
                    print("Epoch {} Step {}/{}: Batch Loss = {:.6f}".format(epoch+1, step+1, steps_per_epoch, loss_val))
            avg_loss = train_loss_epoch / steps_per_epoch
            print(">>> Epoch {} 完成，训练 MSE = {:.6f}".format(epoch+1, avg_loss))
            x_test, y_test = test_data.get_all()
            test_loss = sess.run(sup_loss, feed_dict={x_ph: x_test, y_ph: y_test, is_training: False})
            print("=== 测试集 MSE = {:.6f} ===".format(test_loss))
            saver.save(sess, os.path.join(model_dir, "cap_model.ckpt"), global_step=epoch+1)
        
        print("监督学习训练完成。")
        
        print("开始无监督微调阶段……")
        measured_trace = test_data.get_all()[0][0].reshape(1, -1)
        for i in range(100):
            feed_unsup = {x_ph: measured_trace, is_training: True}
            _, unsup_loss_val = sess.run([unsup_train_op, unsup_loss], feed_dict=feed_unsup)
            if (i+1) % 10 == 0:
                print("无监督训练 Step {}: Loss = {:.6f}".format(i+1, unsup_loss_val))
        
        print("无监督微调完成。")
    
    train_data.reset()


###############################################################################
# 主函数入口
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "cap_experiment"
    model_directory = os.path.join("models", model_name)
    
    # 修改下面路径为实际 HDF5 数据集的路径（确保文件存在）
    train_hdf5_path = r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\train3.hdf5"
    test_hdf5_path  = r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\test3.hdf5"
    
    train_network(train_file=train_hdf5_path, test_file=test_hdf5_path,
                  epochs=70, batch_size=10, lr=1e-4, model_dir=model_directory)
