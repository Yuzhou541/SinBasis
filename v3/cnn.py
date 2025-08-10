#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cnn.py

复现论文《Attosecond streaking phase retrieval with neural network》的训练流程：
1. 监督学习阶段：使用电脑生成的噪声数据集（训练集：train3.hdf5，测试集：test3.hdf5），
   每个 epoch 遍历所有样本（batch size=10，共40个 epoch），采用 Adam 优化器最小化均方误差。
2. 无监督微调阶段（可选）：对实验测量的 streaking trace 进行微调，使生成的重构迹与输入更吻合。

依赖模块：tensorflow, tf_functions, phase_parameters.params, unsupervised_retrieval, tables, matplotlib, numpy, os, shutil
"""

import os
import sys
import shutil
import numpy as np
import tables
import tensorflow as tf
import matplotlib.pyplot as plt

# 导入你工程中的模块（确保这些模块在 PYTHONPATH 中可找到）
import tf_functions
import phase_parameters.params as params
import unsupervised_retrieval
import measured_trace.get_trace as get_measured_trace

# 关闭 eager 执行（使用 TF1.x 风格）
tf.compat.v1.disable_eager_execution()


###############################################################################
# 数据获取类：按批次读取 HDF5 文件（此处用于训练和测试）
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
        # 打开 HDF5 文件，读取当前批次数据
        h5 = tables.open_file(self.train_file, mode="r")
        x_batch = h5.root.noise_trace[self.batch_index:self.batch_index + self.batch_size, :]
        # 训练标签为拼接的 xuv_coefs (5维) 和 ir_params (4维)，共9维
        xuv_coefs = h5.root.xuv_coefs[self.batch_index:self.batch_index + self.batch_size, :]
        ir_params = h5.root.ir_params[self.batch_index:self.batch_index + self.batch_size, :]
        y_batch = np.append(xuv_coefs, ir_params, axis=1)
        h5.close()
        self.batch_index += self.batch_size
        # 转换数据类型为 float32
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
# CNN 网络结构
###############################################################################
def convolutional_layer(input_x, filter_shape, stride, activation='relu', padding='SAME', name="conv"):
    with tf.compat.v1.variable_scope(name):
        # 初始化权重和偏置
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

def dense_layer(input_x, units, activation='relu', name="dense"):
    with tf.compat.v1.variable_scope(name):
        input_dim = int(input_x.get_shape()[1])
        W = tf.compat.v1.get_variable("W", shape=[input_dim, units],
                              initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        b = tf.compat.v1.get_variable("b", shape=[units],
                              initializer=tf.compat.v1.constant_initializer(0.1))
        dense = tf.matmul(input_x, W) + b
        if activation == 'relu':
            return tf.nn.relu(dense)
        else:
            return dense

def build_network(x, is_training):
    """
    构建网络：输入为 flatten streaking trace（尺寸 301x98=29498），
    输出为 9 维向量 [xuv_coefs(5), ir_params(4)].
    网络结构采用多个卷积块（使用不同滤波器尺寸，多分支），再连接两个全连接层，
    第一层 1024 个神经元，最后一层输出 9 维（监督学习目标）。
    """
    # reshape 输入为 2D 图像： shape [None, 301, 98, 1]
    input_image = tf.reshape(x, [-1, 301, 98, 1])

    # 第一组卷积块：模拟论文中多个滤波器尺寸的并行卷积（例如使用 filter sizes 21x8, 13x5等）
    with tf.compat.v1.variable_scope("block1"):
        conv_a = convolutional_layer(input_image, filter_shape=[21, 8, 1, 20], stride=[1,1], activation='relu', name="conv_a")
        pool_a = max_pooling_layer(conv_a, pool_size=[13, 5], strides=[3,3], name="pool_a")
    # 第二组卷积块
    with tf.compat.v1.variable_scope("block2"):
        conv_b1 = convolutional_layer(pool_a, filter_shape=[13, 5, 20, 40], stride=[1,1], activation='relu', name="conv_b1")
        pool_b = max_pooling_layer(conv_b1, pool_size=[9, 3], strides=[2,2], name="pool_b")
    # 第三组：进一步提取特征（这里简化了分支结构，可根据论文调整）
    with tf.compat.v1.variable_scope("block3"):
        conv_c = convolutional_layer(pool_b, filter_shape=[3, 3, 40, 40], stride=[1,1], activation='relu', name="conv_c")
        pool_c = max_pooling_layer(conv_c, pool_size=[2,2], strides=[2,2], name="pool_c")

    # 将卷积层输出展平
    flat = tf.compat.v1.layers.flatten(pool_c)

    # 全连接层：第一层 1024 个神经元
    fc1 = dense_layer(flat, units=1024, activation='relu', name="fc1")
    # dropout
    fc1_drop = tf.compat.v1.layers.dropout(fc1, rate=0.5, training=is_training)

    # 输出层：9维向量（对应 xuv_coefs (5) + ir_params (4)）
    output = dense_layer(fc1_drop, units=9, activation=None, name="fc_out")
    return output

###############################################################################
# 重构 streaking trace（用于无监督学习阶段）
###############################################################################
def reconstruct_trace(pred_params):
    # 分离 xuv_coefs 和 ir_params
    xuv_coefs_pred = pred_params[:, :5]
    ir_params_pred = pred_params[:, 5:]
    # 对 xuv_coefs_pred 前加上 0（无线性相位），如果转换函数需要这样
    zeros = tf.zeros([tf.shape(xuv_coefs_pred)[0], 1], dtype=tf.float32)
    xuv_coefs_nolin = tf.concat([zeros, xuv_coefs_pred[:, 1:]], axis=1)
    # 利用转换函数生成复数场
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_nolin)
    ir_E_prop = tf_functions.ir_from_params(ir_params_pred)["E_prop"]
    # 生成 streaking trace（默认输出形状为 [301, 98]）
    trace = tf_functions.streaking_trace(
        xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
        ir_cropped_f_in=ir_E_prop["f_cropped"][0]
    )
    # 将二维 trace 展平为 [1, 29498]，以匹配输入 x_ph 的形状
    trace_flat = tf.reshape(trace, [1, -1])
    return trace_flat


###############################################################################
# 训练流程
###############################################################################
def train_network(train_file, test_file, epochs=40, batch_size=10, lr=1e-4, model_dir="models"):
    # 定义占位符
    x_ph = tf.compat.v1.placeholder(tf.float32, [None, 29498], name="x_input")
    y_ph = tf.compat.v1.placeholder(tf.float32, [None, 9], name="y_target")
    is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
    
    # 构建网络，输出 predicted 参数
    pred_params = build_network(x_ph, is_training)
    
    # 定义监督学习损失：MSE（对应论文 Eq. (6)）
    sup_loss = tf.reduce_mean(tf.square(pred_params - y_ph))
    
    # 定义优化器（Adam），采用动态学习率
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(sup_loss)
    
    # 用于监督微调无监督阶段的重构误差：重构的 streaking trace 与输入的迹之间的 MSE
    reconstructed = reconstruct_trace(pred_params)
    # 输入 streaking trace 的形状 [29498]，重构 trace 需 reshape 成相同形状
    unsup_loss = tf.reduce_mean(tf.square(reconstructed - x_ph))
    unsup_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    unsup_train_op = unsup_optimizer.minimize(unsup_loss)
    
    # 数据读取器
    train_data = GetData(train_file, batch_size=batch_size)
    test_data = GetTestData(test_file)
    
    # 模型保存
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        num_train_samples = train_data.samples
        steps_per_epoch = num_train_samples // batch_size
        
        print("开始监督学习训练：40个 epoch，每个 epoch 有 {} 个步骤".format(steps_per_epoch))
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
            # 每个 epoch 结束后，可在测试集上评估
            x_test, y_test = test_data.get_all()
            test_loss = sess.run(sup_loss, feed_dict={x_ph: x_test, y_ph: y_test, is_training: False})
            print("=== 测试集 MSE = {:.6f} ===".format(test_loss))
            # 保存模型
            saver.save(sess, os.path.join(model_dir, "cnn_model.ckpt"), global_step=epoch+1)
        
        print("监督学习训练完成。")
        
        # 可选：无监督微调阶段，对实验测量的 streaking trace 进行优化
        # 这里示例取测试集中的第一个样本作为 measured trace 进行微调
        print("开始无监督微调阶段……")
        measured_trace = test_data.get_all()[0][0].reshape(1, -1)
        # 对无监督训练进行若干步迭代（根据实验数据调整步数，此处示例迭代100步）
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
    # 从命令行参数获取模型名称（用于保存模型文件），否则使用默认名称
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "cnn_experiment"
    model_directory = os.path.join("models", model_name)
    
    # HDF5 数据集路径（请确保路径正确，文件存在）
    train_hdf5_path = r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\train3.hdf5"
    test_hdf5_path  = r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\test3.hdf5"
    
    # 调用训练函数：监督学习40个 epoch，batch size=10；学习率可根据需要调整
    train_network(train_file=train_hdf5_path, test_file=test_hdf5_path,
                  epochs=70, batch_size=10, lr=1e-4, model_dir=model_directory)
