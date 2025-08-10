#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hybrid.py

混合模型：结合 CNN 和 ViT 进行 attosecond streaking phase retrieval。
模型思路：
1. 输入：flatten 的 streaking trace（长度 29498，即 301×98），reshape 成图像 [301, 98, 1]；
2. CNN 部分：用三个卷积块（block1、block2、block3）提取局部特征；
3. ViT 部分：将 CNN 输出的特征图（假设形状 [batch, 26, 9, 40]）重塑为 token 序列，
   对每个 token 进行线性投影到 embed_dim（例如128），添加 [CLS] token 与位置编码，
   经过若干 Transformer block（例如 3 层），取 [CLS] token 输出，经全连接层映射到 9 维输出；
4. 训练：监督学习阶段采用均方误差作为损失，Adam 优化器训练；也可进行无监督微调阶段。

依赖模块：tensorflow, tf_functions, phase_parameters.params, unsupervised_retrieval, tables, matplotlib, numpy, os, shutil
"""

import os
import sys
import shutil
import numpy as np
import tables
import tensorflow as tf
import matplotlib.pyplot as plt

# 导入工程中已有的模块
import tf_functions
import phase_parameters.params as params
import unsupervised_retrieval
import measured_trace.get_trace as get_measured_trace

# 关闭 eager 执行（TF1.x 风格）
tf.compat.v1.disable_eager_execution()

###############################################################################
# 数据读取类：按批次读取 HDF5 文件（训练集及测试集均如此）
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
        # 标签为拼接的 xuv_coefs (5 维) 和 ir_params (4 维)，共 9 维
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
# 混合模型网络结构：CNN + ViT
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

# 导入 tf.keras.layers 的 Dense, LayerNormalization, Dropout 用于 Transformer 部分
from tensorflow.keras.layers import Dense as KDense, LayerNormalization, Dropout

def extract_patches(images, patch_size):
    """
    将图片 images [batch, height, width, channels] 划分为 patch，patch 大小为 patch_size×patch_size
    返回张量形状为 [batch, num_patches, patch_dim]
    """
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patch_dim = patch_size * patch_size * images.get_shape()[-1]
    batch_size = tf.shape(images)[0]
    patches = tf.reshape(patches, [batch_size, -1, patch_dim])
    return patches

def transformer_block(x, num_heads, mlp_dim, dropout_rate, name):
    """
    Transformer block：
      1. LayerNorm -> 多头自注意力 -> 残差连接
      2. LayerNorm -> MLP (Dense->GELU->Dropout->Dense->Dropout) -> 残差连接
    """
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        ln1 = LayerNormalization(epsilon=1e-6, name="ln1")(x)
        dim = x.get_shape().as_list()[-1]
        q = KDense(dim, name="q")(ln1)
        k = KDense(dim, name="k")(ln1)
        v = KDense(dim, name="v")(ln1)
        depth = dim // num_heads
        def split_heads(x):
            x = tf.reshape(x, [-1, tf.shape(x)[1], num_heads, depth])
            return tf.transpose(x, [0, 2, 1, 3])
        q_ = split_heads(q)
        k_ = split_heads(k)
        v_ = split_heads(v)
        scale = tf.sqrt(tf.cast(depth, tf.float32))
        attn_logits = tf.matmul(q_, k_, transpose_b=True) / scale
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_output = tf.matmul(attn_weights, v_)
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [-1, tf.shape(x)[1], dim])
        attn_output = Dropout(dropout_rate)(attn_output)
        x = x + attn_output

        ln2 = LayerNormalization(epsilon=1e-6, name="ln2")(x)
        mlp_output = KDense(mlp_dim, activation=tf.nn.gelu, name="mlp_dense1")(ln2)
        mlp_output = Dropout(dropout_rate)(mlp_output)
        mlp_output = KDense(dim, name="mlp_dense2")(mlp_output)
        mlp_output = Dropout(dropout_rate)(mlp_output)
        x = x + mlp_output
        return x

def build_hybrid_network(x, is_training):
    """
    混合模型：结合 CNN 和 ViT
    1. 输入 x: [None, 29498]，reshape 成 [None, 301, 98, 1]
    2. CNN 部分：三个卷积块
         - Block1: conv_a + pool_a
         - Block2: conv_b1 + pool_b
         - Block3: conv_c + pool_c
       假设 Block3 输出形状约为 [batch, 26, 9, 40]
    3. ViT 部分：
         - 重塑 CNN 特征图为 tokens [batch, 26*9, 40]
         - 对 tokens 进行线性投影到 embed_dim (128)
         - 添加 [CLS] token 和位置编码（形状 [1, num_tokens+1, embed_dim]）
         - 通过 3 个 Transformer block（num_heads=4, mlp_dim=256, dropout_rate=0.1）
         - 取 CLS token 输出，经全连接层映射到 9 维输出
    """
    # CNN 部分
    input_image = tf.reshape(x, [-1, 301, 98, 1])
    
    with tf.compat.v1.variable_scope("block1"):
        conv_a = convolutional_layer(input_image, filter_shape=[21, 8, 1, 20],
                                       stride=[1,1], activation='relu', name="conv_a")
        pool_a = max_pooling_layer(conv_a, pool_size=[13,5], strides=[3,3], name="pool_a")
    
    with tf.compat.v1.variable_scope("block2"):
        conv_b1 = convolutional_layer(pool_a, filter_shape=[13,5,20,40],
                                       stride=[1,1], activation='relu', name="conv_b1")
        pool_b = max_pooling_layer(conv_b1, pool_size=[9,3], strides=[2,2], name="pool_b")
    
    with tf.compat.v1.variable_scope("block3"):
        conv_c = convolutional_layer(pool_b, filter_shape=[3,3,40,40],
                                      stride=[1,1], activation='relu', name="conv_c")
        pool_c = max_pooling_layer(conv_c, pool_size=[2,2], strides=[2,2], name="pool_c")
    # 假设 pool_c 输出形状为 [batch, 26, 9, 40]
    shape = pool_c.get_shape().as_list()   # [batch, 26, 9, 40]
    tokens = tf.reshape(pool_c, [-1, shape[1]*shape[2], shape[3]])  # [batch, 26*9, 40]
    
    # ViT 部分
    embed_dim = 128
    # 线性投影 tokens 到 embed_dim
    tokens_embedded = KDense(embed_dim, name="token_projection")(tokens)  # [batch, num_tokens, embed_dim]
    
    batch_size = tf.shape(tokens_embedded)[0]
    num_tokens = tokens_embedded.get_shape().as_list()[1]  # 固定为 26*9 = 234
    
    # 添加 [CLS] token
    cls_token = tf.compat.v1.get_variable("cls_token", shape=[1, 1, embed_dim],
                                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
    cls_tokens = tf.tile(cls_token, [batch_size, 1, 1])
    tokens_all = tf.concat([cls_tokens, tokens_embedded], axis=1)  # [batch, num_tokens+1, embed_dim]
    
    # 添加位置编码（固定形状）
    pos_embed = tf.compat.v1.get_variable("pos_embedding", shape=[1, num_tokens+1, embed_dim],
                                          initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
    tokens_all = tokens_all + pos_embed
    
    # Transformer 部分：3 层
    num_transformer_layers = 3
    num_heads = 4
    mlp_dim = 256
    dropout_rate = 0.1
    x_trans = tokens_all
    for i in range(num_transformer_layers):
        x_trans = transformer_block(x_trans, num_heads=num_heads, mlp_dim=mlp_dim,
                                    dropout_rate=dropout_rate, name=f"transformer_{i}")
    
    # 取 [CLS] token 输出
    cls_output = x_trans[:, 0, :]  # [batch, embed_dim]
    # 最后全连接层映射到 9 维输出
    output = dense_layer(cls_output, units=9, activation=None, name="fc_out")
    return output

###############################################################################
# 重构 streaking trace（用于无监督微调阶段）
###############################################################################
def reconstruct_trace(pred_params):
    # 分离 xuv_coefs 和 ir_params
    xuv_coefs_pred = pred_params[:, :5]
    ir_params_pred = pred_params[:, 5:]
    # 对 xuv_coefs_pred 前加上 0（无线性相位），如果转换函数需要这样
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
def train_network(train_file, test_file, epochs=70, batch_size=10, lr=1e-4, model_dir="models"):
    x_ph = tf.compat.v1.placeholder(tf.float32, [None, 29498], name="x_input")
    y_ph = tf.compat.v1.placeholder(tf.float32, [None, 9], name="y_target")
    is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
    
    # 使用混合模型网络
    pred_params = build_hybrid_network(x_ph, is_training)
    
    sup_loss = tf.reduce_mean(tf.square(pred_params - y_ph))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(sup_loss)
    
    # 无监督损失：重构的 streaking trace 与输入 trace 之间的 MSE
    reconstructed = reconstruct_trace(pred_params)
    unsup_loss = tf.reduce_mean(tf.square(reconstructed - x_ph))
    unsup_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    unsup_train_op = unsup_optimizer.minimize(unsup_loss)
    
    train_data = GetData(train_file, batch_size=batch_size)
    test_data = GetTestData(test_file)
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        num_train_samples = train_data.samples
        steps_per_epoch = num_train_samples // batch_size
        
        print("开始监督学习训练：70个 epoch，每个 epoch 有 {} 个步骤".format(steps_per_epoch))
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
            # 测试阶段
            x_test, y_test = test_data.get_all()
            test_loss = sess.run(sup_loss, feed_dict={x_ph: x_test, y_ph: y_test, is_training: False})
            print("=== Epoch {} 测试集 MSE = {:.6f} ===".format(epoch+1, test_loss))
            saver.save(sess, os.path.join(model_dir, "hybrid_model.ckpt"), global_step=epoch+1)
        
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
        model_name = "hybrid_experiment"
    model_directory = os.path.join("models", model_name)
    
    train_hdf5_path = r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\train3.hdf5"
    test_hdf5_path  = r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\test3.hdf5"
    
    train_network(train_file=train_hdf5_path, test_file=test_hdf5_path,
                  epochs=25, batch_size=10, lr=1e-4, model_dir=model_directory)
