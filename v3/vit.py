#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vit.py

使用 Vision Transformer 替代 CNN 进行 attosecond streaking phase retrieval 的监督训练。
数据集格式与之前保持一致：
  - train3.hdf5 和 test3.hdf5 均包含：
      noise_trace: shape (num_samples, 29498)
      xuv_coefs:   shape (num_samples, 5)
      ir_params:   shape (num_samples, 4)
模型输出为 9 维向量（前5维为 xuv_coefs, 后4维为 ir_params），
损失函数采用均方误差（MSE），训练采用 Adam 优化器。
"""

import os
import sys
import numpy as np
import tensorflow as tf
import tables
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, LayerNormalization, Dropout

tf.compat.v1.disable_eager_execution()

# 设置 GPU 显存按需分配（可避免一次性分配所有显存）
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)

#############################################
# 数据读取类：与之前版本保持一致
#############################################
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

#############################################
# Vision Transformer 模型部分
#############################################
def extract_patches(images, patch_size):
    """
    images: [batch, height, width, channels]
    使用 tf.image.extract_patches 提取大小为 patch_size x patch_size 的 patch，
    返回 shape: [batch, num_patches, patch_dim]
    """
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patch_dims = patch_size * patch_size * images.get_shape()[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches

def transformer_block(x, num_heads, mlp_dim, dropout_rate, name):
    """
    Transformer block：
      LayerNorm -> 多头自注意力 -> 残差连接
      LayerNorm -> MLP（Dense->GELU->Dropout->Dense->Dropout） -> 残差连接
    """
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        ln1 = LayerNormalization(epsilon=1e-6, name="ln1")(x)
        dim = x.get_shape().as_list()[-1]
        q = Dense(dim, name="q")(ln1)
        k = Dense(dim, name="k")(ln1)
        v = Dense(dim, name="v")(ln1)
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
        mlp_output = Dense(mlp_dim, activation=tf.nn.gelu, name="mlp_dense1")(ln2)
        mlp_output = Dropout(dropout_rate)(mlp_output)
        mlp_output = Dense(dim, name="mlp_dense2")(mlp_output)
        mlp_output = Dropout(dropout_rate)(mlp_output)
        x = x + mlp_output
        return x

def build_vit(x, is_training, image_size=(301,98), patch_size=(7,7),
              num_layers=8, num_heads=4, mlp_dim=256, dropout_rate=0.1, embed_dim=128):
    """
    构建 Vision Transformer 模型：
      1. 将输入 x ([batch, 29498]) reshape 为图像 [batch, 301, 98, 1]
      2. 将图像划分为 7x7 的 patches
      3. 对每个 patch 进行线性投影得到 embed_dim 维向量
      4. 添加 [CLS] token 和位置编码（位置编码根据常数 num_patches+1 设定）
      5. 经过 num_layers 个 Transformer block
      6. 取 [CLS] token 输出经过全连接层映射到 9 维输出
    """
    batch_size = tf.shape(x)[0]
    img = tf.reshape(x, [-1, image_size[0], image_size[1], 1])
    patches = extract_patches(img, patch_size[0])
    # 计算 num_patches 为常数：(image_height//patch_height) * (image_width//patch_width)
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    patch_dim = patch_size[0] * patch_size[1] * 1

    patches_embedded = Dense(embed_dim, name="patch_projection")(patches)  # [batch, num_patches, embed_dim]
    # 添加 [CLS] token
    cls_token = tf.compat.v1.get_variable("cls_token", shape=[1, 1, embed_dim],
                                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
    cls_tokens = tf.tile(cls_token, [batch_size, 1, 1])
    tokens = tf.concat([cls_tokens, patches_embedded], axis=1)  # [batch, num_patches+1, embed_dim]
    # 位置编码，注意 num_patches+1 为常数
    pos_embed = tf.compat.v1.get_variable("pos_embedding", shape=[1, num_patches+1, embed_dim],
                                          initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
    tokens = tokens + pos_embed

    x_trans = tokens
    for i in range(num_layers):
        x_trans = transformer_block(x_trans, num_heads=num_heads, mlp_dim=mlp_dim,
                                    dropout_rate=dropout_rate, name=f"transformer_{i}")

    cls_output = x_trans[:, 0, :]  # [batch, embed_dim]
    output = Dense(9, name="head")(cls_output)
    return output

#############################################
# 训练流程
#############################################
def train_vit(train_file, test_file, epochs=40, batch_size=10, lr=1e-4, model_dir="vit_models"):
    """
    训练 Vision Transformer 模型：
      - 输入： noise_trace, shape [batch, 29498]
      - 标签： 拼接后的 xuv_coefs 和 ir_params, 9 维
      - 损失函数：均方误差（MSE）
    """
    x_ph = tf.compat.v1.placeholder(tf.float32, [None, 29498], name="x_input")
    y_ph = tf.compat.v1.placeholder(tf.float32, [None, 9], name="y_target")
    is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")

    pred = build_vit(x_ph, is_training)
    loss = tf.reduce_mean(tf.square(pred - y_ph))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)
    mse_eval = tf.reduce_mean(tf.square(pred - y_ph))

    train_data = GetData(train_file, batch_size=batch_size)
    test_data = GetTestData(test_file)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        num_train_samples = train_data.samples
        steps_per_epoch = num_train_samples // batch_size

        for epoch in range(epochs):
            train_loss = 0.0
            train_data.reset()
            for step in range(steps_per_epoch):
                x_batch, y_batch = train_data.next_batch()
                feed_dict = {x_ph: x_batch, y_ph: y_batch, is_training: True}
                _, l_val = sess.run([train_op, loss], feed_dict=feed_dict)
                train_loss += l_val
                if (step + 1) % 200 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{steps_per_epoch}], Batch Loss = {l_val:.6f}")
            avg_train_loss = train_loss / steps_per_epoch
            print(f">>> Epoch [{epoch+1}/{epochs}] Train MSE = {avg_train_loss:.6f}")

            # 测试阶段：将测试数据分批评估，避免一次性加载整个测试集
            x_test_all, y_test_all = test_data.get_all()
            num_test = x_test_all.shape[0]
            test_loss_total = 0.0
            num_batches = num_test // batch_size
            for i in range(num_batches):
                x_batch = x_test_all[i*batch_size: (i+1)*batch_size]
                y_batch = y_test_all[i*batch_size: (i+1)*batch_size]
                test_loss_total += sess.run(mse_eval, feed_dict={x_ph: x_batch, y_ph: y_batch, is_training: False})
            avg_test_loss = test_loss_total / num_batches
            print(f"=== Epoch [{epoch+1}/{epochs}] Test MSE = {avg_test_loss:.6f} ===")
            saver.save(sess, os.path.join(model_dir, "vit_model.ckpt"), global_step=epoch+1)
        print("Training completed. Model saved to:", model_dir)

#############################################
# 主函数入口
#############################################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "vit_experiment"
    model_directory = os.path.join("vit_models", model_name)
    train_hdf5_path = r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\train3.hdf5"
    test_hdf5_path  = r"C:\Users\ROG\Desktop\attosecond_streaking_phase_retrieval-ir_cep_fix_3\test3.hdf5"

    train_vit(train_file=train_hdf5_path, test_file=test_hdf5_path,
              epochs=25, batch_size=10, lr=1e-4, model_dir=model_directory)
