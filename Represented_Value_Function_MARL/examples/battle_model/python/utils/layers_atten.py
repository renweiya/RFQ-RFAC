import numpy as np
import tensorflow as tf

#please site"Veličković P, Cucurull G, Casanova A, et al. Graph attention networks[J]. arXiv preprint arXiv:1710.10903, 2017"
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,last_layer=False,bias_flag=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.dense(seq, out_sz, activation=None,use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.dense(seq_fts, 1, activation=None)
        f_2 = tf.layers.dense(seq_fts, 1, activation=None)

        if len(seq.shape) == 2:
            logits = f_1 + tf.transpose(f_2)
        else:#batch situation
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])

        if bias_flag==True:
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
        else:
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
            
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.layers.dense(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq
        if last_layer==True:
            return coefs
        elif last_layer==2:
            return coefs,activation(ret)
        else:
            return activation(ret)  # activation

