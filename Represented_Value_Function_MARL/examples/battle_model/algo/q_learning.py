import os
import tensorflow as tf
import numpy as np

from . import base_original 
from . import base

from . import tools
from . import tools_new

from examples.battle_model.adjacency_matrix import adjacency_matrix

class DQN(base_original.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, memory_size=2**10, batch_size=128, update_every=5):

        super().__init__(sess, env, handle, name, update_every=update_every)

        self.replay_buffer = tools.MemoryGroup(self.view_space, self.feature_space, self.num_actions, memory_size, batch_size, sub_len)
        self.sess.run(tf.global_variables_initializer())

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, feats, obs_next, feat_next, dones, rewards, actions, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones)
            loss, q = super().train(state=[obs, feats], target_q=target_q, acts=actions, masks=masks)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))

class MFQ(base_original.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**10, batch_size=128):#batch_size=64 by denote
        super().__init__(sess, env, handle, name, use_mf=True,update_every=update_every)
        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            # 'prob_shape':(self.num_actions,),
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools.MemoryGroup(**config)
        self.update_every = update_every
        self.batch_size=batch_size
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()#*self.batch_size

        ## #add by denote
        for i in range(batch_name):
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()

            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob=act_prob_next)
            
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob=act_prob, acts=acts, masks=masks)
            
            self.update()

            if i % 500 == 1:
                print('[*] LOSS:', loss, '/ Q:', q,'batch_name:',batch_name,'i:',i)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))


class RFQ(base.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**10, neighbor_nums=-1,bias_flag=False, batch_size=64):#this batch_size only decides the batch_name,the batch size is the number of agents per step; batch_size=64 by denote
        super().__init__(sess, env, handle, name, use_mf=True, use_gat=True,bias_flag=bias_flag, update_every=update_every)
        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'prob_shape':(self.num_actions,),
            # 'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools_new.MemoryGroup(**config)
        self.update_every = update_every
        self.batch_size=batch_size#add by denote
        self.neighbor_nums=neighbor_nums

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()#*self.batch_size#traing only one sample per training step

        ## #add by denote
        for i in range(batch_name):

            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()

            if self.neighbor_nums==-1:
                local_or_not=False
                W=np.ones((len(acts),len(acts)))#add by denote
            else:
                local_or_not=True
                neighbor_num=min(self.neighbor_nums,len(acts))
                W=adjacency_matrix(feat,neighbor_num)#need time

                
            target_q = self.calc_target_q(obs=obs_next, feature=feat_next, rewards=rewards, dones=dones, prob=act_prob_next,W=W)
            
            loss, q = super().train(state=[obs, feat], target_q=target_q, prob=act_prob, acts=acts, masks=masks,W=W)
            
            self.update()

            if i % 500 == 1:
                print('[*] LOSS:', loss, '/ Q:', q,'batch_name:',batch_name,'i:',i,"local_graph:",local_or_not)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "rfq_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "rfq_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))
