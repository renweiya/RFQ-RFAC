import os
import numpy as np
import tensorflow as tf

from . import tools
from . import tools_new
from utils import layers_atten

import time
from examples.battle_model.adjacency_matrix import adjacency_matrix


class RFAC:
    def __init__(self, sess, name, handle, env ,neighbor_nums=-1,bias_flag=False,value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4):#learning_rate=1e-4
        self.sess = sess
        self.env = env
        self.name = name

        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        self.reward_decay = gamma

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss

        # init training buffers
        self.view_buf = np.empty((1,) + self.view_space)
        self.feature_buf = np.empty((1,) + self.feature_space)
        self.action_buf = np.empty(1, dtype=np.int32)
        self.reward_buf = np.empty(1, dtype=np.float32)
        self.reward_next_buf = np.empty(1, dtype=np.float32)
        self.reward_original_buf = np.empty(1, dtype=np.float32)
        
        self.step_buf = np.empty(1, dtype=np.int32)
        self.the_id_buf = np.empty(1, dtype=np.int32)

        self.replay_buffer = tools_new.EpisodesBuffer(use_mean=True)
        self.neighbor_nums=neighbor_nums
        self.bias_flag=bias_flag

        with tf.variable_scope(name):
            self.name_scope = tf.get_variable_scope().name
            self.loss_=self._create_network(self.view_space, self.feature_space, )
        # with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # with tf.name_scope('gradient'):
            gradient_all = optimizer.compute_gradients(self.loss_)  # gradient of network (with NoneType)
            grads_vars = [v for (g, v) in gradient_all if g is not None]  # all variable that has gradients
            self.gradient = optimizer.compute_gradients(self.loss_, grads_vars)  # gradient of network (without NoneType)
            self.grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v)
                     for (g, v) in self.gradient]
            self.train_op = optimizer.apply_gradients(self.grads_holder)


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def act(self, **kwargs):
        action = self.sess.run(self.calc_action, {
            self.input_view: kwargs['state'][0],
            self.input_feature: kwargs['state'][1],
            self.input_act_prob_all:kwargs['prob'],
            self.input_W: kwargs['W']
        })
        return action.astype(np.int32).reshape((-1,))

    def _create_network(self, view_space, feature_space):
        # input
        self.input_view = tf.placeholder(tf.float32, (None,) + view_space)
        self.input_feature = tf.placeholder(tf.float32, (None,) + feature_space)
        self.input_W = tf.placeholder(tf.float32, (None,None))
        self.input_act_prob = tf.placeholder(tf.float32, (None, self.num_actions))
        self.input_act_prob_all = tf.placeholder(tf.float32, (None, self.num_actions))
        self.reward = tf.placeholder(tf.float32, [None])
        self.reward_next = tf.placeholder(tf.float32, [None])
        self.reward_original = tf.placeholder(tf.float32, [None])
        self.action = tf.placeholder(tf.int32, [None])
        self.N = tf.placeholder(tf.int32)
        self.N_neighbor = tf.placeholder(tf.float32)

        input_view=self.input_view
        input_feature=self.input_feature
        input_W=self.input_W
        input_act_prob=self.input_act_prob
        input_act_prob_all=self.input_act_prob_all
        action=self.action
        reward=self.reward

        ###
        hidden_size = [256]

        #base method
        conv1 = tf.layers.conv2d(input_view, filters=32, kernel_size=3,activation=tf.nn.relu, name="Conv1")
        conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, activation=tf.nn.relu,name="Conv2")
        flatten_obs = tf.reshape(conv2, [-1, np.prod([v.value for v in conv2.shape[1:]])])
        h_view = tf.layers.dense(flatten_obs, units=256, activation=tf.nn.relu,name="Dense-Obs")

        # similar performance. need tunning.
        # flatten_view = tf.reshape(input_view, [-1, np.prod([v.value for v in input_view.shape[1:]])])
        # h_view = tf.layers.dense(flatten_view, units=hidden_size[0], activation=tf.nn.relu)

        #34-D vector,which contains the id,last action and the position.
        # h_emb = tf.layers.dense(input_feature,  units=hidden_size[0], activation=tf.nn.relu)
        h_emb = tf.layers.dense(input_feature,  units=32, activation=tf.nn.relu)

        concat_layer = tf.concat([h_view, h_emb], axis=1)
        
        ## GAT
        GAT=True
        if GAT==True:
            print("-----------------------first------------------------")
            hid_units = [128] # numbers of hidden units per each attention head in each layer
            n_heads = [4, 1] # additional entry for the output layer
            residual = False
            activation=tf.nn.elu
            attn_drop=0.0    #0.5
            ffd_drop=0.0     #0.5
            #require self.W_input be a 0-1 matrix
            bias_flag=self.bias_flag
            print("rfac:************",bias_flag)
            bias_mat = -10e9 * (1.0 - input_W)#no concetions in W,no coefs 
            attns = []
            #consider the last action as the input

            #
            concat_layer_out1 = tf.concat([concat_layer, input_act_prob_all], axis=1)
            
            for _ in range(n_heads[0]):
                attns.append(layers_atten.attn_head(concat_layer_out1, bias_mat=bias_mat,
                    out_sz=hid_units[0], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False,last_layer=False,bias_flag=bias_flag))
            h_1 = tf.concat(attns, axis=-1)
            #middle
            for i in range(1, len(hid_units)):
                print("-----------------------middle------------------------")
                # h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(layers_atten.attn_head(h_1, bias_mat=bias_mat,
                        out_sz=hid_units[i], activation=activation,
                        in_drop=ffd_drop, coef_drop=attn_drop, residual=residual,last_layer=False,bias_flag=bias_flag))
                h_1 = tf.concat(attns, axis=-1)
            #out

            # out = []
            # for i in range(n_heads[-1]):
            #     out.append(layers_atten.attn_head(h_1, bias_mat=bias_mat,
            #         out_sz=self.num_actions, activation=lambda x: x,
            #         in_drop=ffd_drop, coef_drop=attn_drop, residual=False,last_layer=True,bias_flag=False))
            # graph_attention_coefs = tf.add_n(out) / n_heads[-1]
            
            [graph_attention_coefs,graph_attention_fea]=layers_atten.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=128, activation=lambda x: x,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False,last_layer=2,bias_flag=bias_flag)
            
            
            graph_attention_out_t = tf.matmul(graph_attention_coefs,input_act_prob_all)
            self.graph_attention_out_t=graph_attention_out_t

            
        else:
            graph_attention_out_t=input_act_prob
        
        # policy

        emb_prob = tf.layers.dense(graph_attention_out_t, units=64, activation=tf.nn.relu)
        dense_prob = tf.layers.dense(emb_prob, units=32, activation=tf.nn.relu)
        dense_prob_norm=tf.nn.l2_normalize(dense_prob, dim=1, epsilon=1e-12,name=None)
        graph_attention_fea_norm=tf.nn.l2_normalize(graph_attention_fea, dim=1, epsilon=1e-12,name=None)
        graph_attention_out_norm = tf.concat([graph_attention_fea_norm,dense_prob_norm], axis=1)
        
        concat_layer_norm=tf.nn.l2_normalize(concat_layer, dim=1, epsilon=1e-12,name=None)
        concat_layer_out2_norm = tf.concat([concat_layer_norm, input_act_prob_all], axis=1)    
        
        concat_layer3 = tf.concat([concat_layer_out2_norm, graph_attention_out_norm], axis=1)
        #### policy
        dense = tf.layers.dense(concat_layer3, units=hidden_size[0] * 2, activation=tf.nn.relu)
        policy = tf.layers.dense(dense / 0.1, units=self.num_actions, activation=tf.nn.softmax)
        policy = tf.clip_by_value(policy, 1e-10, 1-1e-10)

        self.calc_action = tf.multinomial(tf.log(policy), 1)

        ##### value
        dense = tf.layers.dense(concat_layer3, units=hidden_size[0], activation=tf.nn.relu)
        value = tf.layers.dense(dense, units=1)
        value = tf.reshape(value, (-1,))
        
        #
        action_mask = tf.one_hot(action, self.num_actions)
        advantage = tf.stop_gradient(reward - value)
        log_policy = tf.log(policy + 1e-6)
        log_prob = tf.reduce_sum(log_policy * action_mask, axis=1)

        
        
        pg_loss = -tf.reduce_mean(advantage * log_prob)#reduce_mean
        vf_loss = self.value_coef * tf.reduce_mean(tf.square(reward - value))#reduce_mean

        ###
        
        neg_entropy = self.ent_coef * tf.reduce_mean(tf.reduce_sum(policy * log_policy, axis=1))#reduce_mean
        
        total_loss = pg_loss + vf_loss+ neg_entropy# + weight_loss+weight_loss2
        
        self.policy, self.value = policy, value

        self.pg_loss, self.vf_loss, self.reg_loss = pg_loss, vf_loss, neg_entropy
        self.total_loss=total_loss
        return total_loss


    def train(self):
        # calc buffer size, n reset
        n = 0
        batch_data = self.replay_buffer.episodes()

        self.replay_buffer = tools_new.EpisodesBuffer(use_mean=True)

        for episode in batch_data:
            n += len(episode.rewards)

        self.view_buf.resize((n,) + self.view_space)
        self.feature_buf.resize((n,) + self.feature_space)
        self.action_buf.resize(n)
        self.reward_buf.resize(n)
        self.reward_original_buf.resize(n)
        self.reward_next_buf.resize(n)
        self.step_buf.resize(n)
        self.the_id_buf.resize(n)
        
        view, feature = self.view_buf, self.feature_buf
        action, reward, step,the_id = self.action_buf, self.reward_buf, self.step_buf,self.the_id_buf
        reward_next,reward_original=self.reward_next_buf, self.reward_original_buf

        act_prob_buff = np.zeros((n, self.num_actions), dtype=np.float32)#pre-computed mean action move for agent
        act_prob_all_buff = np.zeros((n, self.num_actions), dtype=np.float32)#action move for each agent
        act_prob_atten_buff = np.zeros((n, self.num_actions), dtype=np.float32)
        ## ct reset
        
        
        # collect episodes from multiple separate buffers to a continuous buffer
        #1 first restore without compute the r
        ct = 0
        for k, episode in enumerate(batch_data):
            v, f, a, prob,prob_all,step_ct,id_ct = episode.views, episode.features, episode.actions, episode.probs,episode.probs_all, episode.step_ct, episode.id_ct
            m = len(episode.id_ct)
            assert len(prob) > 0 
            view[ct:ct + m] = v
            feature[ct:ct + m] = f
            action[ct:ct + m] = a
            step[ct:ct + m] = step_ct
            the_id[ct:ct + m] = id_ct
            act_prob_buff[ct:ct + m] = prob
            act_prob_all_buff[ct:ct + m] = prob_all
            ct += m
        # check
        
        assert n == ct
        #2 compute the act_prob_atten_buff #the weight sum of actions
        max_step_ct=max(step)
        if self.neighbor_nums==-1:
            local_or_not=False
        else:
            local_or_not=True
        for step_ in range(max_step_ct+1):
            # print("step_",step_)
            index_tmp=np.where(step==step_)[0]
            view_tmp,feature_tmp,act_prob_all_buff_tmp,act_prob_buff_tmp,action_tmp,W_tmp=[],[],[],[],[],[]
            view_tmp=view[index_tmp]
            feature_tmp=feature[index_tmp]
            action_tmp=action[index_tmp]
            act_prob_buff_tmp=act_prob_buff[index_tmp]
            act_prob_all_buff_tmp=act_prob_all_buff[index_tmp]
            # compute the act_prob_atten_buff

            if local_or_not==True:
                neighbor_num=min(self.neighbor_nums,len(action_tmp))
                W_tmp=adjacency_matrix(feature_tmp,neighbor_num)
            else:
                W_tmp=np.ones((len(action_tmp),len(action_tmp)))

            ### 
            act_prob_atten_buff[index_tmp]= self.sess.run(
            [self.graph_attention_out_t], feed_dict={
                self.input_view: view_tmp,
                self.input_feature: feature_tmp,
                self.input_act_prob_all: act_prob_all_buff_tmp,
                self.input_act_prob: act_prob_buff_tmp,
                self.input_W: W_tmp,
            })

        #3 compute the reward
        ct = 0
        gamma = self.reward_decay
        for k, episode in enumerate(batch_data):#for each id episode
            v, f, a, r, prob, prob_all,step_ct,id_ct = episode.views, episode.features, episode.actions, episode.rewards, episode.probs,episode.probs_all, episode.step_ct, episode.id_ct
            # v, f, r, id_ct = episode.views, episode.features, episode.rewards, episode.id_ct

            m = len(episode.rewards)
            assert len(prob) > 0 
            r = np.array(r)
            r_next=np.zeros(m, dtype=np.float32)
            ### reward_original
            reward_original[ct:ct + m]=r
            #
            id_index_tmp=np.where(the_id==id_ct[0])[0]
            prob_tmp=act_prob_atten_buff[id_index_tmp[-1]]#last one for each id episode. id episode:from step 0 to the end for one id.

            W_tmp=np.ones((1,1))
            keep = self.sess.run(self.value, feed_dict={
                self.input_view: [v[-1]],
                self.input_feature: [f[-1]],
                self.input_act_prob_all: [prob_tmp],
                self.input_act_prob: [prob_tmp],
                self.input_W: W_tmp
            })[0]
            
            for i in reversed(range(m)): #from the last step to the first step
                keep = keep * gamma + r[i]
                r[i] = keep
                if i>0:
                    r_next[i-1]=r[i]
            ###
            r_next[m-1]=r[m-1]    
            reward[ct:ct + m] = r
            reward_next[ct:ct + m] = r_next
            ct += m
        assert n == ct
        
        ## train by steps
        max_step_ct=max(step)
        # step_permutation=np.random.permutation(max_step_ct)
        count_i=0
        grads = []
        for step_ in range(max_step_ct+1):#step_permutation:#
            index_tmp=np.where(step==step_)[0]
            # shuffle the mini batch
            index_tmp=np.random.choice(index_tmp,len(index_tmp), replace=False, p=None)
            view_tmp,feature_tmp,act_prob_buff_tmp,action_tmp,reward_tmp,W_tmp=[],[],[],[],[],[]
            view_tmp=view[index_tmp]
            feature_tmp=feature[index_tmp]
            act_prob_buff_tmp=act_prob_buff[index_tmp]
            act_prob_all_buff_tmp=act_prob_all_buff[index_tmp]
            action_tmp=action[index_tmp]
            reward_tmp=reward[index_tmp]
            reward_next_tmp=reward_next[index_tmp]
            reward_original_tmp=reward_original[index_tmp]

            if local_or_not==True:
                neighbor_num=min(self.neighbor_nums,len(action_tmp))
                W_tmp=adjacency_matrix(feature_tmp,neighbor_num)
            else:
                W_tmp=np.ones((len(action_tmp),len(action_tmp)))
                
            grad_i,total_loss,pg_loss, vf_loss, ent_loss, state_value = self.sess.run(
                [self.gradient,self.total_loss,self.pg_loss, self.vf_loss, self.reg_loss, self.value], feed_dict={
                self.input_view: view_tmp,
                self.input_feature: feature_tmp,
                self.input_act_prob: act_prob_buff_tmp,
                self.input_act_prob_all: act_prob_all_buff_tmp,
                self.action: action_tmp,
                self.reward: reward_tmp,
                self.reward_next: reward_next_tmp,
                self.reward_original: reward_original_tmp,
                self.input_W: W_tmp,
                self.N:len(action_tmp),
                self.N_neighbor:len(action_tmp)
                })

            # weights for grad sum
            weight=float(len(index_tmp)/n)
            grad_i[0] = [grad*weight for grad in grad_i[0]]
            #
            grads.append(grad_i)

            count_i=count_i+1
            if count_i%100==0 or count_i==max_step_ct:
                print("self.total_loss",total_loss)
                print('[*] PG_LOSS:', np.round(pg_loss, 6), '/ VF_LOSS:', np.round(vf_loss, 6), '/ ENT_LOSS:', np.round(ent_loss, 6), '/ VALUE:', np.mean(state_value),"local_graph",local_or_not)
        
        
        #######
        grads_sum = {} #dict
        # add up
        print("self.grads_holder",len(self.grads_holder))
        for i in range(len(self.grads_holder)):
            k = self.grads_holder[i][0] #the tensor key
            #sum
            grads_sum[k] = sum([g[i][0] for g in grads])
            #clip
            grads_sum[k] = [np.sign(grad)*np.minimum(5,abs(grad)) for grad in grads_sum[k]]
        #update θ by average gradient
        _ = self.sess.run(self.train_op, feed_dict=grads_sum)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "rfac_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "rfac_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))
