import tensorflow as tf
import numpy as np

from magent.gridworld import GridWorld

from utils import layers_atten

class ValueNet:
    def __init__(self, sess, env, handle, name, update_every=5, use_mf=False, use_gat=False,bias_flag=False,learning_rate=1e-4, tau=0.005, gamma=0.95):
        # assert isinstance(env, GridWorld)
        self.env = env
        self.name = name
        self._saver = None
        self.sess = sess

        self.handle = handle
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        
        self.update_every = update_every
        self.use_mf = use_mf  # trigger of using mean field
        self.temperature = 0.1

        self.lr= learning_rate
        self.tau = tau
        self.gamma = gamma

        self.GAT=use_gat         #add by denote
        self.bias_flag=bias_flag
        with tf.variable_scope(name or "ValueNet"):
            self.name_scope = tf.get_variable_scope().name
            self.obs_input = tf.placeholder(tf.float32, (None,) + self.view_space, name="Obs-Input")
            self.feat_input = tf.placeholder(tf.float32, (None,) + self.feature_space, name="Feat-Input")
            self.mask = tf.placeholder(tf.float32, shape=(None,), name='Terminate-Mask')
            self.W_input =tf.placeholder(tf.float32, shape=(None,None), name="W-Input")#add by denote

            if self.use_mf:
                self.act_prob_input = tf.placeholder(tf.float32, shape=(None, self.num_actions), name="Act-Prob-Input")

            # TODO: for calculating the Q-value, consider softmax usage
            self.act_input = tf.placeholder(tf.int32, (None,), name="Act")
            self.act_one_hot = tf.one_hot(self.act_input, depth=self.num_actions, on_value=1.0, off_value=0.0)

            with tf.variable_scope("Eval-Net"):
                self.eval_name = tf.get_variable_scope().name
                self.e_q = self._construct_net(active_func=tf.nn.relu)
                self.predict = tf.nn.softmax(self.e_q / self.temperature)
                self.e_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.eval_name)

            with tf.variable_scope("Target-Net"):
                self.target_name = tf.get_variable_scope().name
                self.t_q = self._construct_net(active_func=tf.nn.relu)
                self.t_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_name)

            with tf.variable_scope("Update"):
                self.update_op = [tf.assign(self.t_variables[i],
                                            self.tau * self.e_variables[i] + (1. - self.tau) * self.t_variables[i])
                                    for i in range(len(self.t_variables))]

            with tf.variable_scope("Optimization"):
                self.target_q_input = tf.placeholder(tf.float32, (None,), name="Q-Input")
                self.e_q_max = tf.reduce_sum(tf.multiply(self.act_one_hot, self.e_q), axis=1)
                self.loss = tf.reduce_sum(tf.square(self.target_q_input - self.e_q_max) * self.mask) / tf.reduce_sum(self.mask)
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _construct_net(self, active_func=None, reuse=False):
        conv1 = tf.layers.conv2d(self.obs_input, filters=32, kernel_size=3,
                                 activation=active_func, name="Conv1")
        conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, activation=active_func,
                                 name="Conv2")
        flatten_obs = tf.reshape(conv2, [-1, np.prod([v.value for v in conv2.shape[1:]])])

        h_obs = tf.layers.dense(flatten_obs, units=256, activation=active_func,
                                name="Dense-Obs")
        h_emb = tf.layers.dense(self.feat_input, units=32, activation=active_func,
                                name="Dense-Emb", reuse=reuse)

        concat_layer = tf.concat([h_obs, h_emb], axis=1)
        #########  #GAT
        if self.GAT:
            hid_units = [128] # numbers of hidden units per each attention head in each layer
            n_heads = [4, 1] # additional entry for the output layer
            residual = False
            activation=tf.nn.elu
            attn_drop=0.0             
            ffd_drop=0.0              
            #require self.W_input be a 0-1 matrix
            bias_flag=self.bias_flag
            print("rfq:************",bias_flag)
            bias_mat = -10e9 * (1.0 - self.W_input)#no concetions in W,no coefs 
            attns = []
            #first
            concat_layer_out = tf.concat([concat_layer, self.act_prob_input], axis=1)
            for _ in range(n_heads[0]):
                attns.append(layers_atten.attn_head(concat_layer_out, bias_mat=bias_mat,
                    out_sz=hid_units[0], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=False,last_layer=False,bias_flag=bias_flag))
            h_1 = tf.concat(attns, axis=-1)
            #middle
            for i in range(1, len(hid_units)):
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
            
            graph_attention_out_x = tf.matmul(graph_attention_coefs,self.act_prob_input)
            
        #
        prob_emb = tf.layers.dense(graph_attention_out_x, units=64, activation=active_func, name='Prob-Emb')
        h_act_prob = tf.layers.dense(prob_emb, units=32, activation=active_func, name="Dense-Act-Prob")
        dense_prob_norm=tf.nn.l2_normalize(h_act_prob, dim=1, epsilon=1e-12,name=None)
        graph_attention_fea_norm=tf.nn.l2_normalize(graph_attention_fea, dim=1, epsilon=1e-12,name=None)
        graph_attention_out = tf.concat([graph_attention_fea_norm, dense_prob_norm], axis=1)


        concat_layer_norm=tf.nn.l2_normalize(concat_layer, dim=1, epsilon=1e-12,name=None)
        concat_layer_first = tf.concat([concat_layer_norm, self.act_prob_input], axis=1)    
        concat_layer_final = tf.concat([concat_layer_first, graph_attention_out], axis=1)


        dense2 = tf.layers.dense(concat_layer_final, units=128, activation=active_func, name="Dense2")
        out = tf.layers.dense(dense2, units=64, activation=active_func, name="Dense-Out")

        q = tf.layers.dense(out, units=self.num_actions, name="Q-Value")

        return q

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
    


    def calc_target_q(self, **kwargs):
        """Calculate the target Q-value
        kwargs: {'obs', 'feature', 'W', prob', 'dones', 'rewards'}
        """
        feed_dict = {
            self.obs_input: kwargs['obs'],
            self.feat_input: kwargs['feature'],
            self.W_input: kwargs['W']
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        t_q, e_q = self.sess.run([self.t_q, self.e_q], feed_dict=feed_dict)
        act_idx = np.argmax(e_q, axis=1)
        q_values = t_q[np.arange(len(t_q)), act_idx]
        target_q_value = kwargs['rewards'] + (1. - kwargs['dones']) * q_values.reshape(-1) * self.gamma
        
        return target_q_value

    def update(self):
        """Q-learning update"""
        self.sess.run(self.update_op)

    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1],
            self.W_input: kwargs['W']#add by denote
        }

        self.temperature = kwargs['eps']

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input] = kwargs['prob']

        actions = self.sess.run(self.predict, feed_dict=feed_dict)
        
        #pi-by probability,not by max
        acts=[]
        for i in range(len(actions)):
            acts.append(np.random.choice(len(actions[i]), 1, replace=False, p=actions[i]))
        actions_=np.concatenate(acts).astype(np.int32)
        
        #max
        # actions_ = np.argmax(actions, axis=1).astype(np.int32)
        
        return actions_

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [obs, feature], 'target_q', 'prob', 'acts'}
        """
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1],
            self.target_q_input: kwargs['target_q'],
            self.W_input: kwargs['W'],#add by denote
            self.mask: kwargs['masks']
        }
        
        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        feed_dict[self.act_input] = kwargs['acts']
        _, loss, e_q = self.sess.run([self.train_op, self.loss, self.e_q_max], feed_dict=feed_dict)
        return loss, {'Eval-Q': np.round(np.mean(e_q), 6), 'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}
       