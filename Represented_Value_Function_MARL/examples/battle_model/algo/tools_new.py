import numpy as np
import tensorflow as tf
import os
from collections import deque
import random
from queue import Queue

class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'

class Buffer:
    def __init__(self):
        pass

    def push(self, **kwargs):
        raise NotImplementedError

class MetaBuffer(object):
    def __init__(self, shape, max_len, dtype='float32'):
        self.max_len = max_len
        self.data = np.zeros((max_len,) + shape).astype(dtype)
        self.start = 0
        self.length = 0
        self._flag = 0
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def sample(self, idx):
        return self.data[idx % self.length]

    def pulling(self):
        return self.data[:self.length]

    def appending(self, value):
        start = 0
        num = len(value)
        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0
        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def group_appending(self, value):
        start = 0
        num = len(value)
        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0
        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def reset_new(self, start, value):
        self.data[start:] = value

class EpisodesBufferEntry:
    """Entry for episode buffer"""
    def __init__(self):
        self.views = []
        self.features = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.probs_all = []
        self.step_ct = []
        self.id_ct = []
        self.terminal = False

    def append(self, view, feature, action, reward, alive,step_ct,id_ct,probs=None,probs_all=None):
        self.views.append(view.copy())
        self.features.append(feature.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.step_ct.append(step_ct)
        self.id_ct.append(id_ct)
        if probs is not None:
            self.probs.append(probs)
        if probs_all is not None:
            self.probs_all.append(probs_all)
        if not alive:
            self.terminal = True

class EpisodesBuffer_Step(Buffer):# by step
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def __init__(self, use_mean=False):
        super().__init__()
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        view, feature = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        alives = kwargs['alives']
        ids = kwargs['ids']
        step_ct = kwargs['step_ct']

        if self.use_mean:
            probs = kwargs['prob']
            probs_all = kwargs['prob_all']

        buffer = self.buffer
        entry = buffer.get(step_ct)
        if entry is None:
            entry = EpisodesBufferEntry()
            buffer[step_ct] = entry # by step
        
        index = np.random.permutation(len(view))
        for i in range(len(ids)):
            i = index[i]
            if self.use_mean:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i],step_ct,ids[i],probs=probs[i],probs_all=probs_all[i])
            else:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i],step_ct,ids[i])

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    def episodes(self):
        """ get episodes """
        return self.buffer.values()

class EpisodesBuffer(Buffer): # by id
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def __init__(self, use_mean=False):
        super().__init__()
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        view, feature = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        alives = kwargs['alives']
        ids = kwargs['ids']
        step_ct = kwargs['step_ct']

        if self.use_mean:
            probs = kwargs['prob']
            probs_all = kwargs['prob_all']

        buffer = self.buffer
        index = np.random.permutation(len(view))

        for i in range(len(ids)):
            i = index[i]
            entry = buffer.get(ids[i])
            if entry is None:
                entry = EpisodesBufferEntry()
                buffer[ids[i]] = entry # by id

            if self.use_mean:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i],step_ct,ids[i],probs=probs[i],probs_all=probs_all[i])
            else:
                entry.append(view[i], feature[i], acts[i], rewards[i], alives[i],step_ct,ids[i])

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    def episodes(self):
        """ get episodes """
        return self.buffer.values()

class AgentMemory(object):
    def __init__(self, obs_shape,feat_shape, prob_shape, sub_len, use_mean=False):

        self.obs0 = MetaBuffer(obs_shape, sub_len)
        self.feat0 = MetaBuffer(feat_shape, sub_len)
        self.actions = MetaBuffer((), sub_len, dtype='int32')
        self.rewards = MetaBuffer((), sub_len)

        self.alive = MetaBuffer((), sub_len, dtype='bool')
        self.id_set = MetaBuffer((), sub_len, dtype='int32')

        self.use_mean = use_mean

        if self.use_mean:
            self.prob = MetaBuffer(prob_shape, sub_len)

        self.step_ct=MetaBuffer((), sub_len,dtype='int32')
        

    def appended(self, obs0, feat0, act, reward, alive,step_ct,id_set,prob=None):
        self.obs0.appending(np.array([obs0]))
        self.feat0.appending(np.array([feat0]))
        self.actions.appending(np.array([act], dtype=np.int32))
        self.rewards.appending(np.array([reward]))
        # self.terminals.append(np.array([not alive], dtype=np.bool))
        self.alive.appending(np.array([alive], dtype=np.bool))
        self.id_set.appending(np.array([id_set], dtype=np.int32))

        if self.use_mean:
            self.prob.appending(np.array([prob]))

        self.step_ct.appending(np.array([step_ct]))

    def pulled(self):
        res = {
            'obs0': self.obs0.pulling(),
            'feat0': self.feat0.pulling(),
            'act': self.actions.pulling(),
            'rewards': self.rewards.pulling(),
            'alive': self.alive.pulling(),
            'prob': None if not self.use_mean else self.prob.pulling(),
            'step_ct': self.step_ct.pulling(),
            'id_set': self.id_set.pulling()
        }

        return res


class MemoryGroup(object):
    def __init__(self, obs_shape,feat_shape,prob_shape, max_len, batch_size, sub_len, use_mean=False):
        self.agent = dict()
        self.max_len = max_len
        self.batch_size = batch_size

        self.obs_shape = obs_shape
        self.feat_shape = feat_shape
        self.prob_shape = prob_shape

        self.sub_len = sub_len
        self.use_mean = use_mean


        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.alive = MetaBuffer((), max_len, dtype='bool')
        self.masks = MetaBuffer((), max_len, dtype='bool')
        if use_mean:
            self.prob = MetaBuffer(prob_shape, max_len)
        self.step_ct = MetaBuffer((), max_len, dtype='int32')
        self.id_set = MetaBuffer((), max_len, dtype='int32')
        self._new_add = 0
        #####add by denote
        self._new_add_save = 0
        self.all_id=dict()
        self.alive_steps=dict()#alive steps for agent
        self.tmp1=[]
        self.tmp2=[]
        self.tmp3=[]
        #####add end
    def _flush(self, **kwargs):
        self.obs0.group_appending(kwargs['obs0'])
        self.feat0.group_appending(kwargs['feat0'])
        self.actions.group_appending(kwargs['act'])
        self.rewards.group_appending(kwargs['rewards'])
        self.alive.group_appending(kwargs['alive'])
        self.id_set.group_appending(kwargs['id_set'])

        if self.use_mean:
            self.prob.group_appending(kwargs['prob'])

        self.step_ct.group_appending(kwargs['step_ct'])
        # mask = self.alive #np.where(kwargs['terminals'] == True, False, True)
        # mask[-1] = False
        # self.masks.group_appending(mask)

    def push(self, **kwargs):

        for i, _id in enumerate(kwargs['id_set']):

            if 1:#kwargs['step_ct']%11==0 or kwargs['step_ct']%12==0:
                if self.agent.get(kwargs['step_ct']) is None:
                    self.agent[kwargs['step_ct']] = AgentMemory(self.obs_shape, self.feat_shape,self.prob_shape, self.sub_len, use_mean=self.use_mean)

                if self.use_mean:
                    self.agent[kwargs['step_ct']].appended(obs0=kwargs['state'][0][i], feat0=kwargs['state'][1][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i],step_ct=kwargs['step_ct'], id_set=kwargs['id_set'][i],prob=kwargs['prob_all'][i])
                else:
                    self.agent[kwargs['step_ct']].appended(obs0=kwargs['state'][0][i], feat0=kwargs['state'][1][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i],step_ct=kwargs['step_ct'], id_set=kwargs['id_set'][i])

    def tight(self):
        steps_all = list(self.agent.keys())

        ###add by denote
        self.all_id=steps_all 
        self.alive_steps=dict()
        for key in self.agent.keys():
            self.alive_steps[key]=len(self.agent[key].alive)
        # print ("agent_alive_steps----:",self.alive_steps)

        print("self.max_len:",self.max_len)
        #

        for ele in steps_all:
            tmp = self.agent[ele].pulled()
            self._new_add += len(tmp['obs0'])
            self._flush(**tmp)
        self._new_add_save=self._new_add#add by denote
        print("*&*&*&*&*&*&*&*&*&*&*&",self._new_add_save)

        self.agent = dict()  # clear

    def sample(self):
        #select the sample of the batch, each samlpe contains no more than 64 agent(combat casualty) 
        #Bruce method
        while 1:
            bad_happen=False
            x =random.randint(0, self.nb_entries)
            step_now=self.step_ct.sample(x)
            x_now=x
            id_now=self.id_set.sample(x_now)

            if step_now==self.sub_len-1:#bad choice
                continue
            
            ## find the left x
            while 1:
                x=x-1
                id_now_refresh=self.id_set.sample(x)
                if id_now_refresh>=id_now:
                    left_x=x+1
                    break
                id_now=id_now_refresh#update the id index

                if x==0:
                    left_x=0
                    break
                if self.step_ct.sample(x)!=step_now:
                    left_x=x+1
                    break
            #end while
            
            ## find the right x
            x=x_now
            id_now=self.id_set.sample(x_now)
            while 1:
                x=x+1
                id_now_refresh=self.id_set.sample(x)
                if id_now_refresh<=id_now:
                    break
                id_now=id_now_refresh#update the id index

                if x==self.nb_entries:#bad
                    bad_happen=True
                    break
                if self.step_ct.sample(x)!=step_now:#may be good
                    if self.step_ct.sample(x)==step_now+1:#good
                        break
                    
                    if self.step_ct.sample(x)!=step_now+1:#bad
                        bad_happen=True
                        break
            if bad_happen==True:
                continue
            right_x=x-1
            
            # find the next left x
            next_left_x=right_x+1
            if next_left_x>self.nb_entries-2:
                continue
            if self.step_ct.sample(next_left_x)!=self.step_ct.sample(right_x)+1:
                continue
            # find the next right x    
            tmp_x=next_left_x
            while 1:
                tmp_x=tmp_x+1
                if self.step_ct.sample(tmp_x)!=step_now+1 or tmp_x==self.nb_entries:
                    break
            next_right_x=tmp_x-1
            
            # find the common id_set between the former and the after episodes(whole in per step)
            x_set_pre,x_set_aft=[],[]
            for setp_x in range(left_x,right_x+1):
                x_set_pre.append(self.id_set.sample(setp_x))
            for setp_x in range(next_left_x,next_right_x+1):
                x_set_aft.append(self.id_set.sample(setp_x))
            ID_set_common=list(set(x_set_pre).intersection(set(x_set_aft)))
            if len(ID_set_common)==0:
                continue
            break
        ############random select common ID
        ####produce the choose id for batch. id_set should in the common agent'ID set. 
        x_choose_pre=[]
        x_choose_aft=[]
        for x in range(left_x,right_x+1):
            if self.id_set.sample(x) in ID_set_common:
                x_choose_pre.append(x)
        for x in range(next_left_x,next_right_x+1):
            if self.id_set.sample(x) in ID_set_common:
                x_choose_aft.append(x)
            
        #
        x_choose_pre_=np.array(x_choose_pre)
        x_choose_aft_=np.array(x_choose_aft)
        # shuffle_x =np.random.choice(len(x_choose_pre_), len(x_choose_pre_), replace=False, p=None)
        shuffle_x =np.random.permutation(len(x_choose_pre_))
        x_choose_pre_final=x_choose_pre_[shuffle_x]
        x_choose_aft_final=x_choose_aft_[shuffle_x]
        
        # for print
        self.tmp1=x_choose_pre_
        self.tmp2=x_choose_aft_
        self.tmp3=ID_set_common

        #batching
        obs,feature,act_prob,actions=[],[],[],[]
        rewards,dones,masks=[],[],[]
        for x in x_choose_pre_final:
            if self.id_set.sample(x) in ID_set_common:
                obs.append(self.obs0.sample(x))
                feature.append(self.feat0.sample(x))
                act_prob.append(self.prob.sample(x))
                rewards.append(self.rewards.sample(x))
                dones.append(1-self.alive.sample(x))
                actions.append(self.actions.sample(x))
        masks=1-np.array(dones)
        #
        obs_next,feature_next,act_prob_next=[],[],[]
        for x in x_choose_aft_final:
            if self.id_set.sample(x) in ID_set_common:
                obs_next.append(self.obs0.sample(x))
                feature_next.append(self.feat0.sample(x))
                act_prob_next.append(self.prob.sample(x))

        if self.use_mean:
            return np.array(obs), np.array(feature), np.array(actions), np.array(act_prob), np.array(obs_next), np.array(feature_next), np.array(act_prob_next), np.array(rewards), np.array(dones), np.array(masks)
        else:
            return np.array(obs), np.array(feature), np.array(obs_next), np.array(feature_next), np.array(dones), np.array(rewards), np.array(actions), np.array(masks)
    
    def printed(self):
        print("\n1:",len(self.tmp1))
        print("2:",len(self.tmp2))
        print("\n3:",self.id_set.sample(self.tmp1))
        print("4:",self.id_set.sample(self.tmp2))
        print("common_id:",self.tmp3,len(self.tmp3))
        print("\n5:",self.step_ct.sample(self.tmp1))
        print("6:",self.step_ct.sample(self.tmp2))
        print("\n7:",len(self.step_ct.sample(self.tmp1)))
        print("8:",len(self.step_ct.sample(self.tmp2)))
        return 0
            
    def get_batch_num(self):
        res = self._new_add * 2 // self.batch_size
        print('\n[INFO] Length of buffer and new add:', len(self.obs0), self._new_add,' ,res:',res)
        self._new_add = 0
        return res

    @property
    def nb_entries(self):
        return len(self.obs0)


class SummaryObj:
    """
    Define a summary holder
    """
    def __init__(self, log_dir, log_name, n_group=1):
        self.name_set = set()
        self.gra = tf.Graph()
        self.n_group = n_group

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with self.gra.as_default():
            self.sess = tf.Session(graph=self.gra, config=sess_config)
            self.train_writer = tf.summary.FileWriter(log_dir + "/" + log_name, graph=tf.get_default_graph())
            self.sess.run(tf.global_variables_initializer())

    def register(self, name_list):
        """Register summary operations with a list contains names for these operations

        Parameters
        ----------
        name_list: list, contains name whose type is str
        """

        with self.gra.as_default():
            for name in name_list:
                if name in self.name_set:
                    raise Exception("You cannot define different operations with same name: `{}`".format(name))
                self.name_set.add(name)
                setattr(self, name, [tf.placeholder(tf.float32, shape=None, name='Agent_{}_{}'.format(i, name))
                                     for i in range(self.n_group)])
                setattr(self, name + "_op", [tf.summary.scalar('Agent_{}_{}_op'.format(i, name), getattr(self, name)[i])
                                             for i in range(self.n_group)])

    def write(self, summary_dict, step):
        """Write summary related to a certain step

        Parameters
        ----------
        summary_dict: dict, summary value dict
        step: int, global step
        """

        assert isinstance(summary_dict, dict)

        for key, value in summary_dict.items():
            if key not in self.name_set:
                raise Exception("Undefined operation: `{}`".format(key))
            if isinstance(value, list):
                for i in range(self.n_group):
                    self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[i], feed_dict={
                        getattr(self, key)[i]: value[i]}), global_step=step)
            else:
                self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[0], feed_dict={
                        getattr(self, key)[0]: value}), global_step=step)


class Runner(object):
    def __init__(self, sess, env, handles, map_size, max_steps, neighbor_nums, models,
                play_handle, render_every=None, save_every=None, tau=None, log_name=None, log_dir=None, model_dir=None, train=False,original_main=False,original_oppo=False):
        """Initialize runner

        Parameters
        ----------
        sess: tf.Session
            session
        env: magent.GridWorld
            environment handle
        handles: list
            group handles
        map_size: int
            map size of grid world
        max_steps: int
            the maximum of stages in a episode
        render_every: int
            render environment interval
        save_every: int
            states the interval of evaluation for self-play update
        models: list
            contains models
        play_handle: method like
            run game
        tau: float
            tau index for self-play update
        log_name: str
            define the name of log dir
        log_dir: str
            donates the directory of logs
        model_dir: str
            donates the dircetory of models
        """
        self.env = env
        self.models = models
        self.max_steps = max_steps
        self.handles = handles
        self.map_size = map_size
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.train = train
        self.original_main=original_main
        self.original_oppo=original_oppo
        self.wait_update=1
        self.neighbor_nums=neighbor_nums
        
        if self.train:
            print("#########################Begining######################")
            self.summary = SummaryObj(log_name=log_name, log_dir=log_dir)

            summary_items = ['ave_agent_reward', 'total_reward', 'kill', "Sum_Reward", "Kill_Sum"]
            self.summary.register(summary_items)  # summary register
            self.summary_items = summary_items

            assert isinstance(sess, tf.Session)
            assert self.models[0].name_scope != self.models[1].name_scope
            self.sess = sess

            l_vars, r_vars = self.models[0].vars, self.models[1].vars
            print("#############len(l_vars)###############",len(l_vars))
            assert len(l_vars) == len(r_vars)
            ###
            self.sp_op = [tf.assign(r_vars[i], (1. - tau) * l_vars[i] + tau * r_vars[i])
                                for i in range(len(l_vars))]

            # if not os.path.exists(self.model_dir):
            #     os.makedirs(self.model_dir)

    def run(self, variant_eps, iteration, win_cnt=None,draw=None,kill_ratio=None):
        info = {'main': None, 'opponent': None}

        # pass
        info['main'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.}
        info['opponent'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.}

        max_nums, nums, agent_r_records, total_rewards = self.play(env=self.env, n_round=iteration, map_size=self.map_size, max_steps=self.max_steps, neighbor_nums=self.neighbor_nums,handles=self.handles,
                    models=self.models, print_every=50, eps=variant_eps, render=(iteration + 1) % self.render_every==0 if self.render_every > 0 else False, train=self.train,original_main=self.original_main,original_oppo=self.original_oppo)

        for i, tag in enumerate(['main', 'opponent']):
            info[tag]['total_reward'] = total_rewards[i]
            info[tag]['kill'] = max_nums[i] - nums[1 - i]
            info[tag]['ave_agent_reward'] = agent_r_records[i]

        if self.train:
            print('\n[---main] {}'.format(info['main']))
            print('[---oppo] {}'.format(info['opponent']))
            
            #early stop
            if abs(info['main']['total_reward']-info['opponent']['total_reward'])<0.00001:
                self.wait_update=self.wait_update+1
            else:
                self.wait_update=1

            if self.wait_update>50:
                raise Exception("Early stop, try again.")
            ##################### if info['main']['total_reward'] > info['opponent']['total_reward'] and self.wait_update>(iteration/100):#update the opponent

            if info['main']['total_reward'] > info['opponent']['total_reward']:
            #####if info['main']['kill'] > info['opponent']['kill']:#update the opponent
                print(Color.INFO.format('\n[INFO] Begin self-play Update ...'))
                self.sess.run(self.sp_op)
                print(Color.INFO.format('[INFO] Self-play Updated\n'))
            #save
            if self.save_every and (iteration + 1) % self.save_every == 0:#save the model
                print(Color.INFO.format('[INFO] Saving model ...'))
                self.models[0].save(self.model_dir + '-0', iteration)
                self.models[1].save(self.model_dir + '-1', iteration)

                self.summary.write(info['main'], iteration)
                
        else:
            print('[INFO] {0} \n {1}'.format(info['main'], info['opponent']))
            kill_ratio['main']+=info['main']['kill']
            kill_ratio['opponent']+=info['opponent']['kill']
            ####### how to win
            if info['main']['kill'] > info['opponent']['kill']:
                win_cnt['main'] += 1
            elif info['main']['kill'] < info['opponent']['kill']:
                win_cnt['opponent'] += 1
            else:#draw
                draw['main']+=1
                draw['opponent']+=1
            #######
            print("\nTotal games:",win_cnt['main']+win_cnt['opponent']+draw['main'])
            print("me-wins:",win_cnt['main'])
            print("oppo-wins:",win_cnt['opponent'])
            print("draw",draw['main'])
            print("win_ratio",win_cnt['main']/(win_cnt['main']+win_cnt['opponent']+draw['main']))
            print("kill",kill_ratio['main'],"be-killed",kill_ratio['opponent'],"kill-ratio",kill_ratio['main']/max(0.01,kill_ratio['opponent']))
        
        return total_rewards,agent_r_records
