import random
import math
import numpy as np

from examples.battle_model.adjacency_matrix import adjacency_matrix

total_num=[0,0]
def generate_map(env, map_size, handles,fix):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size

    init_num=64 #number of agents for each group
    pos_0 = []
    pos_1 = []
    first_flag=0
    pos_dict_x={}
    pos_dict_y={}
    pos_dict_iter=0
    for i in range(1,width-1):
        for j in range(1,height-1):
            pos_dict_x[pos_dict_iter]=i
            pos_dict_y[pos_dict_iter]=j
            pos_dict_iter=pos_dict_iter+1
    print("max_agents",pos_dict_iter)
    select_=np.random.permutation((width-2)*(height-2))
    
    init_num_select_x=0
    init_num_select_y=0
    this_iter_=0
    while 1:
        if first_flag==0 and init_num_select_x<init_num:
            pos_0.append([pos_dict_x[select_[this_iter_]], pos_dict_y[select_[this_iter_]], 0])
            first_flag=1
            init_num_select_x=init_num_select_x+1
            this_iter_=this_iter_+1
        elif init_num_select_y<init_num:
            pos_1.append([pos_dict_x[select_[this_iter_]], pos_dict_y[select_[this_iter_]], 0])
            first_flag=0
            init_num_select_y=init_num_select_y+1
            this_iter_=this_iter_+1
            
        if init_num_select_x>init_num-1 and init_num_select_y>init_num-1:
            break
    #
    leftID = random.randint(0, 1)
    rightID = 1 - leftID

    env.add_agents(handles[leftID], method="custom", pos=pos_0)
    env.add_agents(handles[rightID], method="custom", pos=pos_1)

    total_num[0]=env.get_num(handles[0])
    total_num[1]=env.get_num(handles[1])
def play(env, n_round, map_size, max_steps, neighbor_nums, handles, models, print_every, eps=1.0, render=False, train=False, original_main=False,original_oppo=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles,fix=0)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.4f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]
    ids_now = [None for _ in range(n_group)]
    rewards_record= [None for _ in range(n_group)]
    former_act_prob_new = [[] for _ in range(n_group)]
    neighbor_act = [[] for _ in range(n_group)]#mean action of neighbors
    former_neighbor_act = [[] for _ in range(n_group)]#former step

    W_matrix= [[] for _ in range(n_group)]

    be_killed = [0,0]#add by denote

    if neighbor_nums==-1:
        print("all connected graph")
        local_or_not=False #all connected graph
    else:
        print("local graph with neighbors:",neighbor_nums)
        local_or_not=True #local graph
    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])

        # print("step",step_ct)
        ##########
        if local_or_not==True:
            for i in range(n_group):
                neighbor_num=min(neighbor_nums,len(ids[i]))
                W_matrix[i]=adjacency_matrix(state[i][1],neighbor_num)#need time
        ##########
        W_act=[[] for _ in range(n_group)]
        if step_ct==0:
            former_act_prob_new = [np.zeros((len(ids[0]), env.get_action_space(handles[0])[0])), np.zeros((len(ids[1]), env.get_action_space(handles[1])[0]))]
            former_neighbor_act= [np.zeros((len(ids[0]), env.get_action_space(handles[0])[0])), np.zeros((len(ids[1]), env.get_action_space(handles[1])[0]))]
        # print("ids",ids)
        ###1
        for i in range(n_group): 
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            if original_main==True:
                #1.local graph
                if local_or_not==True:
                    acts[i] = models[i].act(state=state[i],prob=former_neighbor_act[i], eps=eps) 
                #2.all connected graph
                else:
                    acts[i] = models[i].act(state=state[i],prob=former_act_prob[i], eps=eps) 
            else:
                #1.local graph
                if local_or_not==True:
                    W_act[i]=W_matrix[i]
                #2.all connected graph
                else:
                    W_act[i]=np.ones((len(state[i][0]),len(state[i][0])))####/len(state[i][0])
                #
                acts[i] = models[i].act(state=state[i],W=W_act[i], prob=former_act_prob_new[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        # print("former_act_prob_new[i]",former_act_prob_new[i])
        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])
            
        ##
        ######## ensure ids[0] is sorted from 1 to n.
        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'id_set': ids[0],'step_ct':step_ct
        }

        if local_or_not==True:
            buffer['prob'] = former_neighbor_act[0] #last iter,important. mean act for neighbor agents
        else:
            buffer['prob'] = former_act_prob[0] #last iter,important. mean act for all agents

        buffer['prob_all'] = former_act_prob_new[0] #last iter. act of each agent
        buffer['ids'] = ids[0]

        if step_ct>0:
            assert len(state[0][0])==len(acts[0])

        for i in range(n_group):#mean action of all agents
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)
            former_act_prob_new[i]=np.eye(n_action[i])[acts[i]]#important,original action of agent
        
        #local connect,compute the mean action of neighbors
        if local_or_not==True:
            neighbor_act= [np.zeros((len(ids[0]), env.get_action_space(handles[0])[0])), np.zeros((len(ids[1]), env.get_action_space(handles[1])[0]))]
            for i in range(n_group):#mean action of all agents
                for j in range(len(acts[i])):
                    non_zero_index=np.where(W_matrix[i][j]!=0)
                    acts_now=acts[i][non_zero_index]
                    neighbor_act[i][j]=np.mean(list(map(lambda z: np.eye(n_action[i])[z], acts_now)), axis=0, keepdims=True)
                    
        #####
        if train:
            models[0].flush_buffer(**buffer)
        
        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards_record[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards_record[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()
        nums_last = [env.get_num(handle) for handle in handles] #add by denote
        #compute the common former_act_prob_new[i] before and after env.clear_dead()
        for i in range(n_group):
            ids_now[i] = env.get_agent_id(handles[i])
        for i in range(n_group):
            prob_tmp=[]  #action of agent itself
            neighbor_tmp=[]#mean action among neighbors
            ID_common=list(set(ids[i]).intersection(set(ids_now[i])))
            ID_common.sort()
            lines=0
            for id in ids[i]:
                if id in ID_common:
                    prob_tmp.append(former_act_prob_new[i][lines])
                    if local_or_not==True:
                        neighbor_tmp.append(neighbor_act[i][lines])
                lines=lines+1 #after if
            former_act_prob_new[i]=np.array(prob_tmp)  #(formal step)action of agent,each row is a agent
            if local_or_not==True:
                former_neighbor_act[i]=np.array(neighbor_tmp)#(formal step)mean action of neighbors of agent,each row is a agent
            

        ## add by denote--add agent
        be_killed_0 = total_num[0] - len(env.get_agent_id(handles[0]))
        if be_killed_0>0:
            be_killed[0]+=be_killed_0
        be_killed_1 = total_num[1] - len(env.get_agent_id(handles[1]))
        if be_killed_1>0:
            be_killed[1]+=be_killed_1
        # add end
        info = {"Ave-Reward": np.round(rewards_record, decimals=6), "NUM": nums_last}

        step_ct += 1

        if step_ct % print_every == 0 or done:
            print("> step #{}, info: {}".format(step_ct, info))
        if step_ct==max_steps or done:
            print("kill and be killed",be_killed_1,be_killed_0)
            print("original_main",original_main)

    if train:
        models[0].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums_last, mean_rewards, total_rewards


def battle(env, n_round, map_size, max_steps, neighbor_nums, handles, models, print_every, eps=1.0, render=False, train=False, original_main=False, original_oppo=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles,fix=0)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]
    former_act_prob_new = [[] for _ in range(n_group)]
    ids_now = [None for _ in range(n_group)]
    neighbor_act = [[] for _ in range(n_group)]#mean action of neighbors
    former_neighbor_act = [[] for _ in range(n_group)]#former step
    W_matrix= [[] for _ in range(n_group)]
    ##########
    if neighbor_nums==-1:
        print("all connected graph")
        local_or_not=False #all connected graph
    else:
        print("local graph with neighbors:",neighbor_nums)
        local_or_not=True #local graph
    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])
        
        if step_ct==0:
            former_act_prob_new = [np.zeros((len(ids[0]), env.get_action_space(handles[0])[0])), np.zeros((len(ids[1]), env.get_action_space(handles[1])[0]))]
            former_neighbor_act= [np.zeros((len(ids[0]), env.get_action_space(handles[0])[0])), np.zeros((len(ids[1]), env.get_action_space(handles[1])[0]))]
        
        ### act based on the trained model
        for i in range(n_group): 
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))

        #
        if local_or_not==True:
            for i in range(n_group):
                neighbor_num=min(neighbor_nums,len(ids[i]))
                W_matrix[i]=adjacency_matrix(state[i][1],neighbor_num)#need time
        #

        W_act=[[] for _ in range(n_group)]    
        if original_main==True:
            #1.local graph
            if local_or_not==True:
                acts[0] = models[0].act(state=state[0],prob=former_neighbor_act[0], eps=eps) 
            #2.all connected graph
            else:
                acts[0] = models[0].act(state=state[0],prob=former_act_prob[0], eps=eps) 
        else:
            #1.local graph
            if local_or_not==True:
                W_act[0]=W_matrix[0]
            #2.all connected graph
            else:
                W_act[0]=np.ones((len(state[0][0]),len(state[0][0])))   #####/len(state[0][0])#-np.eye((len(state[0][0])))
            
            acts[0] = models[0].act(state=state[0],W=W_act[0], prob=former_act_prob_new[0], eps=eps)

        if original_oppo==True:
            #1.local graph
            if local_or_not==True:
                acts[1] = models[1].act(state=state[1],prob=former_neighbor_act[1], eps=eps) 
            #2.all connected graph
            else:
                acts[1] = models[1].act(state=state[1],prob=former_act_prob[1], eps=eps) 
        else:
            #1.local graph
            if local_or_not==True:
                W_act[1]=W_matrix[1]
            #2.all connected graph
            else:
                W_act[1]=np.ones((len(state[1][0]),len(state[1][0])))   #####/len(state[1][0])#-np.eye((len(state[1][0])))
            
            acts[1] = models[1].act(state=state[1],W=W_act[1], prob=former_act_prob_new[1], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)
            former_act_prob_new[i]=np.eye(n_action[i])[acts[i]]#important
        #local connect,compute the mean action of neighbors
        if local_or_not==True:
            neighbor_act= [np.zeros((len(ids[0]), env.get_action_space(handles[0])[0])), np.zeros((len(ids[1]), env.get_action_space(handles[1])[0]))]
            for i in range(n_group):#mean action of all agents
                for j in range(len(acts[i])):
                    non_zero_index=np.where(W_matrix[i][j]!=0)
                    acts_now=acts[i][non_zero_index]
                    neighbor_act[i][j]=np.mean(list(map(lambda z: np.eye(n_action[i])[z], acts_now)), axis=0, keepdims=True)
                
        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()
        # clear dead agents
        env.clear_dead()
        nums_last = [env.get_num(handle) for handle in handles]
        #compute the common former_act_prob_new[i] before and after env.clear_dead()
        for i in range(n_group):
            ids_now[i] = env.get_agent_id(handles[i])
        for i in range(n_group):
            prob_tmp=[]
            neighbor_tmp=[]
            ID_common=list(set(ids[i]).intersection(set(ids_now[i])))
            ID_common.sort()
            lines=0
            for id in ids[i]:
                if id in ID_common:
                    prob_tmp.append(former_act_prob_new[i][lines])
                    if local_or_not==True:
                        neighbor_tmp.append(neighbor_act[i][lines])
                lines=lines+1
            # print("before_former_act_prob_new",former_act_prob_new[i].shape)
            former_act_prob_new[i]=np.array(prob_tmp)  #(formal step)action of agent,each row is a agent
            if local_or_not==True:
                former_neighbor_act[i]=np.array(neighbor_tmp)#(formal step)mean action of neighbors of agent,each row is a agent
            
        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums_last}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums_last, mean_rewards, total_rewards

