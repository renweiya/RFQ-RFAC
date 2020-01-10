"""Self Play
"""

import argparse
import os
import tensorflow as tf
import numpy as np
import magent

from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools
from examples.battle_model.algo import tools_new
from examples.battle_model.senario_battle import play  #check
import pylab

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'il','mfq','rfq','ac','mfac','rfac'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--save_every', type=int, default=2000, help='decide the self-play update interval')#2000
    parser.add_argument('--render_every', type=int, default=100, help='decide the render interval')#100
    
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=2000, help='set the trainning round')#2000
    parser.add_argument('--render', action='store_true', default=True, help='render or not (if true, will render every save)')#store_true
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # 40,then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')  #400
    parser.add_argument('--neighbor_nums', type=int, default=-1, help='set number of neighbors')  #-1 means all-connected graph.

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render/{}'.format(args.algo)))
    handles = env.get_handles()

    #config
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    parent_tmp_dir='data/tmp'
    parent_model_dir='data/models'
    parent_result_save_dir='result_save'
    log_dir = os.path.join(BASE_DIR, parent_tmp_dir+'/{}'.format(args.algo))
    model_dir = os.path.join(BASE_DIR, parent_model_dir+'/{}'.format(args.algo))
    save_txt_dir=os.path.join(BASE_DIR, parent_result_save_dir+'/{}'.format(args.algo))
    if not os.path.exists(log_dir):
                os.makedirs(log_dir)
    if not os.path.exists(model_dir):
                os.makedirs(model_dir)
    if not os.path.exists(save_txt_dir):
                os.makedirs(save_txt_dir)

    #train option
    if args.algo in ['ac', 'il','mfq','mfac']:
        original_main = True
        original_oppo = True
    else:
        original_main = False
        original_oppo = False
    if args.neighbor_nums==-1:
        bias_flag=False
    else:
        bias_flag=True
    outprint = True
    start_from = 0

    sess = tf.Session(config=tf_config)
    tau=0.01 #0.01
    #define model
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps, args.neighbor_nums,bias_flag), spawn_ai(args.algo, sess, env, handles[1], args.algo + '-opponent', args.max_steps, args.neighbor_nums,bias_flag)]
    sess.run(tf.global_variables_initializer())
    runner = tools_new.Runner(sess, env, handles, args.map_size, args.max_steps, args.neighbor_nums, models, play,
                            render_every=args.render_every if args.render else 0, save_every=args.save_every, tau=tau, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, train=True,original_main=original_main,original_oppo=original_oppo)
    eps_record=[]
    total_rewards_all=[]
    agent_r_records_all=[]
    
    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        # print("eps: ",eps)
        print("training the:",args.algo)
        if outprint==1:
            [total_rewards,agent_r_records]=runner.run(eps, k)
        if outprint==0:
            runner.run(eps, k)
            
        eps_record.append(eps)
        if outprint==1:
            total_rewards_all.append(total_rewards)
            agent_r_records_all.append(agent_r_records)
            if 1:#k%49==0:
                np.savetxt(os.path.join(save_txt_dir,'total_rewards'),total_rewards_all)
                np.savetxt(os.path.join(save_txt_dir,'agent_r_records'),agent_r_records_all)
    print("okok")

