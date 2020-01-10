"""Battle
"""

import argparse
import os
import tensorflow as tf
import numpy as np
import magent

from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools
from examples.battle_model.algo import tools_new
from examples.battle_model.senario_battle import battle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'il','mfq','rfq','ac','mfac','rfac'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--oppo', type=str, choices={'il','mfq','rfq','ac','mfac','rfac'}, help='indicate the opponent model')
    parser.add_argument('--n_round', type=int, default=30, help='set the trainning round')
    parser.add_argument('--render', action='store_true',default=True,  help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # 40,then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--idx', nargs='*', required=True)
    parser.add_argument('--neighbor_nums', type=int, default=-1, help='set neighbors')  #-1 means all-connected graph.

    # python battle.py --algo mfq --oppo rfac --idx {1999A,1999A}

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    handles = env.get_handles()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    #
    main_model_dir = os.path.join(BASE_DIR, 'data/models/{}-0'.format(args.algo))
    if args.algo==args.oppo:
        oppo_model_dir = os.path.join(BASE_DIR, 'data/models/{}-1'.format(args.oppo))
    else:
        oppo_model_dir = os.path.join(BASE_DIR, 'data/models/{}-0'.format(args.oppo))
    #
    if args.algo in ['ac', 'il','mfq','mfac']:
        original_main = True
    else:
        original_main = False
    #
    if args.oppo in ['ac', 'il','mfq','mfac']:
        original_oppo = True
    else:
        original_oppo = False
    #
    if args.neighbor_nums==-1:
        bias_flag=False
    else:
        bias_flag=True
    sess = tf.Session(config=tf_config)
    #
    if args.algo==args.oppo:
        models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps, args.neighbor_nums, bias_flag), spawn_ai(args.oppo, sess, env, handles[1], args.oppo + '-opponent', args.max_steps, args.neighbor_nums, bias_flag)]
    
    else:
        models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps, args.neighbor_nums, bias_flag), spawn_ai(args.oppo, sess, env, handles[1], args.oppo + '-me', args.max_steps, args.neighbor_nums, bias_flag)]
    
    #
    sess.run(tf.global_variables_initializer())

    models[0].load(main_model_dir, step=args.idx[0])
    models[1].load(oppo_model_dir, step=args.idx[1])

    runner = tools_new.Runner(sess, env, handles, args.map_size, args.max_steps, args.neighbor_nums, models, battle, render_every=1,original_main=original_main,original_oppo=original_oppo)
    win_cnt = {'main': 0, 'opponent': 0}
    draw = {'main': 0, 'opponent': 0}
    kill_ratio = {'main': 0, 'opponent': 0}

    for k in range(0, args.n_round):
        runner.run(0.0, k, win_cnt=win_cnt,draw=draw,kill_ratio=kill_ratio)

    print('\n[*] >>> WIN_RATE: [{0}] {1} / [{2}] {3}'.format(args.algo, win_cnt['main'] / args.n_round, args.oppo, win_cnt['opponent'] / args.n_round))
    print("Total games:",win_cnt['main']+win_cnt['opponent']+draw['main'])
    print("me-wins:",win_cnt['main'])
    print("oppo-wins:",win_cnt['opponent'])
    print("draw",draw['main'])
    print("win_ratio",win_cnt['main']/(win_cnt['main']+win_cnt['opponent']+draw['main']))
    print("kill",kill_ratio['main'],"be-killed",kill_ratio['opponent'],"kill-ratio",kill_ratio['main']/max(0.01,kill_ratio['opponent']))
 
