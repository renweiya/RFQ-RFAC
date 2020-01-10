"""Battle
"""

import argparse
import os
import tensorflow as tf
import numpy as np
import magent

from examples.battle_model.algo import spawn_ai,spawn_ai_test
from examples.battle_model.algo import tools
from examples.battle_model.algo import tools_new
from examples.battle_model.senario_battle import battle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'il','mfq','rfq','ac','mfac','rfac'}, help='choose an algorithm from the preset', required=False)
    parser.add_argument('--oppo', type=str, choices={'il','mfq','rfq','ac','mfac','rfac'}, help='indicate the opponent model')
    parser.add_argument('--n_round', type=int, default=1, help='set the trainning round')
    parser.add_argument('--render', action='store_true',default=True,  help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # 40,then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--idx', nargs='*', required=False)
    parser.add_argument('--neighbor_nums', type=int, default=-1, help='set neighbors')  #-1 means all-connected graph.

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render_real'))
    handles = env.get_handles()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    sess = tf.Session(config=tf_config)
    ####
    #generate all players
    method_algo_idx={}
    #config
    algo=['il','mfq','rfq','ac','mfac','rfac']
    the_move='-me'
    # the_move='-opponent'
    # algo_idx=['1999A','1999B','1999C',
    #     '1999A','1999B','1999C',
    #     '1999A','1999B','1999C',
    #     '1999A','1999B','1999C',
    #     '1999A','1999B','1999C',
    #     '1999A','1999B','1999C'] #train round
    # method_algo_idx[0]={0,1,2} #il
    # method_algo_idx[1]={3,4,5} #mfq
    # method_algo_idx[2]={6,7,8} #rfq
    # method_algo_idx[3]={9,10,11} #ac
    # method_algo_idx[4]={12,13,14} #mfac
    # method_algo_idx[5]={15,16,17} #rfac
    #config end
    algo_idx=['1999A',
        '1999A',
        '1999A',
        '1999A',
        '1999A',
        '1999A'] #train round
    method_algo_idx[0]={0} #il
    method_algo_idx[1]={1} #mfq
    method_algo_idx[2]={2} #rfq
    method_algo_idx[3]={3} #ac
    method_algo_idx[4]={4} #mfac
    method_algo_idx[5]={5} #rfac
    ##
    if args.neighbor_nums==-1:
        bias_flag=False
    else:
        bias_flag=True
    models_all = [None for _ in range(len(algo))]
    for i in range(len(algo)):
        models_all[i]=spawn_ai_test(algo[i], sess, env, handles[0],algo[i] + the_move, args.max_steps, args.neighbor_nums, bias_flag)
    #
    sess.run(tf.global_variables_initializer())
   
    #load parameters
    model_palyer = [None for _ in range(len(algo_idx))]
    for i in range(len(algo_idx)):
        for z in method_algo_idx:
            if i in method_algo_idx[z]:
                j=z
                break
        print("i,j",i,j)
        model_palyer[i]=models_all[j]
        model_dir = os.path.join(BASE_DIR, 'data/models/{}-0'.format(algo[j]))
        model_palyer[i].load(model_dir, step=algo_idx[i])
    #############################
    iter=0
    algo_num=len(algo_idx)
    model_num=len(algo)
    R=1400*np.ones(algo_num)
    player_games=np.zeros(algo_num)
    player_wins=np.zeros(algo_num)
    players_draw=np.zeros(algo_num)
    killed=np.zeros(algo_num)
    be_killed=np.zeros(algo_num)
    win_table=np.zeros((algo_num,algo_num))
    draw_table=np.zeros((algo_num,algo_num))
    save_txt_dir=os.path.join(BASE_DIR, 'ELO_save/')
    if not os.path.exists(save_txt_dir):
                os.makedirs(save_txt_dir)
    while 1:
        #select two player
        index_rand=np.random.permutation(algo_num)
        index=[index_rand[0],index_rand[1]]
        model_index=[-1,-1]
        for i in method_algo_idx:
            if index[0] in method_algo_idx[i]:
                model_index[0]=i
                break
        for i in method_algo_idx:
            if index[1] in method_algo_idx[i]:
                model_index[1]=i
                break

        print("\n\n        #########################\nAn new game, select two wild war ago:",algo[model_index[0]],format(algo_idx[index[0]]),"and ",algo[model_index[1]],format(algo_idx[index[1]]))
        iter=iter+1
        print("Total games:",iter)
        print("index",index[0],index[1])
        print("model_index",model_index[0],model_index[1])
        #select model
        models=[model_palyer[index[0]],model_palyer[index[1]]]

        if algo[model_index[0]] in ['ac','il','mfq','mfac']:
            original_main = True
        else:
            original_main = False

        if algo[model_index[1]] in ['ac','il','mfq','mfac']:
            original_oppo = True
        else:
            original_oppo = False

        runner = tools_new.Runner(sess, env, handles, args.map_size, args.max_steps, args.neighbor_nums, models, battle, render_every=100,original_main=original_main,original_oppo=original_oppo)
        win_cnt = {'main': 0, 'opponent': 0}
        draw = {'main': 0, 'opponent': 0}
        kill_ratio = {'main': 0, 'opponent': 0}

        for k in range(0, args.n_round):
            runner.run(0.0, k, win_cnt=win_cnt,draw=draw,kill_ratio=kill_ratio)
        
        #
        main_win=0
        if win_cnt['main']>win_cnt['opponent']:
            main_win=1 #main win
        elif win_cnt['opponent']>win_cnt['main']:
            main_win=0 #oppo win
        else:
            main_win=0.5 #draw
        ###ELO score     
        R_A=R[index[0]]
        R_B=R[index[1]]
        E_A=1/(1+(10**((R_B-R_A)/400)))
        E_B=1-E_A
        delta=32*(main_win-E_A)
        R[index[0]]=R[index[0]]+delta
        R[index[1]]=R[index[1]]-delta
        ###
        
        ###
        player_games[index[0]]=player_games[index[0]]+1
        player_games[index[1]]=player_games[index[1]]+1
        if main_win==1:
            player_wins[index[0]]=player_wins[index[0]]+1
            win_table[index[0]][index[1]]=win_table[index[0]][index[1]]+1
        elif main_win==0:
            player_wins[index[1]]=player_wins[index[1]]+1
            win_table[index[1]][index[0]]=win_table[index[1]][index[0]]+1
        else:
            draw_table[index[0]][index[1]]=draw_table[index[0]][index[1]]+1
            draw_table[index[1]][index[0]]=draw_table[index[1]][index[0]]+1
            players_draw[index[0]]=players_draw[index[0]]+1
            players_draw[index[1]]=players_draw[index[1]]+1

        killed[index[0]]=killed[index[0]]+kill_ratio['main']
        killed[index[1]]=killed[index[1]]+kill_ratio['opponent']
        be_killed[index[0]]=be_killed[index[0]]+kill_ratio['opponent']
        be_killed[index[1]]=be_killed[index[1]]+kill_ratio['main']

        print("\ntotal_games:",iter,",each join in:",player_games)
        print("player_wins:",player_wins)
        print("players_draw:",players_draw)
        print("\nwin_rate:",np.round(player_wins/player_games, 2))
        print("\nELO_score",np.round(R, 2))
        print("\nkilled",killed)
        print("be_killed",be_killed)
        kill_ratio=np.round(killed/be_killed, 4)
        print("kill_ratio",kill_ratio)

        if iter%50==0:
            print("\n#######win_table among players:\n",win_table)

        
        if iter%50==0:
            np.savetxt(os.path.join(save_txt_dir,'R'),R)
            np.savetxt(os.path.join(save_txt_dir,'player_games'),player_games)
            np.savetxt(os.path.join(save_txt_dir,'player_wins'),player_wins)
            np.savetxt(os.path.join(save_txt_dir,'players_draw'),players_draw)
            np.savetxt(os.path.join(save_txt_dir,'draw_table'),draw_table)
            np.savetxt(os.path.join(save_txt_dir,'win_table'),win_table)
            np.savetxt(os.path.join(save_txt_dir,'killed'),killed)
            np.savetxt(os.path.join(save_txt_dir,'be_killed'),be_killed)
            np.savetxt(os.path.join(save_txt_dir,'kill_ratio'),kill_ratio)
        
        
        if iter>1000000:
            break
