from . import ac
from . import ac_gradient
from . import q_learning

#DQN class
IL = q_learning.DQN
MFQ = q_learning.MFQ
RFQ = q_learning.RFQ

#ac class
AC = ac.ActorCritic
MFAC = ac.MFAC
RFAC = ac_gradient.RFAC

def spawn_ai(algo_name, sess, env, handle, human_name, max_steps,neighbor_nums,bias_flag):
    if algo_name == 'il':
        model = IL(sess, human_name, handle, env, max_steps, memory_size=500000)#80000
    elif algo_name == 'mfq':
        model = MFQ(sess, human_name, handle, env, max_steps, memory_size=500000)#80000
    elif algo_name == 'rfq':
        model = RFQ(sess, human_name, handle, env, max_steps, neighbor_nums=neighbor_nums, memory_size=500000,bias_flag=bias_flag)#80000
    elif algo_name == 'ac':
        model = AC(sess, human_name, handle, env)
    elif algo_name == 'mfac':
        model = MFAC(sess, human_name, handle, env)
    elif algo_name == 'rfac':
        model = RFAC(sess, human_name, handle, env, neighbor_nums=neighbor_nums, bias_flag=bias_flag)

    return model


def spawn_ai_test(algo_name, sess, env, handle, human_name, max_steps, neighbor_nums,bias_flag):
    if algo_name == 'il':
        model = IL(sess, human_name, handle, env, max_steps, memory_size=0)#
    elif algo_name == 'mfq':
        model = MFQ(sess, human_name, handle, env, max_steps, memory_size=0)#
    elif algo_name == 'rfq':
        model = RFQ(sess, human_name, handle, env, max_steps, neighbor_nums=neighbor_nums, memory_size=0,bias_flag=bias_flag)#
    elif algo_name == 'ac':
        model = AC(sess, human_name, handle, env)
    elif algo_name == 'mfac':
        model = MFAC(sess, human_name, handle, env)
    elif algo_name == 'rfac':
        model = RFAC(sess, human_name, handle, env, neighbor_nums=neighbor_nums,bias_flag=bias_flag)

    return model

