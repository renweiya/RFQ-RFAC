# RFQ-RFAC-Represented-Value-Function-Approach-for-Large-Scale-Multi-Agent-Reinforcement-Learning
Represented Value Function Approach for Large Scale Multi Agent Reinforcement Learning


A Tensorflow implementation of RFAC and RFQ in the paper [Represented Value Function Approach for Large Scale Multi Agent Reinforcement Learning ](https://arxiv.org/abs/2001.01096).

 This work is based on the code framework in [MFQ-MFAC](https://github.com/mlii/mfrl.git)
 
 ![image](https://github.com/renweiya/RFQ-RFAC/blob/master/2.gif)

## Code structure

- `./examples/`: contains scenarios for Game and models.
- `battle.py`: contains code for running Battle Game with trained model
- `wild_war.py`: contains code for running Wild war Game with trained model
- `train_battle.py`: contains code for training Battle Game models
- `wildwar_ELO_single.py`: contains code for testing Wild war game by Elo Scores among all players
- `battle_ELO_single`: contains code for testing Battle game by Elo Scores among all players
- `data`: pre-trained model
## Compile Ising environment and run

**Requirements**
- `python>=3.6.0`
- `tensorflow>=1.14.0`

## Compile MAgent platform and run

Before running Battle Game environment, you need to compile it. You can get more helps from: [MAgent](https://github.com/geek-ai/MAgent) 

**Steps for compiling**

cd Represented_Value Function_MARL/examples/battle_model
bash build.sh
**Steps for training models under Battle Game settings**

1. cd Represented_Value Function_MARL
export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}

2. Run training script for training (e.g. rfac):

python train_battle.py --algo rfac

3.train your model and change the name of model file form 1999 to 1999A,1999B,...

**Steps for testing models under Battle Game and Wild_war Game**

3.Battle Game
python battle.py --algo mfq --oppo rfac --idx {1999A,1999A}

4.Wild war Game
python wild_war.py --algo mfq --oppo rfac --idx {1999A,1999A}

5.Compute Elo scores
python battle_ELO_single.py
python wildwar_ELO_single.py

6.once you open a terminal,type:
export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
or you can edit the ~/.bashrc file to save time. 

## Paper citation
If you found it helpful, consider citing the following paper:
<pre>
@InProceedings{
  title = 	 {Represented Value Function Approach for Large Scale Multi Agent Reinforcement Learning},
  author = 	 {Weiya Ren},
  booktitle = 	 {arXiv:2001.01096},
  year = 	 {2020},
  address = 	 {China},
  month = 	 {Jan}
}
</pre>
<pre>
@InProceedings{pmlr-v80-yang18d,
  title = 	 {Mean Field Multi-Agent Reinforcement Learning},
  author = 	 {Yang, Yaodong and Luo, Rui and Li, Minne and Zhou, Ming and Zhang, Weinan and Wang, Jun},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {5567--5576},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Stockholmsmässan, Stockholm Sweden},
  month = 	 {10--15 Jul},
  publisher = 	 {PMLR}
}
</pre>
