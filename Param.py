# -*- coding: utf-8 -*-
'''
@Time    : 2019/11/19 0:45
@Author  : Zekun Cai
@File    : Param.py
@Software: PyCharm
'''

MAX_DIFFUSION = ****  # all
HEIGHT = 20
WIDTH = 20
CHANNEL = 6
# ENCODER_DIM = 32
embed_k = 128
day_fea = 33
INTERVAL = 60
DAYSTEP = int(24 * 60 / INTERVAL)
WEEKTIMESTEP = int(7 * 24 * 60 / INTERVAL)
dataPath = '../../../data/yahoo_bousai_osaka/'
diffusionFile = dataPath + 'all_60min_6.npz'
dayinfoFile = dataPath + 'day_info_1h.csv'
adjFile = dataPath + 'adjacent.npy'

interval_p, interval_t = 1, 7
len_c, len_p, len_t = 3, 1, 1
depends = [range(1, len_c + 1),
           [interval_p * DAYSTEP * i for i in range(1, len_p + 1)],
           [interval_t * DAYSTEP * i for i in range(1, len_t + 1)]]
start = max(depends[-1])
residual = 2
weight_node = 1
weight_edge = 100
weight_mdl = 0.0001

L = 0
ENCODER_DIM = 400
p = DAYSTEP
start_GEML = 6 * p

TIMESTEP = 6
FUT_TIMESTEP = 1
trainRatio = 0.8
SPLIT = 0.2
LOSS = 'mse'
OPTIMIZER = 'adam'
LEARN = 0.001
BATCHSIZE = 1
EPOCH = 200
