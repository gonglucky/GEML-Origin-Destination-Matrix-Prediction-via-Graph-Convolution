# -*- coding: utf-8 -*-
'''
@Time    : 2019/11/21 0:47
@Author  : Zekun Cai
@File    : load_data.py
@Software: PyCharm
'''
import numpy as np
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from Param import *


def data_generator(data, temporal_data, batch=BATCHSIZE):
    num_time = data.shape[0] - TIMESTEP
    seed = 0

    while True:

        time_random = np.arange(num_time)
        np.random.seed(seed)
        np.random.shuffle(time_random)
        seed += 1

        trainX, temporal, trainY = [], [], []
        batch_num = 0

        for t in time_random:
            x = np.array(data[t:t + TIMESTEP].todense()).reshape((TIMESTEP, HEIGHT * WIDTH, HEIGHT * WIDTH, 1))
            y = np.array(data[t + TIMESTEP].todense()).reshape((HEIGHT * WIDTH, HEIGHT * WIDTH, 1))
            temp_fea = temporal_data[t + TIMESTEP]
            trainX.append(x), temporal.append(temp_fea), trainY.append(y)
            batch_num += 1

            if batch_num == batch:
                trainX, temporal, trainY = np.array(trainX), np.array(temporal), np.array(trainY)
                yield [trainX, temporal], trainY
                batch_num = 0
                trainX, temporal, trainY = [], [], []


def test_generator(data, temporal_data, batch=BATCHSIZE):
    num_time = data.shape[0] - TIMESTEP
    while True:
        testX, temporal = [], []
        batch_num = 0

        for t in range(num_time):
            x = np.array(data[t:t + TIMESTEP].todense()).reshape((TIMESTEP, HEIGHT * WIDTH, HEIGHT * WIDTH, 1))
            temp_fea = temporal_data[t + TIMESTEP]
            testX.append(x), temporal.append(temp_fea)
            batch_num += 1

            if batch_num == batch:
                testX, temporal = np.array(testX), np.array(temporal)
                yield [testX, temporal]
                batch_num = 0
                testX, temporal = [], []


def get_true(data):
    return data[TIMESTEP:]


def get_true_dense(data):
    num_time = data.shape[0] - TIMESTEP
    testY = np.array(data[TIMESTEP:].todense()).reshape((num_time, HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
    return testY


def data_generator_pixel(data, temporal_data, batch):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL))
    num_time = data.shape[0] - TIMESTEP
    seed = 0

    while True:

        time_random = np.arange(num_time)
        np.random.seed(seed)
        np.random.shuffle(time_random)
        seed += 1

        trainX, temporal, trainY = [], [], []
        batch_num = 0

        for t in time_random:
            for loc in range(HEIGHT * WIDTH):
                x = data[t:t + TIMESTEP, loc]
                y = data[t + TIMESTEP, loc]
                temp_fea = temporal_data[t + TIMESTEP]
                trainX.append(x), temporal.append(temp_fea), trainY.append(y)
                batch_num += 1

                if batch_num == batch:
                    trainX, temporal, trainY = np.array(trainX), np.array(temporal), np.array(trainY)
                    # yield [trainX, temporal], trainY
                    yield trainX, trainY
                    batch_num = 0
                    trainX, temporal, trainY = [], [], []


def test_generator_pixel(data, temporal_data, batch):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL))
    num_time = data.shape[0] - TIMESTEP
    testX, temporal = [], []
    batch_num = 0

    for t in range(num_time):
        for loc in range(HEIGHT * WIDTH):
            x = data[t:t + TIMESTEP, loc]
            temp_fea = temporal_data[t + TIMESTEP]
            testX.append(x), temporal.append(temp_fea)
            batch_num += 1

            if batch_num == batch:
                testX, temporal = np.array(testX), np.array(temporal)
                # yield [testX, temporal]
                yield testX
                batch_num = 0
                testX, temporal = [], []


def data_generator_400(data, temporal_data, batch, return_temp=False, return_pos=False, dual_norm=False,
                       return_multi=False, shape='NTHW'):
    if dual_norm:
        data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
        data /= (MAX_NON_EYE / MAX_DIFFUSION)
        data[:, range(HEIGHT * WIDTH), range(HEIGHT * WIDTH), :] /= (MAX_DIFFUSION / MAX_NON_EYE)
        data = data.reshape(data.shape[0], HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL)
    else:
        data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL))
    num_time = data.shape[0] - TIMESTEP
    seed = 0

    pos_matrix = np.array(range(400))
    pos_matrix = np.eye(400)[pos_matrix]
    pos_matrix = pos_matrix[np.newaxis, :, :].repeat(BATCHSIZE, axis=0)

    while True:

        time_random = np.arange(num_time)
        np.random.seed(seed)
        np.random.shuffle(time_random)
        seed += 1

        trainX, temporal, trainY = [], [], []
        batch_num = 0

        for t in time_random:
            x = data[t:t + TIMESTEP]
            x = x.transpose((1, 0, 2, 3, 4))
            y = data[t + TIMESTEP]
            temp_fea = temporal_data[t + TIMESTEP]
            trainX.append(x), temporal.append(temp_fea), trainY.append(y)
            batch_num += 1

            if batch_num == batch:
                trainX, temporal, trainY = np.array(trainX), np.array(temporal), np.array(trainY)
                if shape == 'TNN':
                    trainX = trainX.transpose((0, 2, 1, 3, 4, 5))
                    trainX = trainX.reshape((BATCHSIZE, TIMESTEP, HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
                    trainY = trainY.reshape((BATCHSIZE, HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
                elif shape == 'THWN':
                    trainX = trainX.transpose((0, 2, 1, 3, 4, 5))
                    trainX = trainX.reshape((BATCHSIZE, TIMESTEP, HEIGHT, WIDTH, HEIGHT * WIDTH, CHANNEL))
                    trainY = trainY.reshape((BATCHSIZE, HEIGHT, WIDTH, HEIGHT * WIDTH, CHANNEL))
                elif shape == 'NTHW':
                    pass

                if return_pos:
                    temporal = temporal[:, np.newaxis, :].repeat(HEIGHT * WIDTH, axis=1)
                    temporal = np.concatenate((temporal, pos_matrix), axis=-1)

                if return_multi:
                    trainY_sum = np.sum(trainY, axis=(2, 3))
                    Y = [trainY, trainY, trainY_sum]
                else:
                    Y = trainY

                if return_temp:
                    yield [trainX, temporal], Y
                else:
                    yield trainX, Y
                batch_num = 0
                trainX, temporal, trainY = [], [], []


def test_generator_400(data, temporal_data, batch, return_temp=False, return_pos=False, dual_norm=False, shape='NTHW'):
    if dual_norm:
        data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
        data /= (MAX_NON_EYE / MAX_DIFFUSION)
        data[:, range(HEIGHT * WIDTH), range(HEIGHT * WIDTH), :] /= (MAX_DIFFUSION / MAX_NON_EYE)
        data = data.reshape(data.shape[0], HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL)
    else:
        data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL))
    num_time = data.shape[0] - TIMESTEP
    testX, temporal = [], []
    batch_num = 0

    pos_matrix = np.array(range(400))
    pos_matrix = np.eye(400)[pos_matrix]
    pos_matrix = pos_matrix[np.newaxis, :, :].repeat(BATCHSIZE, axis=0)

    for t in range(num_time):
        x = data[t:t + TIMESTEP]
        x = x.transpose((1, 0, 2, 3, 4))
        temp_fea = temporal_data[t + TIMESTEP]
        testX.append(x), temporal.append(temp_fea)
        batch_num += 1

        if batch_num == batch:
            testX, temporal = np.array(testX), np.array(temporal)
            if shape == 'TNN':
                testX = testX.transpose((0, 2, 1, 3, 4, 5))
                testX = testX.reshape((BATCHSIZE, TIMESTEP, HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
            elif shape == 'THWN':
                testX = testX.transpose((0, 2, 1, 3, 4, 5))
                testX = testX.reshape((BATCHSIZE, TIMESTEP, HEIGHT, WIDTH, HEIGHT * WIDTH, CHANNEL))
            elif shape == 'NTHW':
                pass

            if return_pos:
                temporal = temporal[:, np.newaxis, :].repeat(HEIGHT * WIDTH, axis=1)
                temporal = np.concatenate((temporal, pos_matrix), axis=-1)
            if return_temp:
                yield [testX, temporal]
            else:
                yield testX
            batch_num = 0
            testX, temporal = [], []


def data_generator_mini(data, temporal_data, batch):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL))
    num_time = data.shape[0] - TIMESTEP
    seed = 100

    while True:

        time_random = np.arange(num_time)
        np.random.seed(seed)
        np.random.shuffle(time_random)
        seed += 1

        trainX, temporal, trainY = [], [], []
        batch_num = 0

        for t in time_random:
            for loc in range(HEIGHT * WIDTH):
                x = data[t:t + TIMESTEP, loc]
                y = data[t + TIMESTEP, loc]
                temp_fea = temporal_data[t + TIMESTEP]
                trainX.append(x), temporal.append(temp_fea), trainY.append(y)
                batch_num += 1

                if batch_num == batch:
                    trainX, temporal, trainY = np.array([trainX]), np.array(temporal), np.array([trainY])
                    # yield [trainX, temporal], trainY
                    yield trainX, trainY
                    batch_num = 0
                    trainX, temporal, trainY = [], [], []


def test_generator_mini(data, temporal_data, batch):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT, WIDTH, CHANNEL))
    num_time = data.shape[0] - TIMESTEP
    testX, temporal = [], []
    batch_num = 0

    for t in range(num_time):
        for loc in range(HEIGHT * WIDTH):
            x = data[t:t + TIMESTEP, loc]
            temp_fea = temporal_data[t + TIMESTEP]
            testX.append(x), temporal.append(temp_fea)
            batch_num += 1

            if batch_num == batch:
                testX, temporal = np.array([testX]), np.array(temporal)
                # yield [testX, temporal]
                yield testX
                batch_num = 0
                testX, temporal = [], []


def data_generator_res(data, temporal_data, batch, step=None, model='ST'):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
    if step is not None:
        data = data[:, :, :, step:step + 1]

    num_time = data.shape[0]
    seed = 0

    while True:

        time_random = np.arange(start, num_time)
        np.random.seed(seed)
        np.random.shuffle(time_random)
        seed += 1

        XC, XP, XT, XD, YS = [], [], [], [], []
        batch_num = 0

        for t in time_random:
            x_c = [data[t - j] for j in depends[0]]
            x_p = [data[t - j] for j in depends[1]]
            x_t = [data[t - j] for j in depends[2]]
            XC.append(np.dstack(x_c))
            XP.append(np.dstack(x_p))
            XT.append(np.dstack(x_t))
            XD.append(temporal_data[t])
            YS.append(data[t])
            batch_num += 1
            if batch_num == batch:
                XC, XP, XT, XD, YS = np.array(XC), np.array(XP), np.array(XT), np.array(XD), np.array(YS)
                if model == 'ST':
                    yield [XC, XP, XT, XD], YS
                elif model == 'MDL_ST':
                    YS_D = np.sum(YS, axis=-2).reshape((YS.shape[0], HEIGHT, WIDTH, -1))
                    yield [XC, XP, XT, XD], [YS_D, YS]
                elif model == 'MDL':
                    XC_D = np.sum(XC, axis=-2).reshape((XC.shape[0], HEIGHT, WIDTH, -1))
                    XP_D = np.sum(XP, axis=-2).reshape((XP.shape[0], HEIGHT, WIDTH, -1))
                    XT_D = np.sum(XT, axis=-2).reshape((XT.shape[0], HEIGHT, WIDTH, -1))
                    YS_D = np.sum(YS, axis=-2).reshape((YS.shape[0], HEIGHT, WIDTH, -1))
                    XC, XP, XT = XC.reshape((XC.shape[0], HEIGHT, WIDTH, -1)), \
                                 XP.reshape((XP.shape[0], HEIGHT, WIDTH, -1)), \
                                 XT.reshape((XT.shape[0], HEIGHT, WIDTH, -1))
                    YS = YS.reshape((YS.shape[0], HEIGHT, WIDTH, -1))
                    yield [XC_D, XP_D, XT_D, XC, XP, XT, XD], np.concatenate((YS_D, YS), axis=-1)

                elif model == 'MDL_NN':
                    XC_D = np.sum(XC, axis=-2).reshape((XC.shape[0], HEIGHT, WIDTH, -1))
                    XP_D = np.sum(XP, axis=-2).reshape((XP.shape[0], HEIGHT, WIDTH, -1))
                    XT_D = np.sum(XT, axis=-2).reshape((XT.shape[0], HEIGHT, WIDTH, -1))
                    YS_D = np.sum(YS, axis=-2).reshape((YS.shape[0], HEIGHT, WIDTH, -1))
                    YS = YS.reshape((YS.shape[0], HEIGHT, WIDTH, -1))
                    # yield [XC_D, XP_D, XT_D, XC, XP, XT, XD], np.concatenate((YS_D, YS), axis=-1)
                    yield [XC_D, XP_D, XT_D, XC, XP, XT, XD], [YS_D, YS, np.zeros(YS_D.shape)]

                batch_num = 0
                XC, XP, XT, XD, YS = [], [], [], [], []


def test_generator_res(data, temporal_data, batch, step=None, model='ST'):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
    if step is not None:
        data = data[:, :, :, step:step + 1]
    num_time = data.shape[0]
    batch_num = 0

    XC, XP, XT, XD, YS = [], [], [], [], []
    for t in np.arange(start, num_time):
        x_c = [data[t - j] for j in depends[0]]
        x_p = [data[t - j] for j in depends[1]]
        x_t = [data[t - j] for j in depends[2]]
        XC.append(np.dstack(x_c))
        XP.append(np.dstack(x_p))
        XT.append(np.dstack(x_t))
        XD.append(temporal_data[t])
        batch_num += 1
        if batch_num == batch:
            XC, XP, XT, XD, YS = np.array(XC), np.array(XP), np.array(XT), np.array(XD), np.array(YS)
            if model == 'ST' or model == 'MDL_ST':
                yield [XC, XP, XT, XD]
            elif model == 'MDL':

                XC_D = np.sum(XC, axis=-2).reshape((XC.shape[0], HEIGHT, WIDTH, -1))
                XP_D = np.sum(XP, axis=-2).reshape((XP.shape[0], HEIGHT, WIDTH, -1))
                XT_D = np.sum(XT, axis=-2).reshape((XT.shape[0], HEIGHT, WIDTH, -1))

                XC, XP, XT = XC.reshape((XC.shape[0], HEIGHT, WIDTH, -1)), \
                             XP.reshape((XP.shape[0], HEIGHT, WIDTH, -1)), \
                             XT.reshape((XT.shape[0], HEIGHT, WIDTH, -1))
                yield [XC_D, XP_D, XT_D, XC, XP, XT, XD]
            elif model == 'MDL_NN':
                XC_D = np.sum(XC, axis=-2).reshape((XC.shape[0], HEIGHT, WIDTH, -1))
                XP_D = np.sum(XP, axis=-2).reshape((XP.shape[0], HEIGHT, WIDTH, -1))
                XT_D = np.sum(XT, axis=-2).reshape((XT.shape[0], HEIGHT, WIDTH, -1))
                yield [XC_D, XP_D, XT_D, XC, XP, XT, XD]
            batch_num = 0
            XC, XP, XT, XD, YS = [], [], [], [], []


def get_true_res(data, step=None, model='ST'):
    if step is not None:
        data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
        data = data[:, :, :, step:step + 1]
        flow_data = np.sum(data, axis=-2).reshape((data.shape[0], HEIGHT, WIDTH, -1))
        data = ss.csr_matrix(data.reshape(data.shape[0], -1))

    if model == 'ST':
        return data[start:]
    elif model == 'MDL' or model == 'MDL_NN':
        return flow_data[start:], data[start:]


def data_generator_GEML(data, temporal_data, batch=BATCHSIZE, step=0, jump=False):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
    data = data[:, :, :, step]
    if jump:
        num_time = data.shape[0] - start_GEML
        skip = p
        len_his = start_GEML
    else:
        num_time = data.shape[0] - TIMESTEP
        skip = 1
        len_his = TIMESTEP
    seed = 0

    adj = np.load(adjFile)
    adj = np.where(adj > L, 0, adj)
    adj = 1 / adj
    adj[adj == np.inf] = 0
    adj = normalize(adj, norm='l1')
    adj[range(adj.shape[0]), range(adj.shape[1])] = 1
    adj = np.array([adj for i in range(TIMESTEP)])
    adj = np.array([adj for i in range(BATCHSIZE)])

    semantic = data + data.transpose((0, 2, 1))
    # semantic[:, range(HEIGHT * WIDTH), range(HEIGHT * WIDTH)] = 0
    for i in range(semantic.shape[0]):
        semantic[i] = normalize(semantic[i], norm='l1')
        semantic[i][range(semantic.shape[1]), range(semantic.shape[2])] = 1

    while True:

        time_random = np.arange(num_time)
        np.random.seed(seed)
        np.random.shuffle(time_random)
        seed += 1

        trainX, temporal, seman, trainY = [], [], [], []
        batch_num = 0
        for t in time_random:
            x = data[t:t + len_his:skip]
            y = data[t + len_his]
            temp_fea = temporal_data[t + len_his]
            se = semantic[t:t + len_his:skip]
            trainX.append(x), temporal.append(temp_fea), seman.append(se), trainY.append(y)
            batch_num += 1

            if batch_num == batch:
                trainX, temporal, seman, trainY = np.array(trainX), np.array(temporal), np.array(seman), np.array(
                    trainY)
                # outflow, inflow = trainY.sum(-1)[:, :, np.newaxis], trainY.sum(-2)[:, :, np.newaxis]
                yield [trainX, temporal, adj, seman], trainY
                batch_num = 0
                trainX, temporal, seman, trainY = [], [], [], []


def test_generator_GEML(data, temporal_data, batch=BATCHSIZE, step=0, jump=False):
    data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
    data = data[:, :, :, step]
    if jump:
        num_time = data.shape[0] - start_GEML
        skip = p
        len_his = start_GEML
    else:
        num_time = data.shape[0] - TIMESTEP
        skip = 1
        len_his = TIMESTEP

    adj = np.load(adjFile)
    adj = np.where(adj > L, 0, adj)
    adj = 1 / adj
    adj[adj == np.inf] = 0
    adj = normalize(adj, norm='l1')
    adj[range(adj.shape[0]), range(adj.shape[1])] += 1
    adj = np.array([adj for i in range(TIMESTEP)])
    adj = np.array([adj for i in range(BATCHSIZE)])

    semantic = data + data.transpose((0, 2, 1))
    # semantic[:, range(HEIGHT * WIDTH), range(HEIGHT * WIDTH)] = 0
    for i in range(semantic.shape[0]):
        semantic[i] = normalize(semantic[i], norm='l1')
        semantic[i][range(semantic.shape[1]), range(semantic.shape[2])] = 1

    while True:
        testX, temporal, seman = [], [], []
        batch_num = 0

        for t in range(num_time):
            x = data[t:t + len_his:skip]
            temp_fea = temporal_data[t + len_his]
            se = semantic[t:t + len_his:skip]
            testX.append(x), temporal.append(temp_fea), seman.append(se)
            batch_num += 1

            if batch_num == batch:
                testX, temporal, seman = np.array(testX), np.array(temporal), np.array(seman)
                yield [testX, temporal, adj, seman]
                batch_num = 0
                testX, temporal, seman = [], [], []


def get_true_GEML(data, step=None, jump=False):
    if jump:
        len_his = start_GEML
    else:
        len_his = TIMESTEP

    if step is not None:
        data = np.array(data.todense()).reshape((data.shape[0], HEIGHT * WIDTH, HEIGHT * WIDTH, CHANNEL))
        data = data[:, :, :, step]
        data = ss.csr_matrix(data.reshape(data.shape[0], -1))
        return data[len_his:]
    else:
        return data[len_his:]


if __name__ == '__main__':
    diffusion_data = ss.load_npz(diffusionFile)
    diffusion_data = diffusion_data / MAX_DIFFUSION
    dayinfo = np.genfromtxt(dayinfoFile, delimiter=',', skip_header=1)
    for item in data_generator_GEML(diffusion_data, dayinfo, batch=1, step=0):
        print(item)
