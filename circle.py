# -*- coding: utf-8 -*-
import numpy as np
import gym, gym.spaces
import csv,h5py,argparse,os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Average, Conv2D
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from keras import regularizers
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.20))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument("--nCells","-n",dest="nCells"  ,type=int,default=10)
parser.add_argument("--learnRate",dest="learnRate"  ,type=float,default=1e-3)
parser.add_argument("--nStepsPerEpisode","-n",dest="nStepsPerEpisode"  ,type=int,default=20)
parser.add_argument("--saveFolder","-s",dest="saveFolder"  ,type=str,default="models/save")
parser.add_argument("--reload","-r",dest="reload"  , type=str,default=None)
args = parser.parse_args()
nCells = args.nCells

if not os.path.exists(args.saveFolder):
    os.makedirs(args.saveFolder)

class Environment(gym.Env):
    def __init__(self, nCells):
        self.nCells = nCells
        self._reset()
        return

    def _step(self,action): #buy, keep, sell
        # 基本的に、アクションとして出てきた出力を、次の入力として返すだけ
        state = action
        state[:len(self.inX)] = self.inX
        done = False

        # rewardは、どれだけ答えと近かったか
        reward = 1./np.mean(np.abs(state[len(state.inX):len(state.inX)+len(state.outY)] - self.outY))

        return state, reward, done, {}

    def _reset(self):
        inX, outY = self.yieldOne()
        self.inX, self.outY  = inX, outY
        state = np.zeros(self.getTotalCells())
        state[:len(self.inX)] = self.inX

        return state
    
    def getTotalCells(self):
        return self.nCells+len(inX)+len(outY)

    def yieldOne(self):
        # XORを作ってみる
        inX  = np.random.randint(0,1+1,2)
        outY = inX[0] ^ inX[1]
        outY = np.array(outY)
        return inX, outY

regul = 1e-8
# Get the environment and extract the number of actions.
env = Environment()
np.random.seed(123)
env.seed(123)
nb_actions = nTotalCells = env.getTotalCells()

# Next, we build a very simple model.
zdim = 5
modelA = Sequential()
modelA.add(Dense(nTotalCells,input_dim=nTotalCells))
modelA.add(Reshape((1,nTotalCells,1)))
modelA.add(Conv2D(zdim,kernel_size=(1,1)))
modelA.add(Activation('relu'))
modelA.add(Conv2D(zdim,kernel_size=(1,1)))
print(modelA.summary())

modelC = Sequential()
modelC.add(Reshape((1,nTotalCells,1),input_shape=(nTotalCells)))
modelC.add(Conv2D(zdim,kernel_size=(1,1)))
modelC.add(Activation('relu'))
modelC.add(Conv2D(zdim,kernel_size=(1,1)))
modelC.add(Average())
print(modelC.summary())

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
agent = DDPGAgent(nb_actions,modelA,modelC,nb_actions, memory)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

if args.reload:
    agent.load_weights(args.reload)

import rl.callbacks

agent.fit(env, nb_steps=1000000000, visualize=False, verbose=2, callbacks=[])
