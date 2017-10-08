# -*- coding: utf-8 -*-
import numpy as np
import gym, gym.spaces
import csv,h5py,argparse,os, math

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Average, Conv2D, Reshape, Input, Concatenate
from keras.optimizers import Adam
from rl.random import OrnsteinUhlenbeckProcess

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.core import Processor
from keras import regularizers
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.20))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument("--nCells","-n",dest="nCells"  ,type=int,default=10)
parser.add_argument("--learnRate","-r",dest="learnRate"  ,type=float,default=1e-3)
parser.add_argument("--nStepsPerEpisode","-e",dest="nStepsPerEpisode"  ,type=int,default=5)
parser.add_argument("--saveFolder","-s",dest="saveFolder"  ,type=str,default="models/save")
parser.add_argument("--reload","-l",dest="reload"  , type=str,default=None)
parser.add_argument("--zdim","-z",dest="zdim"  , type=int,default=5)
parser.add_argument("--randomSeed",dest="randomSeed"  , type=int,default=1234)
parser.add_argument("--verbose",dest="verbose", action="store_true", default=False)
args = parser.parse_args()
nCells = args.nCells

if not os.path.exists(args.saveFolder):
    os.makedirs(args.saveFolder)


class CircleProcessor(Processor):
    def process_action(self, action):
        return np.clip(action, 0., 1.)

class Environment(gym.Env):
    def __init__(self, nCells, nStepsPerEpisode, verbose=False):
        self.nCells = nCells
        self.nStepsPerEpisode = nStepsPerEpisode
        self.verbose = verbose
        self._reset()
        return

    def _step(self,action): #buy, keep, sell
        # 基本的に、アクションとして出てきた出力を、次の入力として返すだけ
        self.nSteps += 1

        state = action.copy()
        state[:len(self.inX)] = self.inX
        done = True if self.nSteps >= self.nStepsPerEpisode else False

        # rewardは、どれだけ答えと近かったか
        reward = 0.
        if done:
            reward = 1./(1.+np.mean(np.square(state[self.inX.size:self.inX.size+self.outY.size] - self.outY)))

        if math.isnan(reward):
            done = True
            reward = -1. # 発散させたら駄目

        if self.verbose and self.nSteps==self.nStepsPerEpisode:
            print
            print self.nSteps,reward,state

        return state, reward, done, {}

    def _reset(self):
        self.nSteps = 0
        inX, outY = self.yieldOne()
        self.inX, self.outY  = inX, outY
        state = np.zeros(self.getTotalCells())
        state[:len(self.inX)] = self.inX[:]

        return state
    
    def getTotalCells(self):
        return self.nCells+self.inX.size+self.outY.size
    
    def getCellsByName(self,name):
        if name=="inX_num" : return self.inX.size
        if name=="inX_idx1": return 0
        if name=="inX_idx2": return self.inX.size
        if name=="outY_num" : return self.outY.size
        if name=="outY_idx1": return self.inX.size
        if name=="outY_idx2": return self.inX.size + self.outY.size
        if name=="cells_num" : return self.nCells
        if name=="cells_idx1": return self.inX.size + self.outY.size
        if name=="cells_idx2": return self.inX.size + self.outY.size + self.nCells

    def yieldOne(self):
        # XORを作ってみる
        inX  = np.random.randint(0,1+1,2)
        outY = inX[0] ^ inX[1]
        outY = np.array(outY)
        return inX, outY

# Get the environment and extract the number of actions.
env = Environment(args.nCells,args.nStepsPerEpisode, args.verbose)
np.random.seed(args.randomSeed)
env.seed(args.randomSeed)
nb_actions = nTotalCells = env.getTotalCells()

# Next, we build a very simple model.
l2regul = regularizers.l2(1e-2)

zdim = args.zdim
h_input = Input(shape=(1,nTotalCells)) # なぜか、こういう入力次元数らしい
h = h_input
h = Flatten()(h)
h = Dense(nTotalCells,kernel_regularizer=l2regul)(h)
h = Reshape((1,nTotalCells,1))(h)
h = Conv2D(zdim,kernel_size=(1,1),kernel_regularizer=l2regul)(h)
h = Activation('relu')(h)
h = Conv2D(1,kernel_size=(1,1),kernel_regularizer=l2regul)(h)
h = Activation('sigmoid')(h)
h = Flatten()(h) # 出力は1次元
h_output = h

modelA = Model(inputs=h_input, outputs=h_output)
print(modelA.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
h_act = action_input # こっちはactorの出力

observation_input = Input(shape=(1,nTotalCells), name='observation_input')
h_obs = Flatten()(observation_input) # こっちはactorの入力

h = Concatenate()([h_act,h_obs])
#h = Reshape((1,nTotalCells+nTotalCells,1))(h)
#h = Conv2D(zdim,kernel_size=(1,1))(h)
#h = Activation('relu')(h)
#h = Conv2D(zdim,kernel_size=(1,1))(h)
#h = Flatten()(h)
h = Dense(16,kernel_regularizer=l2regul)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Dense(16,kernel_regularizer=l2regul)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Dense(1,kernel_regularizer=l2regul)(h)
h_output = h

modelC = Model(inputs=[action_input,observation_input], outputs=h_output)
print(modelC.summary())

memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=nb_actions)
agent = DDPGAgent(nb_actions=nb_actions,actor=modelA,critic=modelC,critic_action_input=action_input,
                  memory=memory,nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  batch_size=64)#,processor=CircleProcessor)
agent.compile(Adam(lr=args.learnRate, clipnorm=1.), metrics=['mae'])

if args.reload:
    agent.load_weights(args.reload)

import rl.callbacks
class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self,size_inX,size_outY,size_cells):
        self.ofile = open(os.path.join(args.saveFolder,"log.csv"),"w")
        self.cfile = csv.writer(self.ofile)
        self.cfile.writerow(["episode","reward"] + ["inX%d"%i for i in range(size_inX)] + ["outY%d"%i for i in range(size_outY)] + ["cell%d"%i for i in range(size_cells)])
        return

    def on_step_end(self, step, logs):
        self.action  = logs["action"]
        self.observation  = logs["observation"]
        self.log = logs
        return

    def on_episode_end(self, episode, logs):
        action = self.action
        observation = self.observation
        reward = logs['episode_reward']
        self.cfile.writerow([episode,reward] + observation.tolist())
        self.ofile.flush()
        return

class EpisodeTester(rl.callbacks.Callback):
    def __init__(self,size_inX,size_outY,size_cells,freq=100):
        self.size_inX  = size_inX
        self.size_outY = size_outY
        self.size_cells = size_cells
        self.freq = freq
        return

    def on_episode_end(self, episode, logs):
        if not episode % self.freq == 0:
            return
        res = agent.test(env, nb_episodes=5, visualize=False)
        print res
        return

cb_ep = EpisodeLogger(env.getCellsByName("inX_num"),env.getCellsByName("outY_num"),env.getCellsByName("cells_num"))
cb_et = EpisodeTester(env.getCellsByName("inX_num"),env.getCellsByName("outY_num"),env.getCellsByName("cells_num"), freq=10)
#agent.test(env, nb_episodes=10, visualize=False, callbacks=[cb_ep])

agent.fit(env, nb_steps=1000000000, visualize=False, verbose=1, log_interval=1000, callbacks=[cb_ep])
agent.save_weights(os.path.join(args.saveFolder,"weights.h5f"), overwrite=True)
agent.test(env, nb_episodes=100, visualize=False)
