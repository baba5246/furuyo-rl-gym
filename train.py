#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

from talk import Talk

import model
from keras.optimizers import Adam, RMSprop

from rl.agents.dqn import DQNAgent
#from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory
from policy import MaxBoltzmannQPolicy


if __name__ == "__main__":

    # Get the environment and extract the number of actions.
    env = Talk()
    nb_actions = env.action_space.n

    # Build models
    dqn_model = model.build_lstm_ff(env)

    # Learning parameters
    TRAIN_EACH_STEP =2000
    TRAIN_MAX_STEP = 2000

    # build DQN agent
    memory = SequentialMemory(limit=100, window_length=1)
    policy = MaxBoltzmannQPolicy(eps=.75) # BoltzmannGumbelQPolicy(C=0.10) #  #BoltzmannQPolicy()
    dqn = DQNAgent(model=dqn_model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=100,
                   enable_dueling_network=False,
                   dueling_type='avg',
                   target_model_update=1e-2,
                   policy=policy)
    dqn.compile(RMSprop(lr=0.01), metrics=['mae'])

    # Training
    dir_path = "weights/{0}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(dir_path, exist_ok=True)
    slice = int(TRAIN_MAX_STEP/TRAIN_EACH_STEP)
    for s in range(slice):
        if s > 0:
            dqn.load_weights(dir_path + "/dqn_talk_weights_{slice}.h5f".format(slice=s-1))
        dqn.fit(env, nb_steps=TRAIN_EACH_STEP, visualize=False, verbose=2)
        dqn.save_weights(dir_path + "/dqn_talk_weights_{slice}.h5f".format(slice=s))

    # Evaluating
    dqn.test(env, nb_episodes=10, visualize=False)
