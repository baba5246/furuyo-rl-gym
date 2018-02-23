#!/usr/bin/env python
# -*- coding: utf-8 -*-

from talk import Talk
import model

from keras.optimizers import Adam, RMSprop

from rl.agents.dqn import DQNAgent
#from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory
from policy import MaxBoltzmannQPolicy


import logging
logger = logging.getLogger(__name__)
stream_log = logging.StreamHandler()
stream_log.setLevel(logging.DEBUG)
stream_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
logger.addHandler(stream_log)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    # Get the environment and extract the number of actions.
    env = Talk()
    nb_actions = env.action_space.n

    # Build models
    dqn_model = model.build_lstm_ff(env)

    # Learning parameters
    TRAIN_EACH_STEP = 500
    TRAIN_MAX_STEP = 15000

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

    # evaluating
    slice = int(TRAIN_MAX_STEP/TRAIN_EACH_STEP)
    for s in range(slice):
        logger.info("SLICE = {0}".format(s))
        dqn.load_weights("weights/20180222-211054/dqn_talk_weights_{0}.h5f".format(s))
        dqn.test(env, nb_episodes=10, visualize=False)
        logger.info("")
