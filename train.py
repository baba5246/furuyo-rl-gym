#!/usr/bin/env python
# -*- coding: utf-8 -*-

from talk import Talk

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


if __name__ == "__main__":

    # Get the environment and extract the number of actions.
    env = Talk()
    # np.random.seed(123)
    # env.seed(123)
    nb_actions = env.action_space.n

    # model parameters
    max_features = len(env.tokenizer.word_index)
    embedding_dims = 10
    input_length = env.INPUT_MAXLEN

    # Next, we build a very simple model regardless of the dueling architecture
    # if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
    # Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
    # TODO: change models
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Flatten())
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100, window_length=1)
    policy = BoltzmannQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   enable_dueling_network=False, dueling_type='avg', target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=100, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_talk_weights.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=False)
