#!/usr/bin/env python
# -*- coding: utf-8 -*-

from talk import Talk

from tensorflow import unstack
from keras import backend as K
from keras.layers import Input, Lambda, Dense, LSTM, Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam, RMSprop

import embedding

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory


if __name__ == "__main__":

    # Get the environment and extract the number of actions.
    env = Talk()
    # np.random.seed(123)
    # env.seed(123)
    nb_actions = env.action_space.n

    # Set model parameters
    MAX_VOCABULARY = len(env.tokenizer.word_index)
    EMBEDDING_DIM = 16
    LSTM_OUTPUT_DIM = 32
    DROPOUT_RATE = 0.20

    # Create embedding layers
    w2v = embedding.load_w2v_model()

    # Build models
    input = Input(shape=(1, env.INPUT_MAXLEN))
    unstack = Lambda(lambda x, func=unstack: func(x, axis=1))(input)
    embed = Embedding(MAX_VOCABULARY, EMBEDDING_DIM)(unstack)
    lstm = LSTM(LSTM_OUTPUT_DIM, return_sequences=True)(embed)
    max_pool = GlobalMaxPooling1D()(lstm)
    ff = Dense(LSTM_OUTPUT_DIM)(max_pool)
    preds = Dense(nb_actions, activation='softmax')(ff)
    # unstack_inputs = Lambda(lambda x, func=unstack: func(x, axis=2))(inputs)
    # sequence_inputs = [Lambda(lambda x, func=unstack: func(x, axis=1))(u) for u in unstack_inputs]
    # embeds = [embedding.create_embedding_layer(env.tokenizer.word_index,
    #                                            w2v,
    #                                            input_length=env.INPUT_MAXLEN)(s)
    #           for s in sequence_inputs]
    # lstms = [LSTM(LSTM_OUTPUT_DIM, return_sequences=True)(e) for e in embeds]
    # max_pools = [GlobalMaxPooling1D()(l) for l in lstms]
    # stacked = Lambda(lambda x: K.stack(x, axis=1))(max_pools)
    # context_lstm = LSTM(LSTM_OUTPUT_DIM, dropout=DROPOUT_RATE, return_sequences=True)(stacked)
    # context_max_pool = GlobalMaxPooling1D()(context_lstm)
    # context_ff1 = Dense(LSTM_OUTPUT_DIM)(context_max_pool)
    # context_ff2 = Dense(LSTM_OUTPUT_DIM)(context_ff1)
    # preds = Dense(nb_actions, activation='softmax')(context_ff2)
    historical_lstm = Model(input, preds)
    print(historical_lstm.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100, window_length=1)
    policy = MaxBoltzmannQPolicy(eps=.5) # BoltzmannGumbelQPolicy(C=0.10) #  #BoltzmannQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=historical_lstm,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=100,
                   enable_dueling_network=False,
                   dueling_type='avg',
                   target_model_update=1e-2,
                   policy=policy)
    dqn.compile(Adam(lr=0.01), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=500, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_talk_weights.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
