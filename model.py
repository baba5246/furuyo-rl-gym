#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow import unstack
from keras.layers import Input, Lambda, Dense, LSTM, Embedding, concatenate, Bidirectional
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model


def build_lstm_ff(env):

    nb_actions = env.action_space.n
    MAX_VOCABULARY = len(env.tokenizer.word_index) + 1
    EMBEDDING_DIM = 16
    LSTM_OUTPUT_DIM = 32

    # Create embedding layers
    # w2v = embedding.load_w2v_model(embedding.INUI_MODEL_PATH)
    # w2v_embedding_layer = embedding.create_embedding_layer(env.tokenizer.word_index, w2v, env.FEATURE_LENGTH)

    inputs = Input(shape=(1, 2, env.FEATURE_LENGTH))
    unstack_inputs = Lambda(lambda x, func=unstack: func(x, axis=2))(inputs)
    sequence_input = Lambda(lambda x, func=unstack: func(x, axis=1))(unstack_inputs[0])
    context_input = Lambda(lambda x, func=unstack: func(x, axis=1))(unstack_inputs[1])
    embed1 = Embedding(MAX_VOCABULARY, EMBEDDING_DIM, input_length=env.FEATURE_LENGTH)(sequence_input)
    lstm1 = Bidirectional(LSTM(LSTM_OUTPUT_DIM, return_sequences=True))(embed1)
    max_pool1 = GlobalMaxPooling1D()(lstm1)
    merged = concatenate([max_pool1, context_input])
    preds = Dense(nb_actions, activation='softmax')(merged)
    model = Model(inputs, preds)
    print(model.summary())
    return model
