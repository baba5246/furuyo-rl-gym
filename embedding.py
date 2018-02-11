#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from gensim.models import word2vec, KeyedVectors
from keras.layers import Embedding


SHIROYAGI_MODEL_PATH = "resources/w2v/shiroyagi-w2v/word2vec.gensim.model"
INUI_MODEL_PATH = "resources/w2v/inui-e2v/entity_vector.model.bin"


def load_w2v_model(path=SHIROYAGI_MODEL_PATH):
    if path == INUI_MODEL_PATH:
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    else:
        model = word2vec.Word2Vec.load(path)
    return model


def create_embedding_layer(word_index, model, input_length=20):
    num_words = len(word_index)
    output_dim = len(model.wv.syn0[0])
    embedding_matrix = np.zeros((num_words+1, output_dim))
    for word, i in word_index.items():
        if word in model.wv.index2word:
            embedding_matrix[i] = model[word]
    return Embedding(output_dim=output_dim,
                     input_dim=num_words+1,
                     input_length=input_length,
                     weights=[embedding_matrix],
                     trainable=False)
