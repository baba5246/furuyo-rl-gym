#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml

from gym import spaces
from gym.core import Env

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


class Talk(Env):

    INPUT_MAXLEN = 5
    INPUT_MAXVOC = 1000

    def __init__(self):
        # load knowledge base
        kb = yaml.load(open("conf/kb.yml", "r", encoding="utf-8"))
        self.actions = kb.get("actions", [])
        assert len(self.actions) > 0, "no actions!"
        print(type(kb))

        # TODO: corpus
        hiragana = [chr(i) for i in range(12353, 12436)] + ["？"]
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(hiragana)

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0,
                                            high=len(self.tokenizer.word_index),
                                            shape=(Talk.INPUT_MAXLEN,))

    def step(self, action):
        reply = self.actions[action]
        print(action)
        is_api_call = reply.startswith("<")
        print("{0} ({1})".format(reply, is_api_call))

        # TODO: define rewards
        fbk = input("feedback [1 -> 1, -1 -> empty]: ")
        print("feedback = {0}".format(fbk if len(fbk) > 0 else "empty"))
        if len(fbk) > 0:
            reward = 1.0
            done = False
        else:
            reward = 0.0
            done = True

        # TODO: define entities
        if not done:
            msg = reply if is_api_call else input("user: ")
            while len(msg) == 0:
                print("empty message!")
                msg = input("user: ")
            print("user msg = {0}".format(msg))
            seq = self.tokenizer.texts_to_sequences(msg)
            state = sequence.pad_sequences(seq, maxlen=Talk.INPUT_MAXLEN)[0]
        else:
            msg = "<user_login>"
            seq = self.tokenizer.texts_to_sequences(msg)
            state = sequence.pad_sequences(seq, maxlen=Talk.INPUT_MAXLEN)[0]

        return state, reward, done, {}

    def _reset(self):
        msg = "<user_login>"
        print("user msg = {0}".format(msg))
        seq = self.tokenizer.texts_to_sequences(msg)
        state = sequence.pad_sequences(seq, maxlen=Talk.INPUT_MAXLEN)[0]
        return state
