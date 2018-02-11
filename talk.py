#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import yaml
import MeCab

from gym import spaces
from gym.core import Env

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


class Talk(Env):

    INPUT_MAXLEN = 20
    CONTEXT_LENGTH = 3

    def __init__(self):
        # load knowledge base
        kb = yaml.load(open("conf/kb.yml", "r", encoding="utf-8"))
        self.actions = kb.get("actions", [])
        assert len(self.actions) > 0, "no actions!"
        self.done_action = len(self.actions) - 1

        # tokenizer
        self.mt = MeCab.Tagger("-Owakati")
        corpus = open("resources/actions.txt", "r", encoding="utf-8").readlines()
        texts = [self.mt.parse(c) for c in corpus]
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(texts)

        # action and observation space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0,
                                            high=len(self.tokenizer.word_index),
                                            shape=(Talk.INPUT_MAXLEN,))
        # contexts
        self.utter_count = 0
        self.states = [np.zeros(self.INPUT_MAXLEN) for _ in range(self.CONTEXT_LENGTH)]

    def step(self, action):
        reply = self.actions[action]
        print(reply)

        # TODO: define rewards
        fbk = input("feedback [1 -> 1, -1 -> empty]: ")
        print("feedback = {0}".format(fbk if len(fbk) > 0 else "empty"))
        if len(fbk) > 0:
            reward = 1.0 * (self.utter_count + 1)
            done = False if action != self.done_action else True
        else:
            reward = 0.0
            done = True

        # TODO: define entities
        if not done:
            msg = input("user: ") if reply == "<listen>" else reply
            while len(msg) == 0:
                print("empty message!")
                msg = input("user: ")
            print("user msg = {0}".format(msg))
            seq = self.tokenizer.texts_to_sequences([self.mt.parse(msg)])
            state = sequence.pad_sequences(seq, maxlen=Talk.INPUT_MAXLEN)[0]
            self.states = [self.states[i+1] if i < self.CONTEXT_LENGTH - 1 else state
                           for i in range(self.CONTEXT_LENGTH)]
            self.utter_count += 1
            print(self.utter_count, self.states, reward, done)

        return self.states, reward, done, {}

    def _reset(self):
        self.utter_count = 0
        msg = "<user_login>"
        print("user msg = {0}".format(msg))
        seq = self.tokenizer.texts_to_sequences(msg)
        state = sequence.pad_sequences(seq, maxlen=Talk.INPUT_MAXLEN)[0]
        self.states = [np.zeros(self.INPUT_MAXLEN) if i > self.utter_count else state
                       for i in reversed(range(self.CONTEXT_LENGTH))]
        print(self.utter_count, self.states)
        return self.states
