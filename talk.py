#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import yaml
import MeCab
import random

from gym import spaces
from gym.core import Env

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


class Talk(Env):

    INPUT_MAXLEN = 10
    CONTEXT_LENGTH = 3

    def __init__(self):
        # load knowledge base
        kb = yaml.load(open("resources/kb.yml", "r", encoding="utf-8"))
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

        self.msg = ""
        self.reply = ""
        self.answers = yaml.load(open("resources/answers.yml", "r", encoding="utf-8"))
        self.answers = self.answers["answers"]

    def step(self, action):
        self.reply = self.actions[action]
        print(self.reply)

        reward = self._result(self.msg, self.reply)
        done = True if reward < 1.0 or action == self.done_action else False

        # TODO: define entities
        if not done:
            self.msg = "東京" if self.reply == "<listen>" else self.reply
            while len(self.msg) == 0:
                print("empty message!")
                self.msg = input("user: ")
            print("user msg = {0}".format(self.msg))
            seq = self.tokenizer.texts_to_sequences([self.mt.parse(self.msg)])
            self.state = sequence.pad_sequences(seq, maxlen=Talk.INPUT_MAXLEN)[0]
            # self.states = [self.states[i+1] if i < self.CONTEXT_LENGTH - 1 else state
            #                for i in range(self.CONTEXT_LENGTH)]
            self.utter_count += 1
            print(self.utter_count, self.state, reward, done)

        return self.state, reward, done, {}

    def _reset(self):
        self.utter_count = 0
        # self.msg = input("user: ")
        self.msg = np.random.choice(["天気教えて", "東京の天気教えて"])
        print("user msg = {0}".format(self.msg))
        seq = self.tokenizer.texts_to_sequences([self.mt.parse(self.msg)])
        self.state = sequence.pad_sequences(seq, maxlen=Talk.INPUT_MAXLEN)[0]
        # self.states = [np.zeros(self.INPUT_MAXLEN) if i > self.utter_count else state
        #                for i in reversed(range(self.CONTEXT_LENGTH))]
        print(self.utter_count, self.state)
        return self.state # self.states

    def _result(self, msg, reply):
        ans = self.answers.get(msg)
        assert ans, "unexpected user message!"

        reward = 1.0 if reply == ans else 0.0
        return reward
