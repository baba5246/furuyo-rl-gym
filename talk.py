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

from collections import Counter
from sklearn.feature_extraction import FeatureHasher


class Talk(Env):

    INPUT_MAXLEN = 10
    N_HASHER_FEATURE = 10
    SPEAKER_USER = "user"
    SPEAKER_BOT = "bot"

    def __init__(self):
        # actions and entities
        kb = yaml.load(open("resources/kb.yml", "r", encoding="utf-8"))
        self.actions = kb.get("actions", [])
        self.entities = kb.get("entities", {})
        assert len(self.actions) > 0, "no actions!"
        # context variables
        self.message = ""
        self.reply = ""
        self.utter_count = 0
        self.speaker = Talk.SPEAKER_USER
        self.done_action = len(self.actions) - 1
        self.state = []
        # tokenizer
        self.mt = MeCab.Tagger("-Owakati")
        logs = open("resources/dialogue.log", "r", encoding="utf-8").readlines()
        wakati_logs = [self.mt.parse(l) for l in logs]
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(wakati_logs)
        # hasher
        self.entity_hasher = FeatureHasher(n_features=Talk.N_HASHER_FEATURE)
        # action and observation space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0,
                                            high=len(self.tokenizer.word_index),
                                            shape=(Talk.INPUT_MAXLEN,))
        # simulator
        self.answers = yaml.load(open("resources/answers.yml", "r", encoding="utf-8"))
        self.answers = self.answers["answers"]

    def step(self, action):
        # get reply from actions
        self.reply = self.actions[action]
        print(self.reply)
        # compute reward and terminal
        reward = self.compute_reward(self.message, self.reply)
        done = True if reward < 1.0 or action == self.done_action else False
        # environments if the dialogue continues
        if not done:
            # wait user input
            self.message = input("user: ") if self.reply == "<listen>" else self.reply
            while len(self.message) == 0:
                print("empty message!")
                self.message = input("user: ")
            print("user msg = {0}".format(self.message))
            # create feature vectors
            self.state = self.create_features(
                self.message,
                self.utter_count,
                self.speaker,
                self.entities
            )
            self.utter_count += 1
            print(self.utter_count, reward, done)

        return self.state, reward, done, {}

    def _reset(self):
        # reset context variables
        self.utter_count = 0
        # wait user input
        self.speaker = Talk.SPEAKER_USER
        self.message = input("user: ")
        while len(self.message) == 0:
            print("empty message!")
            self.message = input("user: ")
        print("user msg = {0}".format(self.message))
        # create feature vectors
        self.state = self.create_features(
            self.message,
            self.utter_count,
            self.speaker,
            self.entities
        )
        return self.state

    def compute_reward(self, message, reply):
        answer = self.answers.get(message)
        assert answer, "unexpected user message!"
        reward = 1.0 if reply == answer else 0.0
        return reward

    # TODO: entity extraction
    def extract_entities(self, message):
        pass

    def create_features(self, message, utter_count, speaker, entities):
        features = []
        # create word index sequence
        seq = self.tokenizer.texts_to_sequences([self.mt.parse(message)])
        message_vector = sequence.pad_sequences(seq, maxlen=Talk.INPUT_MAXLEN)[0]
        features.append(message_vector)
        # create context vector
        context_vector = np.zeros(2)
        context_vector[0] = utter_count
        context_vector[1] = 1 if speaker == "user" else 0
        features.append(context_vector)
        # create entity vector
        entity_keys = Counter(list(entities.keys()))
        f = self.entity_hasher.transform([entity_keys])
        entiry_vector = f.toarray()[0]
        features.append(entiry_vector)
        return features

