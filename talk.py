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

import entity_extraction
import functions
import mounter

import logging
logger = logging.getLogger(__name__)
stream_log = logging.StreamHandler()
stream_log.setLevel(logging.DEBUG)
stream_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
logger.addHandler(stream_log)
logger.setLevel(logging.DEBUG)


class Talk(Env):

    INPUT_MAXLEN = 10
    N_HASHER_FEATURE = 10
    N_CONTEXT_FEATURE = 2
    FEATURE_LENGTH = INPUT_MAXLEN + N_HASHER_FEATURE + N_CONTEXT_FEATURE
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
                                            shape=(Talk.FEATURE_LENGTH,))

        # simulator
        self.answers = yaml.load(open("resources/answers.yml", "r", encoding="utf-8"))
        self.answers = self.answers["answers"]

    def step(self, action):

        # get reply from actions
        self.reply = self.actions[action]
        print(self.reply)

        # compute reward and terminal
        reward = self.compute_reward(self.message, self.reply, manual=True)
        done = True if reward < 1.0 or action == self.done_action else False

        # environments if the dialogue continues
        if not done:
            # do action
            if self.reply == "<listen>":
                self.message = input("user: ")
                while len(self.message) == 0:
                    logger.warning("empty message!")
                    self.message = input("user: ")
                self.speaker = Talk.SPEAKER_USER
            elif self.reply.startswith("<") and self.reply.endswith(">"):
                self.entities = functions.run_function(self.reply, self.entities)
                self.message = self.reply
                self.speaker = Talk.SPEAKER_BOT
            else:
                try:
                    self.message = mounter.mount(self.reply, self.entities)
                    self.speaker = Talk.SPEAKER_BOT
                except KeyError as e:
                    self.message = "Error:" + e.message
                    done = True
            logger.info("user msg = {0}".format(self.message))

            # extract entities
            extracted = entity_extraction.extract(self.message)
            if "entity" in extracted and "type" in extracted:
                self.entities[extracted.get("type")] = extracted.get("entity")

            # create feature vectors
            self.state = self.create_features(
                self.message,
                self.utter_count,
                self.speaker,
                self.entities
            )
            self.utter_count += 1
            logger.info([self.utter_count, reward, done])
            logger.debug(self.state)

        return self.state, reward, done, {}

    def _reset(self):
        # reset context variables
        self.utter_count = 0
        self.entities = {}

        # wait user input
        self.speaker = Talk.SPEAKER_USER
        self.message = input("user: ")
        while len(self.message) == 0:
            logger.warning("empty message!")
            self.message = input("user: ")
        logger.info("user msg = {0}".format(self.message))

        # extract entities
        extracted = entity_extraction.extract(self.message)
        if "entity" in extracted and "type" in extracted:
            self.entities[extracted.get("type")] = extracted.get("entity")

        # create feature vectors
        self.state = self.create_features(
            self.message,
            self.utter_count,
            self.speaker,
            self.entities
        )
        logger.debug(self.state)

        return self.state

    # TODO: automatic input
    def wait_user_input(self, reply, manual=False):
        if reply == "どこの天気が知りたいですか？"
            pass

    def compute_reward(self, message, reply, manual=False):
        if manual:
            feedback = input("feedback(correct -> 1 / wrong -> empty): ")
            reward = 1.0 if len(feedback) > 0 else 0.0
        else:
            answer = self.answers.get(message)
            assert answer, "unexpected user message!"
            reward = 1.0 if reply == answer else 0.0
        return reward

    def create_features(self, message, utter_count, speaker, entities):
        features = []
        # create word index sequence
        seq = self.tokenizer.texts_to_sequences([self.mt.parse(message)])
        message_vector = sequence.pad_sequences(seq, maxlen=Talk.INPUT_MAXLEN)[0]
        features.append(message_vector)
        # create context vector
        context_vector = np.zeros(Talk.N_CONTEXT_FEATURE)
        context_vector[0] = utter_count
        context_vector[1] = 1 if speaker == "user" else 0
        features.append(context_vector)
        # create entity vector
        entity_keys = Counter(list(entities.keys()))
        f = self.entity_hasher.transform([entity_keys])
        entiry_vector = f.toarray()[0]
        features.append(entiry_vector)
        features = [a for array in features for a in array]
        return features

