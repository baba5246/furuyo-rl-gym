#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def message(pre_message, reply, entities):
    if not pre_message and not reply and len(entities.keys()) == 0:
        choices = [
            "天気教えて",
            "東京の天気を教えて",
            "大阪の天気は？"
        ]
    elif pre_message == "どこの天気が知りたいですか？" and reply == "<listen>":
        choices = [
            "東京",
            "東京の天気",
            "大阪",
            "大阪の天気"
        ]
    elif "でよろしいですか？" in pre_message and reply == "<listen>":
        choices = [
            "はい",
            "いいえ",
            "OK",
            "ちがう",
            "他のところがいい"
        ]
    elif pre_message == "別の言い方で再度入力してください" and reply == "<listen>":
        choices = [
            "東京",
            "東京の天気",
            "大阪",
            "大阪の天気"
        ]
    else:
        choices =[
            ""
        ]
    return np.random.choice(choices)


def reward(message, reply, entities):
    scenario = {
        "天気教えて": "どこの天気が知りたいですか？",
        "東京の天気を教えて": "<get_address_from_db>",
        "大阪の天気を教えて": "<get_address_from_db>",
        "どこの天気が知りたいですか？": "<listen>",
        "東京": "ありがとうございます",
        "東京の天気": "ありがとうございます",
        "大阪": "ありがとうございます",
        "大阪の天気": "ありがとうございます",
        "ありがとうございます": "<get_address_from_db>",
        "<get_address_from_db>": "\{address\}でよろしいですか？",
        "でよろしいですか？": "<listen>",
        "はい": "<get_weather_from_db>",
        "OK": "<get_weather_from_db>",
        "いいえ": "別の言い方で再度入力してください",
        "ちがう": "別の言い方で再度入力してください",
        "他のところがいい": "別の言い方で再度入力してください",
        "別の言い方で再度入力してください": "<listen>",
        "<get_weather_from_db>": "<get_weather_from_api>",
        "<get_weather_from_api>": "\{address\}は\{weather\}です"
    }
    answer_key = [k for k in scenario if message.endswith(k)]
    reward = 1.0 if len(answer_key) > 0 and scenario[answer_key[0]] == reply else 0.0
    return reward
