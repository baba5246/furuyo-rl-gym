#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def extract(message):
    if not message or (type(message) != np.str_ and type(message) != str):
        return {}
    elif "東京" in message:
        return {"entity": "東京", "type": "place_name"}
    elif "大阪" in message:
        return {"entity": "大阪", "type": "place_name"}
    else:
        return {}