#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_address_from_db(entities):
    results = {}
    db = {
        "東京": "東京都中央区",
        "大阪": "大阪市北区"
    }
    place_name = entities.get("place_name", "")
    address = db.get(place_name)
    if address:
        results = {"address": address}
    return results


def get_weather_from_db(entities):
    results = {}
    db = [
        {"東京都中央区": "晴れ", "time": "recent"},
        {"大阪市北区": "雨", "time": "not recent"}
    ]
    address = entities.get("address", "")
    weathers = [d for d in db if d.get(address)]
    if len(weathers) > 0:
        w = weathers[0]
        results = {"weather": w[address], "time": w["time"]}
    return results


def get_weather_from_api(entities):
    results = {}
    time = entities.get("time")
    if time and time == "not recent":
        results = {"weather": "晴れ", "time": "recent"}
    return results


FUNCTIONS = {
    "<get_address_from_db>": get_address_from_db,
    "<get_weather_from_db>": get_weather_from_db,
    "<get_weather_from_api>": get_weather_from_api,
}


def run_function(name, entities):
    # run function
    func = FUNCTIONS.get(name)
    assert func, "invalid function name!"
    result_dict = func(entities)
    # save entities
    entities.update(result_dict)
    return entities


