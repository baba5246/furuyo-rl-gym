#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import functions
import entity_extraction
import mounter


class TestFunctionsModule(unittest.TestCase):

    def test_get_address_from_db(self):
        function_name = "<get_address_from_db>"
        entities1 = {"place_name": "東京"}
        entities2 = {"place_name": "名古屋"}
        entities3 = {}
        # case 1:
        actual = functions.run_function(function_name, entities1)
        expected = {"place_name": "東京", "address": "東京都中央区"}
        self.assertDictEqual(expected, actual)
        # case 2:
        actual = functions.run_function(function_name, entities2)
        expected = {"place_name": "名古屋"}
        self.assertDictEqual(expected, actual)
        # case 3:
        actual = functions.run_function(function_name, entities3)
        expected = {}
        self.assertDictEqual(expected, actual)

    def test_get_weather_from_db(self):
        function_name = "<get_weather_from_db>"
        entities1 = {"place_name": "東京", "address": "東京都中央区"}
        entities2 = {"place_name": "名古屋"}
        entities3 = {}
        # case 1:
        actual = functions.run_function(function_name, entities1)
        expected = {"place_name": "東京",
                    "address": "東京都中央区",
                    "weather": "晴れ",
                    "time": "recent"}
        self.assertDictEqual(expected, actual)
        # case 2:
        actual = functions.run_function(function_name, entities2)
        expected = {"place_name": "名古屋"}
        self.assertDictEqual(expected, actual)
        # case 3:
        actual = functions.run_function(function_name, entities3)
        expected = {}
        self.assertDictEqual(expected, actual)

    def test_get_weather_from_api(self):
        function_name = "<get_weather_from_api>"
        entities1 = {"place_name": "東京",
                     "address": "東京都中央区",
                     "weather": "晴れ",
                     "time": "recent"}
        entities2 = {"place_name": "大阪",
                     "address": "大阪市北区",
                     "weather": "雨",
                     "time": "not recent"}
        entities3 = {}
        # case 1:
        actual = functions.run_function(function_name, entities1)
        expected = {"place_name": "東京",
                    "address": "東京都中央区",
                    "weather": "晴れ", "time": "recent"}
        self.assertDictEqual(expected, actual)
        # case 2:
        actual = functions.run_function(function_name, entities2)
        expected = {"place_name": "大阪",
                    "address": "大阪市北区",
                    "weather": "晴れ", "time": "recent"}
        self.assertDictEqual(expected, actual)
        # case 3:
        actual = functions.run_function(function_name, entities3)
        expected = {}
        self.assertDictEqual(expected, actual)


class TestEntityExtractionModule(unittest.TestCase):

    def test_extract(self):
        message1 = "東京"
        message2 = "大阪でお願いします"
        message3 = "名古屋で"
        message4 = None
        # case 1:
        actual = entity_extraction.extract(message1)
        expected = {"entity": "東京", "type": "place_name"}
        self.assertDictEqual(expected, actual)
        # case 2:
        actual = entity_extraction.extract(message2)
        expected = {"entity": "大阪", "type": "place_name"}
        self.assertDictEqual(expected, actual)
        # case 2:
        actual = entity_extraction.extract(message3)
        expected = {}
        self.assertDictEqual(expected, actual)
        # case 2:
        actual = entity_extraction.extract(message4)
        expected = {}
        self.assertDictEqual(expected, actual)


class TestMounterModule(unittest.TestCase):

    def test_mount(self):
        message1 = "\{address\}でよろしいですか？"
        entities1_1 = {"place_name": "東京", "address": "東京都中央区"}
        entities1_2 = {"place_name": "東京"}
        message2 = "\{address\}は\{weather\}です"
        entities2 = {"place_name": "東京",
                     "address": "東京都中央区",
                     "weather": "晴れ",
                     "time": "recent"}
        # case 1:
        actual = mounter.mount(message1, entities1_1)
        expected = "東京都中央区でよろしいですか？"
        self.assertEqual(expected, actual)
        # case 2:
        with self.assertRaises(KeyError):
            mounter.mount(message1, entities1_2)
        # case 3:
        actual = mounter.mount(message2, entities2)
        expected = "東京都中央区は晴れです"
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
