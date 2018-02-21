#!/usr/bin/env python
# -*- coding: utf-8 -*-


def mount(message, entities):
    format = message.replace("\\", "")
    mounted = format.format(**entities)
    return mounted
