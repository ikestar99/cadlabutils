#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


class NullObject:
    """Null class where all attribute and function calls return self."""
    __slots__ = ()

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __bool__(self):
        return False

    def __repr__(self):
        return "Null"


null_object = NullObject()
