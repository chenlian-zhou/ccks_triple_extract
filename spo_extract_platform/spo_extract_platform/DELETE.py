# -*- coding: utf-8 -*-
import os, re, json, traceback
from tqdm import tqdm
import time

lst = ["a", "b", "c", "d"]
bar = tqdm(lst)
for letter, char in zip(lst, bar):

    time.sleep(1)
    bar.set_description("Processing %s" % char)
