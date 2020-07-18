# -*- coding: utf-8 -*-
import os, re, json, traceback

with open("dev.json", "r", encoding="utf-8") as f:
    content = [_.strip() for _ in f.readlines()]

f = open("cluener.dev", "w", encoding="utf-8")

for line in content:
    sample = json.loads(line)
    text = sample["text"]
    tags = ['O'] * len(text)
    for label, label_dict in sample["label"].items():
        for key, val in label_dict.items():
            start_index = val[0][0]
            tags[start_index] = 'B-%s' % label
            end_index = val[0][1]
            for i in range(start_index+1, end_index+1):
                tags[i] = 'I-%s' % label

    for char, tag in zip(text, tags):
        f.write(char+' '+tag+'\n')

    f.write('\n')
