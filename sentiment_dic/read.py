# -*- coding: utf-8 -*-

import codecs
file = codecs.open('word_polarity.txt')
nums = []
for l in file.readlines():
    strs = l.split(' ')
    nums.append(float(strs[1]))
