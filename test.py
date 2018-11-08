#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jingnan Jia
@contact: jiajingnan2222@gmail.com
@file: test.py.py
@time: 18-11-7 07 上午10:56
'''
import os
import input_
import numpy as np
images = np.random.uniform(size = (1,3,3,2))

b = (np.random.uniform(size=images.shape) < images).astype(np.float32)
print(images)
print(b)