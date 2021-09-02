#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:33:08 2021

@author: manuel
"""

import numpy as np
import glob
inv = np.load('/home/manuel/fof_data/sess_inv_extended.npz', allow_pickle=1)

# NO TTTLS
print('==============================')
print('NO TTLS')
no_ttls = [s for s, c in zip(inv['session'], inv['state']) if c == 'no_ttls']

for f in no_ttls:
    print('---------------')
    print(glob.glob(f+'/*'))

# NO BEHAVIOR
print('==============================')
print('NO BEHAVIOR')
no_ttls = [s for s, c in zip(inv['session'], inv['state']) if c == 'no_behavior']

for f in no_ttls:
    print('---------------')
    print(glob.glob(f+'/*'))
