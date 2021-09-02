#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:33:08 2021

@author: manuel
"""

import numpy as np
import glob
import inventory
behav_folder = '/archive/rat/behavioral_data/'
inv = np.load('/home/molano/fof_data/sess_inv_sbsFalse.npz', allow_pickle=1)

# NO TTTLS
print('==============================')
print('NO TTLS')
no_ttls = [s for s, c in zip(inv['session'], inv['state']) if c == 'no_ttls']
print(len(no_ttls))
for f in no_ttls:
    print('---------------')
    print(f)
    print(glob.glob(f+'/*.dat'))

# NO BEHAVIOR
print('==============================')
print('NO BEHAVIOR')
no_behav = [s for s, c in zip(inv['session'], inv['state'])
            if c == 'no_behavior']
print(len(no_behav))
for f in no_behav:
    print('---------------')
    print(f)
    rat_num = f[f.find('/LE')+3:]
    bhv_f = glob.glob(behav_folder+'*'+str(rat_num))
    # check that the behav folder exists
    bhv_f = inventory.get_bhv_folder(bhv_f)

    print(glob.glob(bhv_f+'/*.csv'))


# N.C.
print('==============================')
print('NAN')
nan = [s for s, c in zip(inv['session'], inv['state']) if c == 'nan']
print(len(nan))
for f in nan:
    print('---------------')
    print(f)
    rat_num = f[f.find('/LE')+3:]
    bhv_f = glob.glob(behav_folder+'*'+str(rat_num))
    # check that the behav folder exists
    bhv_f = inventory.get_bhv_folder(bhv_f)

    print(glob.glob(bhv_f+'/*.csv'))
