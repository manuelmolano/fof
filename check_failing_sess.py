#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:33:08 2021

@author: manuel
"""

import numpy as np
import glob
import os


def get_bhv_folder(bhv_f):
    assert len(bhv_f) > 0
    if len(bhv_f) > 1:
        oks = []
        for f in bhv_f:
            oks.append(os.path.exists(f+'/sessions'))
        assert np.sum(oks) == 1
        bhv_f = np.array(bhv_f)[np.where(np.array(oks))[0]]
    return bhv_f[0]


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
    rat_num = f[f.find('/LE')+3:f.rfind('/LE')]
    bhv_f = glob.glob(behav_folder+'*'+str(rat_num))
    bhv_f = get_bhv_folder(bhv_f)
    print(bhv_f)
    dt_indx = f.find(rat_num+'_20')+len(rat_num)+1
    date = f[dt_indx:dt_indx+10]
    b_f = [f for f in glob.glob(bhv_f+'/*') if f.find(date) != -1]
    print(glob.glob(b_f+'/*.csv'))


# N.C.
print('==============================')
print('NAN')
nan = [s for s, c in zip(inv['session'], inv['state']) if c == 'nan']
print(len(nan))
for f in nan:
    print('---------------')
    print(f)
    rat_num = f[f.find('/LE')+3:f.rfind('/LE')]
    bhv_f = glob.glob(behav_folder+'*'+str(rat_num))
    bhv_f = get_bhv_folder(bhv_f)

    print(glob.glob(bhv_f+'/*.csv'))
