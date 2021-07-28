#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:16:43 2021

@author: manuel
"""

import glob
import os


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    sfx = [int(x[-6:]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


spks_sorting_folder = '/archive/lbektic/AfterClustering/'
electro_folder = '/archive/rat/electrophysiology_recordings/'
behav_folder = '/archive/rat/behavioral_data/'
rats = glob.glob(spks_sorting_folder+'LE*')
for r in rats:
    print('---------------')
    rat_num = r[r.find('/LE')+3:]
    print('Rat LE'+rat_num)
    e_fs = glob.glob(electro_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
    print('Number of electro sessions:', str(len(e_fs)))
    b_fs = glob.glob(behav_folder+'*'+str(rat_num)+'/sessions/*'+str(rat_num)+'*')
    for e_f in e_fs:
        # '/archive/rat/electrophysiology_recordings/LE77/LE77_2020-11-20_09-00-26'
        dt_indx = e_f.find(rat_num+'_20')+len(rat_num)+1
        date = e_f[dt_indx:dt_indx+10]
        date = date.replace('-', '')
        b_f = [f for f in b_fs if f.find(date) != -1]
        if len(b_f) == 0:
            print('---')
            print(date+' behavioral file not found')
        elif len(b_f) > 1:
            print('---')
            print(date+': several behavioral files found')
            print(b_f)
            sorted_files = order_by_sufix(file_list=b_f)
            print('Files:')
            print(sorted_files)
            print('Used file: ', sorted_files[-1])
