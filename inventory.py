#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:16:43 2021

@author: manuel
"""

import glob
import os
import utils


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    sfx = [int(x[-6:]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


def inventory(s_rate=3e4, s_rate_eff=2e3):
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
        b_f = glob.glob(behav_folder+'*'+str(rat_num))
        assert len(b_f) == 1
        path, name = os.path.split(b_f[0])
        p = utils.get_behavior(main_folder=path, subject=name)
        for e_f in e_fs:
            dt_indx = e_f.find(rat_num+'_20')+len(rat_num)+1
            date = e_f[dt_indx:dt_indx+10]
            date = date.replace('-', '')
            b_f = [f for f in p.available if f.find(date) != -1]
            if len(b_f) == 0:
                print('---')
                print(date+' behavioral file not found')
                continue
            elif len(b_f) > 1:
                print('---')
                print(date+': several behavioral files found')
                print(b_f)
                sorted_files = order_by_sufix(file_list=b_f)
                print('Files:')
                print(sorted_files)
                print('Used file: ', sorted_files[-1])
                b_f = b_f[-1:]
            p.load(b_f[0])
            p.process()
            p.trial_sess.head()  # preprocessed df stored in attr. trial_sess
            samples = utils.get_electro(path=e_f, s_rate=s_rate,
                                        s_rate_eff=s_rate_eff)


if __name__ == '__main__':
    inventory()
