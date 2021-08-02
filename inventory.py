#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:16:43 2021

@author: manuel
"""

import glob
import os
import utils
import numpy as np


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    sfx = [int(x[-6:]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


def inventory(s_rate=3e4, s_rate_eff=2e3):
    spks_sort_folder = '/archive/lbektic/AfterClustering/'
    electro_folder = '/archive/rat/electrophysiology_recordings/'
    behav_folder = '/archive/rat/behavioral_data/'
    rats = glob.glob(spks_sort_folder+'LE*')
    inventory = {}
    for r in rats:
        inventory[r] = {'ok': [], 'no_behavior': [], 'no_electro': [],
                        'diff_num_events': [], 'too_much_diff': []}
        print('---------------')
        rat_num = r[r.find('/LE')+3:]
        print('Rat LE'+rat_num)
        e_fs = glob.glob(spks_sort_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
        e_fs_bis = glob.glob(electro_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
        print('Number of electro sessions:', str(len(e_fs)))
        b_f = glob.glob(behav_folder+'*'+str(rat_num))
        assert len(b_f) == 1
        path, name = os.path.split(b_f[0])
        p = utils.get_behavior(main_folder=path+'/', subject=name)
        for e_f in e_fs:
            print('-----------')
            print(e_f)
            dt_indx = e_f.find(rat_num+'_20')+len(rat_num)+1
            date = e_f[dt_indx:dt_indx+10]
            e_f_bis = [f for f in e_fs_bis if f.find(date) != -1]
            date = date.replace('-', '')
            b_f = [f for f in p.available if f.find(date) != -1]
            if len(b_f) == 0:
                print('---')
                print(date+' behavioral file not found')
                inventory[r]['no_behavior'].append(date)
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
            df = p.sess
            bhv_strt_stim_sec, _ = utils.get_startSound_times(df=df)
            bhv_strt_stim_sec -= bhv_strt_stim_sec[0]
            try:
                samples = utils.get_electro(path=e_f, s_rate=s_rate,
                                            s_rate_eff=s_rate_eff)
            except IndexError:
                print('Electro file .dat not found in '+e_f)
                if len(e_f_bis):
                    print('Trying folder '+e_f_bis[0])
                    samples = utils.get_electro(path=e_f_bis[0], s_rate=s_rate,
                                                s_rate_eff=s_rate_eff)
                else:
                    inventory[r]['no_electro'].append(date)
                    continue
            # get stim ttl starts/ends
            ttl_stim_strt, ttl_stim_end, _ =\
                utils.find_events(samples=samples, chnls=[35, 36],
                                  s_rate=s_rate_eff, events='stim_ttl')
            ttl_stim_strt -= ttl_stim_strt[0]
            if len(bhv_strt_stim_sec) != len(ttl_stim_strt):
                print('Different number of start sounds')
                print('CSV times', len(bhv_strt_stim_sec))
                print('TTL times', len(ttl_stim_strt))
                inventory[r]['diff_num_events'].append(date)
            else:
                print('Median difference between start sounds')
                print(np.median(bhv_strt_stim_sec-ttl_stim_strt))
                print('Max difference between start sounds')
                print(np.max(bhv_strt_stim_sec-ttl_stim_strt))
                if np.max(bhv_strt_stim_sec-ttl_stim_strt) > 0.1:
                    inventory[r]['too_much_diff'].append(date)
                else:
                    inventory[r]['ok'].append(date)
            np.savez(spks_sort_folder+'/inventory.npz', **inventory)


if __name__ == '__main__':
    inventory()

    # # get original stim starts/ends
    # ttl_stim_ori_strt, ttl_stim_ori_end, _ =\
    #     utils.find_events(samples=samples, chnls=[37, 38],
    #                       s_rate=s_rate_eff, events='stim_ori')
