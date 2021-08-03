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


def check_stim_starts(samples, chnls, s_rate, events, evnts_compare, inventory,
                      date):
    stim_strt, _, _ = utils.find_events(samples=samples, chnls=chnls,
                                        s_rate=s_rate, events=events)
    if len(evnts_compare) != len(stim_strt):
        print('Different number of start sounds')
        print('CSV times', len(evnts_compare))
        print('TTL times', len(stim_strt))
        inventory['num_events'][-1] = [len(evnts_compare), len(stim_strt)]
        inventory['offset'][-1] = [np.nan, np.nan]
    else:
        stim_strt -= stim_strt[0]
        print('Median difference between start sounds')
        print(np.median(evnts_compare-stim_strt))
        print('Max difference between start sounds')
        print(np.max(evnts_compare-stim_strt))
        inventory['num_events'][-1] = [len(evnts_compare), len(stim_strt)]
        inventory['offset'][-1] = [np.median(evnts_compare-stim_strt),
                                   np.max(evnts_compare-stim_strt)]


def checked(dic, date):
    return len(dic['date']) > 0 and (np.array(dic['date']) == date).any()


def inventory(s_rate=3e4, s_rate_eff=2e3, redo=False):
    spks_sort_folder = '/archive/lbektic/AfterClustering/'
    electro_folder = '/archive/rat/electrophysiology_recordings/'
    behav_folder = '/archive/rat/behavioral_data/'
    rats = glob.glob(spks_sort_folder+'LE*')
    try:
        invtry_ref = np.load('/home/molano/fof/inventory.npz', allow_pickle=True)
        inventory = {}
        for k in invtry_ref.keys():
            inventory[k] = invtry_ref[k].tolist()
    except FileNotFoundError:
        inventory = {'sil_per': [], 'rat': [], 'session': [], 'state': [],
                     'date': [], 'num_events': [], 'offset': []}
    for r in rats:
        rat_name = os.path.basename(r)
        print('---------------')
        rat_num = r[r.find('/LE')+3:]
        print(rat_name)
        e_fs = glob.glob(spks_sort_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
        e_fs_bis = glob.glob(electro_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
        print('Number of electro sessions:', str(len(e_fs)))
        b_f = glob.glob(behav_folder+'*'+str(rat_num))
        assert len(b_f) == 1, str(b_f)
        path, name = os.path.split(b_f[0])
        p = utils.get_behavior(main_folder=path+'/', subject=name)
        for e_f in e_fs:
            print('-----------')
            print(e_f)
            dt_indx = e_f.find(rat_num+'_20')+len(rat_num)+1
            date = e_f[dt_indx:dt_indx+10]
            e_f_bis = [f for f in e_fs_bis if f.find(date) != -1]
            date = date.replace('-', '')
            if not checked(dic=inventory, date=date) or redo:
                inventory['rat'].append(rat_name)
                inventory['session'].append(e_f)
                inventory['date'].append(date)
                inventory['num_events'].append([np.nan, np.nan])
                inventory['offset'].append([np.nan, np.nan])
                inventory['sil_per'].append(np.nan)
                b_f = [f for f in p.available if f.find(date) != -1]
                if len(b_f) == 0:
                    print('---')
                    print(date+' behavioral file not found')
                    inventory['state'].append('no_behavior')
                    continue
                elif len(b_f) > 1:
                    print('---')
                    print(date+': several behavioral files found')
                    print(b_f)
                    sorted_files = order_by_sufix(file_list=b_f)
                    print('Files:')
                    print(sorted_files)
                    print('Used file: ', sorted_files[-1])
                    b_f = [sorted_files[-1]]
                # Load behavioral data
                try:
                    p.load(b_f[0])
                    p.process()
                    p.trial_sess.head()  # preprocssd df stored in attr. trial_sess
                except KeyError:
                    print('Could not load behavioral data')
                    inventory['state'].append('no_behavior')
                    continue
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
                        samples = utils.get_electro(path=e_f_bis[0],
                                                    s_rate=s_rate,
                                                    s_rate_eff=s_rate_eff)
                    else:
                        inventory['state'].append('no_electro')
                        continue
                sil_per = np.sum(np.std(samples, axis=1) == 0)/samples.shape[0]
                inventory['sil_per'][-1] = sil_per
                # get stim ttl starts/ends
                check_stim_starts(samples=samples, chnls=[35, 36],  date=date,
                                  s_rate=s_rate_eff, events='stim_ttl',
                                  evnts_compare=bhv_strt_stim_sec,
                                  inventory=inventory)
        np.savez('/home/molano/fof/inventory.npz', **inventory)


if __name__ == '__main__':
    inventory(redo=False)

    # # get original stim starts/ends
    # ttl_stim_ori_strt, ttl_stim_ori_end, _ =\
    #     utils.find_events(samples=samples, chnls=[37, 38],
    #                       s_rate=s_rate_eff, events='stim_ori')
