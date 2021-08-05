#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:16:43 2021

@author: manuel
"""

import glob
import os
import utils as ut
import numpy as np


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    sfx = [int(x[-6:]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


def check_stim_starts(samples, s_rate, evs_comp, inventory):
    # load ttls
    stim_strt, _, _ = ut.find_events(samples=samples, chnls=[35, 36],
                                     s_rate=s_rate, events='stim_ttl')
    if len(stim_strt) > 0:
        inventory['offset'][-1] = stim_strt[0]
        stim_strt -= stim_strt[0]
        inventory['num_events'][-1] = [len(evs_comp), len(stim_strt)]
        if len(evs_comp) > len(stim_strt):
            dists = np.array([np.min(np.abs(evs_comp-ttl)) for ttl in stim_strt])
        elif len(evs_comp) < len(stim_strt):
            dists = np.array([np.min(np.abs(stim_strt-evs)) for evs in evs_comp])
        else:
            dists = np.abs(evs_comp-stim_strt)
        inventory['evs_dists'][-1] = [np.median(dists), np.max(dists)]
        inventory['state'].append('ok')
        print('Median difference between start sounds')
        print(np.median(dists))
        print('Max difference between start sounds')
        print(np.max(dists))
        print('Offset')
        print(inventory['offset'][-1])
    else:
        inventory['state'].append('no_ttls')
    inventory['stim_ttl'][-1] = stim_strt
    # load analogue
    stim_analogue_strt, _, _ = ut.find_events(samples=samples,
                                              chnls=[37, 38],
                                              s_rate=s_rate,
                                              events='stim_analogue')
    inventory['stim_analogue'][-1] = stim_analogue_strt


def compute_signal_stats(samples, inventory):
    inventory['signal_stats'][-1] = [np.median(samples[:, 35:39], axis=0),
                                     np.std(samples[:, 35:39], axis=0)]

def checked(dic, date):
    checked = False
    if len(dic['date']) > 0:
        indx = np.where(dic['date'] == date)[0]
        if len(indx) > 0:
            checked = True
            print('Date '+date+' already in inventory')
            print('Rat ', dic['rat'])
            print('Session ', dic['session'])
            print('Stats ', dic['signal_stats'])
            print('Offset ', dic['offset'])
    return checked


def inventory(s_rate=3e4, s_rate_eff=2e3, redo=False):
    spks_sort_folder = '/archive/lbektic/AfterClustering/'
    electro_folder = '/archive/rat/electrophysiology_recordings/'
    behav_folder = '/archive/rat/behavioral_data/'
    rats = glob.glob(spks_sort_folder+'LE*')
    if os.path.exists('/home/molano/fof/inventory.npz') and not redo:
        invtry_ref = np.load('/home/molano/fof/inventory.npz', allow_pickle=True)
        inventory = {}
        for k in invtry_ref.keys():
            inventory[k] = invtry_ref[k].tolist()
    else:
        inventory = {'sil_per': [], 'rat': [], 'session': [], 'bhv_session': [],
                     'state': [], 'date': [], 'num_events': [], 'evs_dists': [],
                     'offset': [], 'stim_ttl': [], 'stim_analogue': [],
                     'signal_stats': []}
    for r in rats:
        rat_name = os.path.basename(r)
        print('---------------')
        rat_num = r[r.find('/LE')+3:]
        print(rat_name)
        e_fs = glob.glob(spks_sort_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
        e_fs_bis = glob.glob(electro_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
        print('Number of electro sessions:', str(len(e_fs)))
        b_f = glob.glob(behav_folder+'*'+str(rat_num))
        assert len(b_f) > 0
        if len(b_f) > 1:
            print(rat_name+': several behavioral files found')
            oks = []
            for f in b_f:
                oks.append(os.path.exists(f+'/sessions'))
            assert np.sum(oks) == 1
            b_f = np.array(b_f)[np.where(np.array(oks))[0]]
            print('Used file: ', b_f[0])
        path, name = os.path.split(b_f[0])
        p = ut.get_behavior(main_folder=path+'/', subject=name)
        for e_f in e_fs:
            print('-----------')
            print(e_f)
            dt_indx = e_f.find(rat_num+'_20')+len(rat_num)+1
            date = e_f[dt_indx:dt_indx+10]
            e_f_bis = [f for f in e_fs_bis if f.find(date) != -1]
            date = date.replace('-', '')
            if not checked(dic=inventory, date=date):
                inventory['rat'].append(rat_name)
                inventory['session'].append(e_f)
                inventory['date'].append(date)
                inventory['bhv_session'].append('')
                inventory['num_events'].append([np.nan, np.nan])
                inventory['evs_dists'].append([np.nan, np.nan])
                inventory['sil_per'].append(np.nan)
                inventory['offset'].append(np.nan)
                inventory['stim_ttl'].append([])
                inventory['stim_analogue'].append([])
                inventory['signal_stats'].append([])
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
                except (KeyError, IndexError):
                    print('Could not load behavioral data')
                    inventory['state'].append('no_behavior')
                    continue
                inventory['bhv_session'][-1] = b_f[0]
                df = p.sess
                bhv_strt_stim_sec, _ = ut.get_startSound_times(df=df)
                bhv_strt_stim_sec -= bhv_strt_stim_sec[0]
                try:
                    samples = ut.get_electro(path=e_f, s_rate=s_rate,
                                             s_rate_eff=s_rate_eff)
                except IndexError:
                    print('Electro file .dat not found in '+e_f)
                    if len(e_f_bis):
                        print('Trying folder '+e_f_bis[0])
                        samples = ut.get_electro(path=e_f_bis[0],
                                                 s_rate=s_rate,
                                                 s_rate_eff=s_rate_eff)
                    else:
                        inventory['state'].append('no_electro')
                        continue
                sil_per = np.sum(np.std(samples, axis=1) == 0)/samples.shape[0]
                inventory['sil_per'][-1] = sil_per
                # get stim ttl starts/ends
                check_stim_starts(samples=samples, s_rate=s_rate_eff,
                                  evs_comp=bhv_strt_stim_sec,
                                  inventory=inventory)
                # compute signal stats
                compute_signal_stats(samples=samples, inventory=inventory)
                np.savez('/home/molano/fof/inventory.npz', **inventory)


if __name__ == '__main__':
    inventory(redo=True)

    # # get original stim starts/ends
    # ttl_stim_ori_strt, ttl_stim_ori_end, _ =\
    #     ut.find_events(samples=samples, chnls=[37, 38],
    #                       s_rate=s_rate_eff, events='stim_ori')
