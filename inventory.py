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
VERBOSE = False


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    sfx = [int(x[-6:]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


def check_stim_starts(samples, s_rate, evs_comp, inventory):
    # get stim from ttls
    stim_strt, _, _ = ut.find_events(samples=samples, chnls=[35, 36],
                                     s_rate=s_rate, events='stim_ttl')
    if len(stim_strt) > 0:
        inventory['offset'][-1] = stim_strt[0]
        stim_strt -= stim_strt[0]
        # TODO: update keys!
        inventory['num_stms_csv'][-1] = len(evs_comp)
        inventory['num_stms_ttl'][-1] = len(stim_strt)
        if len(evs_comp) > len(stim_strt):
            dists = np.array([np.min(np.abs(evs_comp-ttl)) for ttl in stim_strt])
        elif len(evs_comp) < len(stim_strt):
            dists = np.array([np.min(np.abs(stim_strt-evs)) for evs in evs_comp])
        else:
            dists = np.abs(evs_comp-stim_strt)
        inventory['stms_dists_med'][-1] = np.median(dists)
        inventory['stms_dists_max'][-1] = np.max(dists)
        inventory['state'].append('ok')
        if VERBOSE:
            print('Median difference between start sounds')
            print(np.median(dists))
            print('Max difference between start sounds')
            print(np.max(dists))
            print('Offset')
            print(inventory['offset'][-1])
    else:
        inventory['state'].append('no_ttls')
    return stim_strt


def compute_signal_stats(samples, inventory):
    inventory['sgnl_stts'][-1] = [np.median(samples[:, 35:39], axis=0),
                                  np.std(samples[:, 35:39], axis=0)]


def checked(dic, session):
    checked = False
    if len(dic['session']) > 0:
        indx = np.where(np.array(dic['session']) == session)[0]
        if len(indx) > 0:
            assert len(indx) == 1
            checked = True
            if VERBOSE:
                print('Session '+session+' already in inventory')
                print('Rat ', dic['rat'][indx[0]])
                print('Session ', dic['session'][indx[0]])
                print('Stats ', dic['sgnl_stts'][indx[0]])
                print('Offset ', dic['offset'][indx[0]])
    return checked


def add_tms_to_df(df, csv_tms, ttl_tms, col):
    ttl_indx = np.searchsorted(csv_tms, ttl_tms)
    df[col] = np.nan
    df[col][ttl_indx] = 1


def add_spks_to_df(df, path, csv_tms, s_rate, offset):
    spike_times, spike_clusters, sel_clstrs, clstrs_qlt = ut.get_spikes(path=path)
    for i_cl, cl in enumerate(sel_clstrs):
        spks_cl = spike_times[spike_clusters == cl]/s_rate-offset
        add_spks_to_df(df=df, csv_tms=csv_tms, ttl_tms=spks_cl)


def inventory(s_rate=3e4, s_rate_eff=2e3, redo=False):
    def init_inventory(inv):
        for v in inv.values():
            v.append(np.nan)
        inv['rat'].append(rat_name)
        inv['session'].append(e_f)
        inv['date'].append(date)
        inv['bhv_session'].append('')

    def get_bhv_folder(bhv_f):
        assert len(bhv_f) > 0, 'No behavioral folder for rat '+rat_name
        if len(bhv_f) > 1:
            oks = []
            for f in bhv_f:
                oks.append(os.path.exists(f+'/sessions'))
            assert np.sum(oks) == 1
            bhv_f = np.array(bhv_f)[np.where(np.array(oks))[0]]
            if VERBOSE:
                print(rat_name+': several behavioral files found')
                print('Used file: ', bhv_f[0])
        return bhv_f[0]

    def get_bhv_session(b_f):
        if len(b_f) == 0:
            if VERBOSE:
                print('---')
                print(date+' behavioral file not found')
            inventory['state'].append('no_behavior')
            b_f = None
        elif len(b_f) > 1:
            sorted_files = order_by_sufix(file_list=b_f)
            b_f = sorted_files[-1]
            if VERBOSE:
                print('---')
                print(date+': several behavioral files found')
                print(b_f)
                print('Files:')
                print(sorted_files)
                print('Used file: ', sorted_files[-1])
        else:
            b_f = b_f[0]
        return b_f

    def load_behavior(b_f):
        try:
            p.load(b_f)
            p.process()
            p.trial_sess.head()  # preprocssd df stored in attr. trial_sess
            df = p.sess
        except (KeyError, IndexError):
            print('Could not load behavioral data')
            inventory['state'].append('no_behavior')
            df = None
        return df

    def load_electro(e_f):
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
                samples = None
        return samples

    # folders
    spks_sort_folder = '/archive/lbektic/AfterClustering/'
    electro_folder = '/archive/rat/electrophysiology_recordings/'
    behav_folder = '/archive/rat/behavioral_data/'
    # get rats from spike sorted files folder
    rats = glob.glob(spks_sort_folder+'LE*')
    # load inventory or start from scratch
    if os.path.exists('/home/molano/fof/sess_inv.npz') and not redo:
        invtry_ref = np.load('/home/molano/fof/sess_inv.npz', allow_pickle=True)
        inventory = {}
        for k in invtry_ref.keys():
            inventory[k] = invtry_ref[k].tolist()
    else:
        inventory = {'rat': [], 'session': [], 'bhv_session': [], 'sgnl_stts': [],
                     'state': [], 'date': [],  'sil_per': [], 'offset': [],
                     'num_stms_csv': [], 'num_stms_anlg': [],
                     'num_stms_ttl': [], 'num_fx_ttl': [], 'num_outc_ttl': [],
                     'stms_dists_med': [], 'stms_dists_max': []}
    for r in rats:
        rat_name = os.path.basename(r)
        # get rat number to look for the electro and behav folders
        rat_num = r[r.find('/LE')+3:]
        e_fs = glob.glob(spks_sort_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
        e_fs_bis = glob.glob(electro_folder+'*'+str(rat_num)+'/*'+str(rat_num)+'*')
        if VERBOSE:
            print('---------------')
            print(rat_name)
            print('Number of electro sessions:', str(len(e_fs)))
        bhv_f = glob.glob(behav_folder+'*'+str(rat_num))
        # check that the behav folder exists
        bhv_f = get_bhv_folder(bhv_f)
        path, name = os.path.split(bhv_f)
        p = ut.get_behavior(main_folder=path+'/', subject=name)
        for e_f in e_fs:
            if VERBOSE:
                print('-----------')
                print(e_f)
            dt_indx = e_f.find(rat_num+'_20')+len(rat_num)+1
            date = e_f[dt_indx:dt_indx+10]
            e_f_bis = [f for f in e_fs_bis if f.find(date) != -1]
            date = date.replace('-', '')
            if not checked(dic=inventory, session=e_f):
                # initialize entry
                init_inventory(inv=inventory)
                b_f = [f for f in p.available if f.find(date) != -1]
                # Load behavioral data
                b_f = get_bhv_session(b_f)
                if b_f is None:
                    continue
                # load load behavior
                df = load_behavior(b_f=b_f)
                if df is None:
                    continue
                inventory['bhv_session'][-1] = b_f
                # get start-sound times
                bhv_strt_stim_sec = ut.get_startSound_times(df=df)
                csv_offset = bhv_strt_stim_sec[0]
                bhv_strt_stim_sec -= csv_offset
                # load electro
                samples = load_electro(e_f=e_f)
                if samples is None:
                    continue
                sil_per = np.sum(np.std(samples, axis=1) == 0)/samples.shape[0]
                inventory['sil_per'][-1] = sil_per
                # get stim ttl starts/ends
                stim_ttl_strt = check_stim_starts(samples=samples,
                                                  s_rate=s_rate_eff,
                                                  evs_comp=bhv_strt_stim_sec,
                                                  inventory=inventory)
                # compute signal stats
                compute_signal_stats(samples=samples, inventory=inventory)
                np.savez('/home/molano/fof/sess_inv.npz', **inventory)

                csv_tms = ut.date_2_secs(df['PC-TIME'])
                # add times to bhv data
                stim_ttl_strt += csv_offset
                add_tms_to_df(df=df, csv_tms=csv_tms, ttl_tms=stim_ttl_strt,
                              col='stim_ttl_strt')
                # get stims starts from analogue signal
                stim_anlg_strt, _, _ = ut.find_events(samples=samples,
                                                      chnls=[37, 38],
                                                      s_rate=s_rate,
                                                      events='stim_analogue')
                stim_anlg_strt -= inventory['offset'][-1] - csv_offset
                add_tms_to_df(df=df, csv_tms=csv_tms, ttl_tms=stim_anlg_strt,
                              col='stim_anlg_strt')
                # get fixations from ttl
                fix_strt, _, _ = ut.find_events(samples=samples, chnls=[35, 36],
                                                s_rate=s_rate, events='fix')
                fix_strt -= inventory['offset'][-1] - csv_offset
                add_tms_to_df(df=df, csv_tms=csv_tms, ttl_tms=fix_strt,
                              col='fix_strt')
                # get outcome starts from ttl
                outc_strt, _, _ = ut.find_events(samples=samples, chnls=[35, 36],
                                                 s_rate=s_rate, events='outcome')
                outc_strt -= inventory['offset'][-1] - csv_offset
                add_tms_to_df(df=df, csv_tms=csv_tms, ttl_tms=outc_strt,
                              col='outc_strt')

                # add spikes
                add_spks_to_df(df=df, path=e_fs, csv_tms=csv_tms, s_rate=s_rate,
                               offset=inventory['offset'][-1]-csv_offset)
                df.to_pickle(e_fs+'/extended_df')


if __name__ == '__main__':
    inventory(redo=False)
