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
import time
import pandas as pd
VERBOSE = True


def order_by_sufix(file_list):
    """
    Sort list by sufix.

    Parameters
    ----------
    file_list : list
        list to sort.

    Returns
    -------
    sorted_list : list
        sorted list.

    """
    file_list = [os.path.basename(x) for x in file_list]
    sfx = [int(x[-6:]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


def check_evs_alignment(samples, s_rate, evs_comp, inventory, chnls=[35, 36],
                        evs='stim_ttl', offset=None):
    """
    Check distances between events found in samples and evs in evs_comp.

    Parameters
    ----------
    samples : array
        TTL signals corresponding to the last 4 channels (if there is no image).
    s_rate : int
        sampling rate of recordings.
    evs_comp : array
        reference events for comparison.
    inventory : dict
        inventory.
    chnls : list, optional
        channels to find events ([35, 36])
    evs : str, optional
        events to find ('stim_ttl')
    offset : int, optional
        if not None, offset to subtract from events (None)

    Returns
    -------
    stim_strt : list
        offset-subtracted events.
    offset : int
        offset.
    state : str
        tag indicating whether the session is ok or there are no ttls.

    """
    state = 'no_ttls'
    # get stim from ttls
    stim_strt, _, _ = ut.find_events(samples=samples, chnls=chnls,
                                     s_rate=s_rate, events=evs)
    inventory['num_'+evs][-1] = len(stim_strt)
    if len(stim_strt) > 0:
        if offset is None:
            offset = stim_strt[0]
        stim_strt -= offset
        if len(evs_comp) > 0:
            if len(evs_comp) != len(stim_strt):
                dists = np.array([np.min(np.abs(evs_comp-ttl))
                                  for ttl in stim_strt])
            else:
                dists = np.abs(evs_comp-stim_strt)
            inventory[evs+'_dists_med'][-1] = np.median(dists)
            inventory[evs+'_dists_max'][-1] = np.max(dists)
            state = 'ok'
            if VERBOSE:
                print('Median difference between start sounds')
                print(np.median(dists))
                print('Max difference between start sounds')
                print(np.max(dists))
                print('Offset')
                print(offset)
    else:
        offset = 0

    return stim_strt, offset, state


def compute_signal_stats(samples, inventory):
    """
    Compute median and std of ttl signals.

    Parameters
    ----------
    samples : array
        TTL signals corresponding to the last 4 channels (if there is no image).
    inventory : dict
        inventory.

    Returns
    -------
    None.

    """
    inventory['sgnl_stts'][-1] = [np.median(samples[:, 35:39], axis=0),
                                  np.std(samples[:, 35:39], axis=0)]


def checked(dic, session):
    """
    Check whether a session was already processed.

    Parameters
    ----------
    dic : dict
        inventory.
    session : str
        session.

    Returns
    -------
    checked : boolean
        whether a session was already processed..

    """
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
                print('Offset ', dic['offset'][indx[0]])
    return checked


def get_spks(path, limit, s_rate, offset):
    spike_times, spike_clusters, sel_clstrs, clstrs_qlt = ut.get_spikes(path=path)
    spike_times = spike_times.flatten()
    if len(spike_times) != len(spike_clusters):
        print('ERROR: Diff. lengths for spike-times and spike-clusters vectors')
        return []
    sel_clstrs = sel_clstrs[clstrs_qlt != 'noise']
    clstrs_qlt = clstrs_qlt[clstrs_qlt != 'noise']
    if len(sel_clstrs) > 0:
        clst_filt = np.array([(x, y) for x, y in zip(spike_times, spike_clusters)
                              if y in sel_clstrs])
        spks = clst_filt[:, 0]
        clsts = clst_filt[:, 1]
        if VERBOSE:
            print('--------------------------')
            print(sel_clstrs)
            print(clstrs_qlt)
        spks = spks/s_rate-offset
    else:
        spks = []
        clsts = []
    return spks, clsts, sel_clstrs, clstrs_qlt


def inventory(s_rate=3e4, s_rate_eff=2e3, redo=False, spks_sort_folder=None,
              electro_folder=None, behav_folder=None, sv_folder=None,
              sel_rats=None, sbsmpld_electr=False):
    def init_inventory(inv):
        for v in inv.values():
            v.append(np.nan)
        inv['rat'][-1] = rat_name
        inv['session'][-1] = e_f
        inv['date'][-1] = date
        inv['bhv_session'][-1] = ''

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
            inventory['state'][-1] = 'no_behavior'
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
            df_trials = p.trial_sess
        except (KeyError, IndexError):
            print('Could not load behavioral data')
            inventory['state'][-1] = 'no_behavior'
            df = None
            df_trials = None
        return df, df_trials

    def load_electro(e_f):
        try:
            if sbsmpld_electr:
                samples = np.load(sv_f_sess+'/ttls_sbsmpl.npz')
                samples = samples['samples']
                dummy_data = np.ones((samples.shape[0], 35))
                samples = np.concatenate((dummy_data, samples), axis=1)
            else:
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
                inventory['state'][-1] = 'no_electro'
                samples = None
        return samples

    def create_folder(f):
        if not os.path.exists(f):
            os.mkdir(f)
    # folders
    if spks_sort_folder is None:
        spks_sort_folder = '/archive/lbektic/AfterClustering/'
    if electro_folder is None:
        electro_folder = '/archive/rat/electrophysiology_recordings/'
    if behav_folder is None:
        behav_folder = '/archive/rat/behavioral_data/'
    if sv_folder is None:
        sv_folder = '/home/molano/fof_data/'
    # get rats from spike sorted files folder
    rats = glob.glob(spks_sort_folder+'LE*')
    if sel_rats is None:
        sel_rats = [os.path.basename(x) for x in rats]
    # load inventory or start from scratch
    if os.path.exists(sv_folder+'sess_inv.npz') and not redo:
        invtry_ref = np.load(sv_folder+'sess_inv.npz', allow_pickle=True)
        inventory = {}
        for k in invtry_ref.keys():
            inventory[k] = invtry_ref[k].tolist()
    else:
        inventory = {'rat': [], 'session': [], 'bhv_session': [], 'sgnl_stts': [],
                     'state': [], 'date': [],  'sil_per': [], 'offset': [],
                     'num_stms_csv': [], 'num_stim_analogue': [],
                     'num_stim_ttl': [], 'num_fx_ttl': [], 'num_outc_ttl': [],
                     'stim_ttl_dists_med': [], 'stim_ttl_dists_max': [],
                     'stim_analogue_dists_med': [], 'stim_analogue_dists_max': [],
                     'num_clstrs': []}
    for r in rats:
        rat_name = os.path.basename(r)
        if rat_name not in sel_rats:
            print(rat_name+' not in '+str(sel_rats))
            continue
        sv_f_rat = sv_folder+'/'+rat_name+'/'
        create_folder(sv_f_rat)
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
            sv_f_sess = sv_f_rat+'/'+os.path.basename(e_f)
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
                try:
                    df = pd.read_pickle(sv_f_sess+'/df')
                    df_trials = pd.read_pickle(sv_f_sess+'/df_trials')
                except FileNotFoundError as e:
                    print(e)
                    df, df_trials = load_behavior(b_f=b_f)
                if df is None:
                    continue
                inventory['bhv_session'][-1] = b_f
                # get start-sound times
                bhv_strt_stim_sec = ut.get_startSound_times(df=df)
                assert len(bhv_strt_stim_sec) > 0, str(len(bhv_strt_stim_sec))
                csv_offset = bhv_strt_stim_sec[0]
                bhv_strt_stim_sec -= csv_offset
                inventory['num_stms_csv'][-1] = len(bhv_strt_stim_sec)
                # load electro
                samples = load_electro(e_f=e_f)
                if samples is None:
                    continue
                sil_per = np.sum(np.std(samples, axis=1) == 0)/samples.shape[0]
                inventory['sil_per'][-1] = sil_per
                # compute signal stats
                compute_signal_stats(samples=samples, inventory=inventory)

                # pass pc-time to seconds
                csv_tms = ut.date_2_secs(df['PC-TIME'])

                # ADD EVENT TIMES TO BEHAVIORAL DATA
                # get stim ttl starts/ends
                stim_ttl_strt, offset, state =\
                    check_evs_alignment(samples=samples, s_rate=s_rate_eff,
                                        evs_comp=bhv_strt_stim_sec,
                                        inventory=inventory)
                offset = offset - csv_offset
                inventory['offset'][-1] = offset
                inventory['state'][-1] = state

                # ELECTRO DATA
                e_dict = {}
                # get stim starts from ttl signal
                stim_ttl_strt += csv_offset
                e_dict['stim_ttl_strt'] = stim_ttl_strt
                # get stims starts from analogue signal
                stim_anlg_strt, _, _ =\
                    check_evs_alignment(samples=samples, s_rate=s_rate_eff,
                                        evs_comp=stim_ttl_strt,
                                        inventory=inventory, chnls=[37, 38],
                                        evs='stim_analogue', offset=offset)
                e_dict['stim_anlg_strt'] = stim_anlg_strt
                # get fixations from ttl
                fix_strt, _, _ = ut.find_events(samples=samples, chnls=[35, 36],
                                                s_rate=s_rate_eff, events='fix')
                fix_strt -= offset
                e_dict['fix_strt'] = fix_strt
                # get outcome starts from ttl
                outc_strt, _, _ = ut.find_events(samples=samples, chnls=[35, 36],
                                                 s_rate=s_rate_eff,
                                                 events='outcome')
                outc_strt -= offset
                e_dict['outc_strt'] = outc_strt
                # add spikes
                try:
                    spks, clsts, sel_clstrs, clstrs_qlt =\
                        get_spks(path=e_f, limit=csv_tms[-1], s_rate=s_rate,
                                 offset=offset)
                except (KeyError, ValueError, FileNotFoundError) as e:
                    print(e)
                    spks = []
                    clsts = []
                    sel_clstrs = []
                    clstrs_qlt = []
                inventory['num_clstrs'][-1] = len(sel_clstrs)
                e_dict['spks'] = spks
                e_dict['clsts'] = clsts
                e_dict['sel_clstrs'] = sel_clstrs
                e_dict['clstrs_qlt'] = clstrs_qlt
                e_dict['s_rate'] = s_rate
                e_dict['s_rate_eff'] = s_rate_eff
                e_dict['code'] = 'inventory.py'
                create_folder(sv_f_sess)
                df.to_pickle(sv_f_sess+'/df')
                df_trials.to_pickle(sv_f_sess+'/df_trials')
                np.savez(sv_f_sess+'/e_data.npz', **e_dict)
                np.savez(sv_folder+'sess_inv_sbs'+str(sbsmpld_electr)+'.npz',
                         **inventory)
                if samples.shape[1] == 40:
                    smpls = samples[:, -5:-1]
                elif samples.shape[1] == 39:
                    smpls = samples[:, -4:]
                ttls_sbsmpl = {'samples': smpls}
                np.savez(sv_f_sess+'/ttls_sbsmpl.npz', **ttls_sbsmpl)


if __name__ == '__main__':
    default = True
    redo = True
    use_subsampled_electro = False
    if default:
        inventory(redo=redo, sbsmpld_electr=use_subsampled_electro)
    else:
        inventory(redo=redo,
                  spks_sort_folder='/home/molano/fof_data/AfterClustering/',
                  behav_folder='/home/molano/fof_data/behavioral_data/',
                  sv_folder='/home/molano/fof_data/', sel_rats=['LE113'])
