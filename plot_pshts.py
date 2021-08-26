#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:26:32 2021

@author: molano
"""
import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import norm
import pandas as pd
import glob
import seaborn as sns
# import utils as ut
# from scipy.ndimage import gaussian_filter1d
# import sys
# sys.path.remove('/home/molano/rewTrained_RNNs')
import utils as ut
colors = sns.color_palette()


def plt_psths(cl, spk_tms, evs, ax, margin_psth=1000, std_conv=20, lbl='',
              color='k'):
    ax.axvline(x=0, linestyle='--', color=(.7, .7, .7))
    psth_cnv = ut.convolve_psth(spk_times=spk_tms, events=evs, std=std_conv,
                                margin=margin_psth)
    if len(psth_cnv) > 0:
        xs = np.arange(2*margin_psth)-margin_psth
        xs = xs/1000
        ax.plot(xs, psth_cnv, label=lbl+' (n: '+str(len(evs))+')', color=color)


def plot_scatter(ax, spk_tms, evs, margin_psth, color, offset=0):
    spk_raster = 1000*spk_tms
    evs = 1000*evs
    for i_ev, ev in enumerate(evs):
        spk_tmp = spk_raster-ev
        spk_tmp = spk_tmp[np.logical_and(spk_tmp > -margin_psth,
                                         spk_tmp < margin_psth)]
        ax.scatter(spk_tmp, np.ones((len(spk_tmp)))+i_ev+offset,
                   color=color, s=1)


def find_repeated_evs(evs_dists, indx_evs):
    """
    Find repeated events in indx_evs and only keeps the ones with the minimum
    distance to the associated TTL ev.

    Both vectors have the length equal to the number of csv events.

    Parameters
    ----------
    evs_dists : array
        vector containing, for each csv fixation event, the min distance to TTL ev.
    indx_evs : array
        vector containing, for each csv fix event, the index of the closest TTL ev.

    Returns
    -------
    idx_assoc_tr : array
        mask indicating which indexes of the csv vector that should be discarded
        because there is not associated TTL event.

    """
    # remove repeated events (when 2 csv evs have the same ttl event associated)
    evs_unq, indx, counts = np.unique(indx_evs, return_index=1,
                                      return_counts=1)
    idx_assoc_tr = np.ones((len(evs_dists),)) == 1
    indx_rep = np.where(counts > 1)[0]
    if len(indx_rep) > 0:  # if true, there are 2 csv evs w/ the same ttl ev assoc
        for i_r in indx_rep:
            # get indx with the same associated ttl ev
            idx_i_rep = np.where(indx_evs == evs_unq[i_r])[0]
            assert len(idx_i_rep) > 1
            # find the csv ev that is closer to the ttl ev
            min_dist_idx = np.argmin(evs_dists[idx_i_rep])
            # remove from mask the rest of indxes
            idx_assoc_tr[np.delete(idx_i_rep, min_dist_idx)] = False
    return idx_assoc_tr


def psth_choice_cond(cl, e_data, b_data, session, ax, ev='stim_ttl_strt',
                     std_conv=20, margin_psth=1000, fixtn_time=.3, evs_mrgn=1e-2):
    # pass times to seconds
    trial_times = ut.date_2_secs(b_data.fix_onset_dt)
    # get events
    events = e_data[ev]
    # remove special trials and trials when sound failed
    idx_good_trs = np.logical_and(b_data['special_trial'].values == 0,
                                  b_data['soundrfail'].values == 0)
    # look for ttl event associated to each csv fixation event
    aux = np.array([(np.min(np.abs(events-tr)), np.argmin(np.abs(events-tr)))
                    for tr in trial_times])
    evs_dists = aux[:, 0]
    indx_evs = aux[:, 1].astype(int)
    filt_evs = events[indx_evs]
    # find repeated evs
    idx_assoc_tr = find_repeated_evs(evs_dists, indx_evs)
    # get indexes of ttl events that are close to csv fixation events
    if ev == 'fix_strt':
        synchr_cond = evs_dists < evs_mrgn
    elif ev == 'stim_ttl_strt' or ev == 'stim_anlg_strt':
        synchr_cond = np.abs(evs_dists-fixtn_time) < evs_mrgn
    elif ev == 'outc_strt':
        synchr_cond = np.ones((len(evs_dists),)) == 1
    # indx of regular trials
    indx_good_evs = np.logical_and.reduce((synchr_cond, idx_good_trs,
                                           idx_assoc_tr))
    spk_tms = e_data['spks'][e_data['clsts'] == cl][:, None]
    choice = b_data['R_response'].values

    # plot psth for right trials
    indx_ch = np.logical_and(choice > .5, indx_good_evs)
    evs = filt_evs[indx_ch]
    if len(evs) > 0:
        assert len(np.unique(evs)) == len(evs), 'Repeated events!'
        plot_scatter(ax=ax[0], spk_tms=spk_tms, evs=evs, margin_psth=margin_psth,
                     color=colors[0])
        plt_psths(cl=cl, spk_tms=spk_tms, evs=evs, ax=ax[1], std_conv=std_conv,
                  margin_psth=margin_psth, lbl='Right', color=colors[0])
    spk_offset = len(evs)
    # plot psth for left trials
    indx_ch = np.logical_and(choice < .5, indx_good_evs)
    evs = filt_evs[indx_ch]
    if len(evs) > 0:
        assert len(np.unique(evs)) == len(evs), 'Repeated events!'
        plot_scatter(ax=ax[0], spk_tms=spk_tms, evs=evs, margin_psth=margin_psth,
                     color=colors[1], offset=spk_offset)
        plt_psths(cl=cl, spk_tms=spk_tms, evs=evs, ax=ax[1], std_conv=std_conv,
                  margin_psth=margin_psth, lbl='Left', color=colors[1])


if __name__ == '__main__':
    plt.close('all')
    std_conv = 20
    margin_psth = 1000
    home = 'molano'
    main_folder = '/home/'+home+'/fof_data/'
    if home == 'manuel':
        sv_folder = main_folder+'/psths/'
    elif home == 'molano':
        sv_folder = '/home/molano/Dropbox/project_Barna/FOF_project/psths/'
    inv = np.load('/home/'+home+'/fof_data/sess_inv_extended.npz', allow_pickle=1)
    sel_rats = []  # ['LE113']  # 'LE101'
    sel_sess = []  # ['LE104_2021-05-17_12-02-40']  # ['LE104_2021-06-02_13-14-24']
    # ['LE77_2020-12-04_08-27-33']  # ['LE113_2021-06-05_12-38-09']
    # file = main_folder+'/'+rat+'/sessions/'+session+'/extended_df'
    home = 'molano'
    rats = glob.glob(main_folder+'LE*')
    for r in rats:
        rat = os.path.basename(r)
        sessions = glob.glob(r+'/LE*')
        for sess in sessions:
            session = os.path.basename(sess)
            print('----')
            print(session)
            if session not in sel_sess and rat not in sel_rats and\
               (len(sel_sess) != 0 or len(sel_rats) != 0):
                continue
            idx = [i for i, x in enumerate(inv['session']) if x.endswith(session)]
            if len(idx) != 1:
                print(str(idx))
                continue
            else:
                i = idx[0]
                print(str(np.round(inv['num_stms_csv'][i], 3))+' / ' +
                      str(np.round(inv['sil_per'][i], 3))+' /// ' +
                      str(np.round(inv['num_stim_ttl'][i], 3))+' / ' +
                      str(np.round(inv['stim_ttl_dists_med'][i], 3))+' / ' +
                      str(np.round(inv['stim_ttl_dists_max'][i], 3))+' /// ' +
                      str(np.round(inv['num_stim_analogue'][i], 3))+' / ' +
                      str(np.round(inv['stim_analogue_dists_med'][i], 3))+' / ' +
                      str(np.round(inv['stim_analogue_dists_max'][i], 3)))
            e_file = sess+'/e_data.npz'
            e_data = np.load(e_file, allow_pickle=1)
            sel_clstrs = e_data['sel_clstrs']
            print(inv['sess_class'][idx[0]])
            if inv['sess_class'][idx[0]] == 'good' and len(sel_clstrs) > 0:
                b_file = sess+'/df_trials'
                b_data = pd.read_pickle(b_file)
                for i_cl, cl in enumerate(sel_clstrs):
                    cl_qlt = e_data['clstrs_qlt'][i_cl]
                    f, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 4),
                                         sharey='row')
                    if inv['stim_ttl_dists_max'][idx[0]] < 0.1:
                        ev = 'fix_strt'
                        print(ev)
                        psth_choice_cond(cl=cl, e_data=e_data, b_data=b_data,
                                         session=session, ev=ev, std_conv=std_conv,
                                         ax=ax[:, 0], margin_psth=margin_psth)
                        ev = 'stim_ttl_strt'
                        print(ev)
                        psth_choice_cond(cl=cl, e_data=e_data, b_data=b_data,
                                         session=session, ev=ev, std_conv=std_conv,
                                         ax=ax[:, 1], margin_psth=margin_psth)
                        ev = 'outc_strt'
                        print(ev)
                        psth_choice_cond(cl=cl, e_data=e_data, b_data=b_data,
                                         session=session, ev=ev, std_conv=std_conv,
                                         ax=ax[:, 2], margin_psth=margin_psth)
                    if inv['stim_analogue_dists_max'][idx[0]] < 0.1:
                        ev = 'stim_anlg_strt'
                        print(ev)
                        psth_choice_cond(cl=cl, e_data=e_data, b_data=b_data,
                                         session=session, ev=ev, std_conv=std_conv,
                                         ax=ax[:, 1], margin_psth=margin_psth)
                    ax[0, 0].set_ylabel('Trial')
                    ax[1, 0].set_ylabel('Firing rate (Hz)')
                    ax[1, 0].set_xlabel('Peri-fixation time (s)')
                    ax[1, 1].set_xlabel('Peri-stim time (s)')
                    ax[1, 2].set_xlabel('Peri-outcome time (s)')
                    for a in ax.flatten():
                        ut.rm_top_right_lines(a)
                        a.legend()
                    num_spks = np.sum(e_data['clsts'] == cl)
                    f.suptitle(str(cl)+' / #spks: '+str(num_spks) +
                               ' / qlt: '+cl_qlt)
                    f.savefig(sv_folder+cl_qlt+'/'+str(cl)+'_'+session+'.png')
                    plt.close(f)
